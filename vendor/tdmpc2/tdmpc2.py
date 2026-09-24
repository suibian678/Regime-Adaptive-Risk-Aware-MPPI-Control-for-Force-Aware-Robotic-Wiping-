import copy

import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		self._bc_teacher = None
		self._bc_teacher_external = False
		self._eval_ema_model = None
		self._update_counter = 0
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._force.parameters() if self.cfg.force_pred else []},
			{'params': self.model._force_regime.parameters() if self.cfg.force_regime_pred else []},
			{'params': self.model._envelope.parameters() if getattr(self.cfg, 'envelope_pred', False) else []},
			{'params': self.model._phase.parameters() if self.cfg.phase_policy else []},
			{'params': self.model._target_gate.parameters() if self.cfg.target_policy else []},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(
			self.model._pi.parameters(),
			lr=self.cfg.lr * float(getattr(self.cfg, "pi_lr_scale", 1.0)),
			eps=1e-5,
			capturable=True,
		)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	def capture_bc_teacher(self):
		"""Freeze a post-BC encoder+policy teacher for online action anchoring."""
		self._bc_teacher = copy.deepcopy(self.model).to(self.device)
		self._bc_teacher_external = False
		self._bc_teacher.eval()
		for p in self._bc_teacher.parameters():
			p.requires_grad_(False)

	def load_bc_teacher(self, fp):
		"""Load a frozen teacher from an external checkpoint.

		External teachers can come from a different random seed, so their latent
		space is not assumed to align with the current model. Policy anchoring
		therefore uses observation-space teacher queries when this flag is set.
		"""
		self._bc_teacher = copy.deepcopy(self.model).to(self.device)
		state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self._bc_teacher.state_dict(), state_dict)
		self._bc_teacher.load_state_dict(state_dict)
		self._bc_teacher_external = True
		self._bc_teacher.eval()
		for p in self._bc_teacher.parameters():
			p.requires_grad_(False)

	def capture_eval_ema(self):
		"""Initialize an evaluation EMA model from the current post-BC model."""
		self._eval_ema_model = copy.deepcopy(self.model).to(self.device)
		self._eval_ema_model.eval()
		for p in self._eval_ema_model.parameters():
			p.requires_grad_(False)

	@torch.no_grad()
	def update_eval_ema(self):
		"""Polyak-average model parameters for smoother evaluation policies."""
		if self._eval_ema_model is None:
			return
		tau = float(self.cfg.eval_ema_tau)
		for ema_p, p in zip(self._eval_ema_model.parameters(), self.model.parameters()):
			ema_p.data.mul_(tau).add_(p.data, alpha=1.0 - tau)
		for ema_b, b in zip(self._eval_ema_model.buffers(), self.model.buffers()):
			if torch.is_floating_point(ema_b):
				ema_b.data.mul_(tau).add_(b.data, alpha=1.0 - tau)
			else:
				ema_b.data.copy_(b.data)

	def _has_bc_teacher(self):
		return self._bc_teacher is not None

	@torch.no_grad()
	def _teacher_pi_mean_from_obs(self, obs, task):
		z = self._bc_teacher.encode(obs, task)
		_, info = self._bc_teacher.pi(z, task)
		return info["mean"]

	@torch.no_grad()
	def _teacher_pi_mean_from_z(self, z, task):
		_, info = self._bc_teacher.pi(z, task)
		return info["mean"]

	def phase_labels_from_obs(self, obs):
		"""Map observations to policy-expert labels.

		Default mode keeps the historical force-regime labels:
		0=low/no-contact, 1=in-band, 2=high/over-force.

		Task-stage mode is for staged BC:
		0=contact acquisition, 1=force regulation/hold, 2=forward wiping.
		"""
		force = obs[..., int(self.cfg.phase_policy_force_obs_idx)].abs()
		target = obs[..., int(self.cfg.phase_policy_target_obs_idx)].abs()
		band_fraction = float(getattr(self.cfg, "phase_policy_band_fraction", 0.0))
		if band_fraction > 0.0:
			# Per-target proportional band: keeps labeling consistent with the
			# runtime success band (target * wipe_target_force_band_fraction).
			band = target * band_fraction
		else:
			band = target.new_full(target.shape, float(self.cfg.phase_policy_band_n))
		min_force = torch.full_like(target, float(self.cfg.phase_policy_min_force_n))
		max_force = torch.full_like(target, float(self.cfg.phase_policy_max_force_n))
		low = torch.maximum(min_force, target - band)
		high = torch.minimum(max_force, target + band)
		mode = str(getattr(self.cfg, "phase_policy_label_mode", "force"))
		if mode in {"force", "force_regime"}:
			labels = torch.ones_like(force, dtype=torch.long)
			labels = torch.where(force < low, torch.zeros_like(labels), labels)
			labels = torch.where(force > high, torch.full_like(labels, 2), labels)
			return labels
		if mode in {"task_stage", "stage"}:
			progress_idx = int(getattr(self.cfg, "phase_policy_progress_obs_idx", -1))
			if 0 <= progress_idx < obs.shape[-1]:
				progress = obs[..., progress_idx].clamp(min=0)
			else:
				progress = torch.zeros_like(force)
			contact = force >= min_force
			in_band = (force >= low) & (force <= high)
			wipe_start = float(getattr(self.cfg, "phase_policy_stage_wipe_progress_start", 0.03))
			labels = torch.zeros_like(force, dtype=torch.long)
			# Once contact exists, prioritize force regulation/hold unless the
			# state is already safely in band and has started meaningful progress.
			labels = torch.where(contact, torch.ones_like(labels), labels)
			wipe_mask = contact & in_band & (progress >= wipe_start)
			labels = torch.where(wipe_mask, torch.full_like(labels, 2), labels)
			return labels
		raise ValueError(f"Unsupported phase_policy_label_mode: {mode}")

	def target_policy_labels_from_obs(self, obs):
		"""Map target force to low, mid, or high policy expert supervision."""
		target = obs[..., int(self.cfg.target_policy_target_obs_idx)].abs()
		labels = torch.ones_like(target, dtype=torch.long)
		labels = torch.where(
			target <= float(self.cfg.target_policy_low_threshold_n),
			torch.zeros_like(labels),
			labels,
		)
		labels = torch.where(
			target >= float(self.cfg.target_policy_high_threshold_n),
			torch.full_like(labels, 2),
			labels,
		)
		return labels

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)
		return

	def set_demo_plan_library(self, obs, action_seqs):
		"""Cache successful demo trajectory snippets for optional MPPI proposals."""
		if obs is None or action_seqs is None or len(action_seqs) == 0:
			self._demo_plan_obs = None
			self._demo_plan_actions = None
			return
		self._demo_plan_obs = obs.to(self.device, non_blocking=True)
		self._demo_plan_actions = action_seqs.to(self.device, non_blocking=True)

	def _sample_demo_plan_actions(self, obs):
		"""Return nearest-neighbor demo action sequences for the current observation."""
		if (
			not self.cfg.demo_plan
			or not hasattr(self, "_demo_plan_obs")
			or self._demo_plan_obs is None
			or self._demo_plan_actions is None
			or len(self._demo_plan_actions) == 0
		):
			return None
		num_trajs = min(
			int(self.cfg.demo_plan_num_trajs),
			len(self._demo_plan_actions),
			self.cfg.num_samples - self.cfg.num_pi_trajs,
		)
		if num_trajs <= 0:
			return None
		obs_dim = obs.shape[-1]
		if obs_dim > 39:
			idx = torch.tensor([self.cfg.force_obs_idx, 37, 38, 39], device=self.device)
			weights = torch.tensor([
				self.cfg.demo_plan_force_weight,
				self.cfg.demo_plan_progress_weight,
				self.cfg.demo_plan_progress_weight,
				self.cfg.demo_plan_progress_weight,
			], device=self.device)
			query = obs[0, idx]
			keys = self._demo_plan_obs[:, idx]
			dist = ((keys - query) * weights).square().sum(dim=-1)
		else:
			dist = (self._demo_plan_obs - obs[0]).square().mean(dim=-1)
		nearest = torch.topk(-dist, k=num_trajs).indices
		return self._demo_plan_actions[nearest].transpose(0, 1).contiguous()

	def _mppi_z_bounds(self, obs=None):
		"""Return planner z-action bounds, optionally governed by current contact force."""
		z_min = float(getattr(self.cfg, "mppi_action_z_min", -1.0))
		z_max = float(getattr(self.cfg, "mppi_action_z_max", 1.0))
		if not bool(getattr(self.cfg, "mppi_force_z_governor", False)) or obs is None:
			return z_min, z_max
		force_idx = int(getattr(self.cfg, "mppi_force_obs_idx", getattr(self.cfg, "force_obs_idx", 34)))
		if obs.shape[-1] <= force_idx:
			return z_min, z_max
		current_force = float(obs[0, force_idx].detach().cpu())
		if current_force < float(getattr(self.cfg, "mppi_force_low_n", 3.0)):
			return (
				float(getattr(self.cfg, "mppi_force_low_z_min", z_min)),
				float(getattr(self.cfg, "mppi_force_low_z_max", z_max)),
			)
		if current_force > float(getattr(self.cfg, "mppi_force_high_n", 10.0)):
			return (
				float(getattr(self.cfg, "mppi_force_high_z_min", z_min)),
				float(getattr(self.cfg, "mppi_force_high_z_max", z_max)),
			)
		return (
			float(getattr(self.cfg, "mppi_force_mid_z_min", z_min)),
			float(getattr(self.cfg, "mppi_force_mid_z_max", z_max)),
		)

	def _apply_mppi_action_safety(self, actions, obs=None):
		"""Apply optional planner-only action bounds before evaluating MPPI candidates."""
		if not bool(getattr(self.cfg, "mppi_action_z_clamp", False)) and not bool(getattr(self.cfg, "mppi_force_z_governor", False)):
			return actions
		z_idx = int(getattr(self.cfg, "mppi_action_z_idx", 2))
		if actions.shape[-1] <= z_idx:
			return actions
		z_min, z_max = self._mppi_z_bounds(obs)
		actions = actions.clone()
		actions[..., z_idx] = actions[..., z_idx].clamp(z_min, z_max)
		return actions

	def _apply_mppi_reactive_z(self, action, obs=None):
		"""Apply a final force-reactive z correction to the selected MPPI action."""
		if not bool(getattr(self.cfg, "mppi_force_reactive_z", False)) or obs is None:
			return action
		z_idx = int(getattr(self.cfg, "mppi_action_z_idx", 2))
		force_idx = int(getattr(self.cfg, "mppi_force_obs_idx", getattr(self.cfg, "force_obs_idx", 34)))
		if action.shape[-1] <= z_idx or obs.shape[-1] <= force_idx:
			return action
		current_force = obs[0, force_idx].detach()
		target_force = torch.as_tensor(float(getattr(self.cfg, "mppi_force_target_n", 7.5)), device=action.device)
		gain = float(getattr(self.cfg, "mppi_force_reactive_gain", 0.004))
		# Negative z presses down in the ForceWipe end-effector delta controller.
		z_cmd = -gain * (target_force - current_force.to(action.device))
		z_cmd = z_cmd.clamp(
			float(getattr(self.cfg, "mppi_force_reactive_min_z", -0.04)),
			float(getattr(self.cfg, "mppi_force_reactive_max_z", 0.03)),
		)
		action = action.clone()
		action[..., z_idx] = z_cmd
		return action

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
		model = self._eval_ema_model if eval_mode and self.cfg.eval_ema and self._eval_ema_model is not None else self.model
		z = model.encode(obs, task)
		action, info = model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task, force_target=None, model=None):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		model = self.model if model is None else model
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		force_hard_cap = float(getattr(self.cfg, "mppi_force_hard_cap", -1.0))
		unsafe = torch.zeros(self.cfg.num_samples, 1, dtype=torch.bool, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(model.reward(z, actions[t], task), self.cfg)
			pred_force = None
			if self.cfg.force_pred and (self.cfg.force_plan or force_hard_cap > 0):
				pred_force = model.force(z, actions[t], task)
			if self.cfg.force_plan and self.cfg.force_pred:
				force_err = torch.abs(pred_force - force_target)
				force_penalty = F.smooth_l1_loss(
					torch.relu(force_err - self.cfg.force_plan_deadband),
					torch.zeros_like(force_err),
					reduction='none',
				)
				reward = reward - self.cfg.force_plan_coef * force_penalty
			if force_hard_cap > 0 and pred_force is not None:
				unsafe = unsafe | (pred_force > force_hard_cap)
			z = model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (model.termination(z, task) > 0.5).float(), max=1.)
		action, _ = model.pi(z, task)
		value = G + discount * (1-termination) * model.Q(z, action, task, return_type='avg')
		if force_hard_cap > 0:
			value = value - unsafe.float() * float(getattr(self.cfg, "mppi_unsafe_penalty", 1e6))
		return value

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		model = self._eval_ema_model if eval_mode and self.cfg.eval_ema and self._eval_ema_model is not None else self.model
		z = model.encode(obs, task)
		force_target = None
		if self.cfg.force_plan and self.cfg.force_pred:
			force_target = obs[:, self.cfg.force_plan_target_obs_idx].repeat(self.cfg.num_samples, 1)
		actor_mean_actions = None
		if bool(getattr(self.cfg, "mppi_actor_mean_init", False)):
			actor_mean_actions = torch.empty(self.cfg.horizon, self.cfg.action_dim, device=self.device)
			_z_actor = z
			for t in range(self.cfg.horizon-1):
				_, actor_info = model.pi(_z_actor, task)
				actor_mean_actions[t] = actor_info["mean"][0]
				_z_actor = model.next(_z_actor, actor_mean_actions[t].unsqueeze(0), task)
			_, actor_info = model.pi(_z_actor, task)
			actor_mean_actions[-1] = actor_info["mean"][0]
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = model.pi(_z, task)
				_z = model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = model.pi(_z, task)
		demo_actions = self._sample_demo_plan_actions(obs)
		num_demo_trajs = 0 if demo_actions is None else demo_actions.shape[1]

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if actor_mean_actions is not None:
			mean.copy_(actor_mean_actions)
		elif not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		fixed_trajs = 0
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions
			fixed_trajs += self.cfg.num_pi_trajs
		if num_demo_trajs > 0:
			actions[:, fixed_trajs:fixed_trajs+num_demo_trajs] = demo_actions
			fixed_trajs += num_demo_trajs

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			num_random_trajs = self.cfg.num_samples - fixed_trajs
			if num_random_trajs > 0:
				r = torch.randn(self.cfg.horizon, num_random_trajs, self.cfg.action_dim, device=std.device)
				actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
				actions_sample = actions_sample.clamp(-1, 1)
				actions[:, fixed_trajs:] = actions_sample
			actions = self._apply_mppi_action_safety(actions, obs)
			if self.cfg.multitask:
				actions = actions * model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task, force_target=force_target, model=model).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * model._action_masks[task]
				std = std * model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		a = self._apply_mppi_reactive_z(a, obs)
		self._prev_mean.copy_(mean)
		return self._apply_mppi_action_safety(a, obs).clamp(-1, 1)

	def update_pi(
		self,
		zs,
		task,
		demo_obs=None,
		demo_action=None,
		corrective_obs=None,
		corrective_action=None,
		anchor_obs=None,
	):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_q_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss = pi_q_loss
		demo_bc_loss = torch.tensor(0., device=self.device)
		if self.cfg.demo_bc_online_coef > 0 and demo_obs is not None and demo_action is not None:
			with torch.no_grad():
				demo_z = self.model.encode(demo_obs, task=None)
			_, demo_info = self.model.pi(demo_z, task=None)
			demo_bc_loss = F.mse_loss(demo_info["mean"], demo_action)
			pi_loss = pi_loss + self.cfg.demo_bc_online_coef * demo_bc_loss
		dagger_bc_loss = torch.tensor(0., device=self.device)
		if (
			float(getattr(self.cfg, "dagger_bc_coef", 0.0)) > 0
			and corrective_obs is not None
			and corrective_action is not None
		):
			with torch.no_grad():
				corrective_z = self.model.encode(corrective_obs, task=None)
			_, corrective_info = self.model.pi(corrective_z, task=None)
			dagger_bc_loss = F.mse_loss(
				corrective_info["mean"],
				corrective_action,
			)
			pi_loss = pi_loss + self.cfg.dagger_bc_coef * dagger_bc_loss
		teacher_anchor_loss = torch.tensor(0., device=self.device)
		teacher_anchor_coef = float(getattr(self.cfg, "teacher_anchor_coef", 0.0))
		anchor_decay_updates = int(getattr(self.cfg, "teacher_anchor_decay_updates", 0))
		if anchor_decay_updates > 0:
			anchor_min_coef = float(getattr(self.cfg, "teacher_anchor_min_coef", teacher_anchor_coef))
			if anchor_min_coef >= 0.0:
				alpha = min(max(self._update_counter / max(anchor_decay_updates, 1), 0.0), 1.0)
				teacher_anchor_coef = teacher_anchor_coef + alpha * (anchor_min_coef - teacher_anchor_coef)
		if teacher_anchor_coef > 0 and self._has_bc_teacher():
			if self._bc_teacher_external and anchor_obs is not None:
				teacher_obs = anchor_obs.reshape(-1, *anchor_obs.shape[2:])
				if teacher_obs.shape[0] > int(self.cfg.teacher_anchor_batch_size):
					idx = torch.randperm(teacher_obs.shape[0], device=self.device)[:int(self.cfg.teacher_anchor_batch_size)]
					teacher_obs = teacher_obs[idx]
				with torch.no_grad():
					teacher_mean = self._teacher_pi_mean_from_obs(teacher_obs, task=None)
					# Encode without grad: the action anchor must constrain the
					# policy head only. A grad-carrying encode here leaks encoder
					# gradients past pi_optim into the next model-update step.
					teacher_z = self.model.encode(teacher_obs, task=None)
				_, teacher_anchor_info = self.model.pi(teacher_z, task=None)
				teacher_anchor_loss = F.mse_loss(teacher_anchor_info["mean"], teacher_mean)
			else:
				with torch.no_grad():
					teacher_mean = self._teacher_pi_mean_from_z(zs.detach(), task)
				teacher_anchor_loss = F.mse_loss(info["mean"], teacher_mean)
			pi_loss = pi_loss + teacher_anchor_coef * teacher_anchor_loss
		force_regime_actor_loss = torch.tensor(0., device=self.device)
		regime_actor_coef = float(getattr(self.cfg, "force_regime_actor_coef", 0.0))
		regime_warmup = int(getattr(self.cfg, "force_regime_actor_warmup_updates", 0))
		if (
			regime_actor_coef > 0
			and self.cfg.force_regime_pred
			and self._update_counter >= regime_warmup
		):
			# Freeze the classifier parameters for this forward pass while retaining
			# its differentiable action gradient into the policy.
			regime_params = list(self.model._force_regime.parameters())
			regime_requires_grad = [param.requires_grad for param in regime_params]
			for param in regime_params:
				param.requires_grad_(False)
			regime_logits = self.model.force_regime(zs, action, task)
			for param, requires_grad in zip(regime_params, regime_requires_grad):
				param.requires_grad_(requires_grad)
			safe_labels = torch.ones(
				regime_logits.shape[:-1],
				dtype=torch.long,
				device=self.device,
			)
			regime_per_step = F.cross_entropy(
				regime_logits.reshape(-1, 3),
				safe_labels.reshape(-1),
				reduction="none",
			).reshape(regime_logits.shape[:-1])
			regime_rho = torch.pow(
				self.cfg.rho,
				torch.arange(regime_per_step.shape[0], device=self.device),
			).view(-1, *([1] * (regime_per_step.ndim - 1)))
			force_regime_actor_loss = (
				(regime_per_step * regime_rho).sum()
				/ regime_rho.expand_as(regime_per_step).sum().clamp(min=1.0)
			)
			pi_loss = pi_loss + regime_actor_coef * force_regime_actor_loss
		force_band_action_loss = torch.tensor(0., device=self.device)
		force_band_action_coef = float(getattr(self.cfg, "force_band_action_coef", 0.0))
		if force_band_action_coef > 0 and anchor_obs is not None:
			force_idx = int(getattr(self.cfg, "force_band_action_force_obs_idx", getattr(self.cfg, "force_obs_idx", -1)))
			target_idx = int(getattr(self.cfg, "force_band_action_target_obs_idx", getattr(self.cfg, "force_plan_target_obs_idx", -1)))
			z_idx = int(getattr(self.cfg, "force_band_action_z_idx", 2))
			if (
				0 <= force_idx < anchor_obs.shape[-1]
				and 0 <= target_idx < anchor_obs.shape[-1]
				and 0 <= z_idx < info["mean"].shape[-1]
			):
				obs_for_pi = anchor_obs[:info["mean"].shape[0]].to(self.device)
				force = obs_for_pi[..., force_idx].abs()
				target_force = obs_for_pi[..., target_idx].abs()
				error = force - target_force
				deadband_fraction = float(getattr(self.cfg, "force_band_action_deadband_fraction", 0.0))
				if deadband_fraction > 0:
					deadband = torch.clamp(
						deadband_fraction * target_force,
						min=float(getattr(self.cfg, "force_band_action_min_deadband_n", 0.5)),
					)
				else:
					deadband = torch.full_like(
						target_force,
						max(float(getattr(self.cfg, "force_band_action_deadband_n", 2.0)), 1e-6),
					)
				gain = float(getattr(self.cfg, "force_band_action_z_gain", 0.006))
				max_abs = float(getattr(self.cfg, "force_band_action_z_max", 0.06))
				if bool(getattr(self.cfg, "force_band_action_high_only", False)):
					high_error = torch.clamp(error - deadband, min=0.0)
					desired_z = (gain * torch.clamp(error, min=0.0)).clamp(min=0.0, max=max_abs).detach()
					weight = (high_error / deadband).clamp(min=0.0, max=1.0)
				else:
					desired_z = (gain * error).clamp(min=-max_abs, max=max_abs).detach()
					weight = ((error.abs() - deadband) / deadband.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
				progress_idx = int(getattr(self.cfg, "force_band_action_progress_obs_idx", -1))
				if 0 <= progress_idx < obs_for_pi.shape[-1]:
					min_progress = float(getattr(self.cfg, "force_band_action_min_progress", 0.0))
					progress = obs_for_pi[..., progress_idx]
					weight = weight * (progress >= min_progress).float()
				contact_idx = int(getattr(self.cfg, "force_band_action_contact_obs_idx", -1))
				if 0 <= contact_idx < obs_for_pi.shape[-1]:
					min_contact = float(getattr(self.cfg, "force_band_action_min_contact", 0.5))
					contact_value = obs_for_pi[..., contact_idx]
					weight = weight * (contact_value >= min_contact).float()
				min_force_n = float(getattr(self.cfg, "force_band_action_min_force_n", -1.0))
				if min_force_n > 0:
					weight = weight * (force >= min_force_n).float()
				weighted_error = (info["mean"][..., z_idx] - desired_z).pow(2) * weight
				force_band_action_loss = weighted_error.sum() / weight.sum().clamp(min=1.0)
				pi_loss = pi_loss + force_band_action_coef * force_band_action_loss
		low_force_acquisition_action_loss = torch.tensor(0., device=self.device)
		low_force_acquisition_action_coef = float(getattr(self.cfg, "low_force_acquisition_action_coef", 0.0))
		if low_force_acquisition_action_coef > 0 and anchor_obs is not None:
			force_idx = int(getattr(self.cfg, "low_force_acquisition_force_obs_idx", getattr(self.cfg, "force_obs_idx", -1)))
			target_idx = int(getattr(self.cfg, "low_force_acquisition_target_obs_idx", getattr(self.cfg, "force_plan_target_obs_idx", -1)))
			progress_idx = int(getattr(self.cfg, "low_force_acquisition_progress_obs_idx", -1))
			contact_idx = int(getattr(self.cfg, "low_force_acquisition_contact_obs_idx", -1))
			x_idx = int(getattr(self.cfg, "low_force_acquisition_x_idx", 0))
			z_idx = int(getattr(self.cfg, "low_force_acquisition_z_idx", 2))
			if (
				0 <= force_idx < anchor_obs.shape[-1]
				and 0 <= target_idx < anchor_obs.shape[-1]
				and 0 <= x_idx < info["mean"].shape[-1]
				and 0 <= z_idx < info["mean"].shape[-1]
			):
				obs_for_pi = anchor_obs[:info["mean"].shape[0]].to(self.device)
				force = obs_for_pi[..., force_idx].abs()
				target_force = obs_for_pi[..., target_idx].abs()
				low_target = target_force <= float(getattr(self.cfg, "low_force_acquisition_target_threshold_n", 5.5))
				min_force = float(getattr(self.cfg, "low_force_acquisition_min_force_n", 3.0))
				weight = low_target.float() * (force < min_force).float()
				if 0 <= contact_idx < obs_for_pi.shape[-1]:
					max_contact = float(getattr(self.cfg, "low_force_acquisition_max_contact", 0.5))
					weight = weight * (obs_for_pi[..., contact_idx] <= max_contact).float()
				if 0 <= progress_idx < obs_for_pi.shape[-1]:
					max_progress = float(getattr(self.cfg, "low_force_acquisition_max_progress", 1.0))
					weight = weight * (obs_for_pi[..., progress_idx] <= max_progress).float()
				deficit = torch.clamp(min_force - force, min=0.0)
				base_down = abs(float(getattr(self.cfg, "low_force_acquisition_base_down_z", 0.012)))
				z_gain = abs(float(getattr(self.cfg, "low_force_acquisition_z_gain", 0.004)))
				max_down = abs(float(getattr(self.cfg, "low_force_acquisition_max_down_z", 0.026)))
				desired_z = -torch.clamp(base_down + z_gain * deficit, max=max_down).detach()
				z_loss = (info["mean"][..., z_idx] - desired_z).pow(2)
				safe_x = float(getattr(self.cfg, "low_force_acquisition_safe_x", 0.004))
				x_loss = torch.clamp(info["mean"][..., x_idx] - safe_x, min=0.0).pow(2)
				x_weight = float(getattr(self.cfg, "low_force_acquisition_x_weight", 1.0))
				low_force_acquisition_action_loss = (
					(z_loss + x_weight * x_loss) * weight
				).sum() / weight.sum().clamp(min=1.0)
				pi_loss = pi_loss + low_force_acquisition_action_coef * low_force_acquisition_action_loss
		late_forward_action_loss = torch.tensor(0., device=self.device)
		late_forward_unsafe_action_loss = torch.tensor(0., device=self.device)
		late_forward_action_coef = float(getattr(self.cfg, "late_forward_action_coef", 0.0))
		if late_forward_action_coef > 0 and anchor_obs is not None:
			force_idx = int(getattr(self.cfg, "late_forward_action_force_obs_idx", getattr(self.cfg, "force_obs_idx", -1)))
			target_idx = int(getattr(self.cfg, "late_forward_action_target_obs_idx", getattr(self.cfg, "force_plan_target_obs_idx", -1)))
			progress_idx = int(getattr(self.cfg, "late_forward_action_progress_obs_idx", -1))
			contact_idx = int(getattr(self.cfg, "late_forward_action_contact_obs_idx", -1))
			x_idx = int(getattr(self.cfg, "late_forward_action_x_idx", 0))
			if (
				0 <= force_idx < anchor_obs.shape[-1]
				and 0 <= target_idx < anchor_obs.shape[-1]
				and 0 <= progress_idx < anchor_obs.shape[-1]
				and 0 <= x_idx < info["mean"].shape[-1]
			):
				obs_for_pi = anchor_obs[:info["mean"].shape[0]].to(self.device)
				force = obs_for_pi[..., force_idx].abs()
				target_force = obs_for_pi[..., target_idx].abs()
				progress = obs_for_pi[..., progress_idx]
				force_error = (force - target_force).abs()
				min_progress = float(getattr(self.cfg, "late_forward_action_min_progress", 0.65))
				max_progress = float(getattr(self.cfg, "late_forward_action_max_progress", 0.94))
				band_fraction = float(getattr(self.cfg, "late_forward_action_force_band_fraction", 0.0))
				if band_fraction > 0:
					force_band_n = torch.clamp(
						band_fraction * target_force,
						min=float(getattr(self.cfg, "late_forward_action_min_band_n", 0.5)),
					)
				else:
					force_band_n = torch.full_like(
						target_force,
						float(getattr(self.cfg, "late_forward_action_force_band_n", 2.0)),
					)
				min_force_n = float(getattr(self.cfg, "late_forward_action_min_force_n", 3.0))
				weight = (
					(progress >= min_progress)
					& (progress <= max_progress)
					& (force >= min_force_n)
					& (force_error <= force_band_n)
				).float()
				if 0 <= contact_idx < obs_for_pi.shape[-1]:
					min_contact = float(getattr(self.cfg, "late_forward_action_min_contact", 0.5))
					weight = weight * (obs_for_pi[..., contact_idx] >= min_contact).float()
				target_x = float(getattr(self.cfg, "late_forward_action_target_x", 0.045))
				hinge = torch.clamp(target_x - info["mean"][..., x_idx], min=0.0)
				late_forward_action_loss = (hinge.pow(2) * weight).sum() / weight.sum().clamp(min=1.0)
				pi_loss = pi_loss + late_forward_action_coef * late_forward_action_loss
				unsafe_coef = float(getattr(self.cfg, "late_forward_action_unsafe_coef", 0.0))
				if unsafe_coef > 0:
					unsafe_min_progress = float(getattr(self.cfg, "late_forward_action_unsafe_min_progress", min_progress))
					unsafe_max_progress = float(getattr(self.cfg, "late_forward_action_unsafe_max_progress", max_progress))
					unsafe_band_fraction = float(getattr(self.cfg, "late_forward_action_unsafe_band_fraction", 0.0))
					if unsafe_band_fraction > 0:
						unsafe_band = torch.clamp(
							unsafe_band_fraction * target_force,
							min=float(getattr(self.cfg, "late_forward_action_min_band_n", 0.5)),
						)
						unsafe_high_force = target_force + unsafe_band
						unsafe_low_force = torch.maximum(
							torch.full_like(force, min_force_n),
							target_force - unsafe_band,
						)
					else:
						unsafe_high_force = torch.full_like(
							force,
							float(getattr(self.cfg, "late_forward_action_unsafe_high_force_n", 12.0)),
						)
						unsafe_low_force = torch.full_like(
							force,
							float(getattr(self.cfg, "late_forward_action_unsafe_low_force_n", min_force_n)),
						)
					unsafe_weight = (
						(progress >= unsafe_min_progress)
						& (progress <= unsafe_max_progress)
						& ((force >= unsafe_high_force) | (force <= unsafe_low_force))
					).float()
					if 0 <= contact_idx < obs_for_pi.shape[-1]:
						unsafe_weight = unsafe_weight * (obs_for_pi[..., contact_idx] >= 0.0).float()
					safe_x = float(getattr(self.cfg, "late_forward_action_safe_x", 0.005))
					unsafe_hinge = torch.clamp(info["mean"][..., x_idx] - safe_x, min=0.0)
					late_forward_unsafe_action_loss = (
						unsafe_hinge.pow(2) * unsafe_weight
					).sum() / unsafe_weight.sum().clamp(min=1.0)
					pi_loss = pi_loss + unsafe_coef * late_forward_unsafe_action_loss
		high_force_down_action_loss = torch.tensor(0., device=self.device)
		high_force_down_action_coef = float(getattr(self.cfg, "high_force_down_action_coef", 0.0))
		if high_force_down_action_coef > 0 and anchor_obs is not None:
			force_idx = int(getattr(self.cfg, "high_force_down_action_force_obs_idx", getattr(self.cfg, "force_obs_idx", -1)))
			progress_idx = int(getattr(self.cfg, "high_force_down_action_progress_obs_idx", -1))
			contact_idx = int(getattr(self.cfg, "high_force_down_action_contact_obs_idx", -1))
			z_idx = int(getattr(self.cfg, "high_force_down_action_z_idx", 2))
			if (
				0 <= force_idx < anchor_obs.shape[-1]
				and 0 <= progress_idx < anchor_obs.shape[-1]
				and 0 <= z_idx < info["mean"].shape[-1]
			):
				obs_for_pi = anchor_obs[:info["mean"].shape[0]].to(self.device)
				force = obs_for_pi[..., force_idx].abs()
				progress = obs_for_pi[..., progress_idx]
				min_progress = float(getattr(self.cfg, "high_force_down_action_min_progress", 0.75))
				max_progress = float(getattr(self.cfg, "high_force_down_action_max_progress", 0.98))
				high_force_n = float(getattr(self.cfg, "high_force_down_action_force_n", 15.0))
				min_contact = float(getattr(self.cfg, "high_force_down_action_min_contact", 0.5))
				min_z = float(getattr(self.cfg, "high_force_down_action_min_z", 0.0))
				weight = (
					(progress >= min_progress)
					& (progress <= max_progress)
					& (force >= high_force_n)
				).float()
				if 0 <= contact_idx < obs_for_pi.shape[-1]:
					weight = weight * (obs_for_pi[..., contact_idx] >= min_contact).float()
				down_hinge = torch.clamp(min_z - info["mean"][..., z_idx], min=0.0)
				high_force_down_action_loss = (
					down_hinge.pow(2) * weight
				).sum() / weight.sum().clamp(min=1.0)
				pi_loss = pi_loss + high_force_down_action_coef * high_force_down_action_loss
		pcgrad_active = (
			bool(getattr(self.cfg, "pcgrad_multitarget", False))
			and anchor_obs is not None
			and task is None  # single-task only; multi-task needs per-task mask logic
		)
		if pcgrad_active:
			pi_shared_loss = pi_loss - pi_q_loss
			self._pi_backward_pcgrad(pi_q_loss, pi_shared_loss, zs, rho, anchor_obs)
		else:
			pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
			"demo_bc_online_loss": demo_bc_loss,
			"dagger_bc_loss": dagger_bc_loss,
			"teacher_anchor_loss": teacher_anchor_loss,
			"force_regime_actor_loss": force_regime_actor_loss,
			"force_band_action_loss": force_band_action_loss,
			"low_force_acquisition_action_loss": low_force_acquisition_action_loss,
			"late_forward_action_loss": late_forward_action_loss,
			"late_forward_unsafe_action_loss": late_forward_unsafe_action_loss,
			"high_force_down_action_loss": high_force_down_action_loss,
		})
		return info

	@staticmethod
	def _pcgrad_project(grad_list):
		"""PCGrad gradient surgery (Yu et al., 2020, arXiv:2001.06782).

		For each task gradient pair (g_i, g_j), if they conflict (dot product < 0),
		project g_i by removing the component in the direction of g_j.  Returns
		the list of projected gradients whose sum is the final update direction.

		Args:
			grad_list: list of flat 1-D gradient tensors, one per task group.
		Returns:
			list of projected flat 1-D gradient tensors (same length).
		"""
		projected = [g.clone() for g in grad_list]
		for i in range(len(projected)):
			for j in range(len(grad_list)):
				if i == j:
					continue
				g_i = projected[i]
				g_j = grad_list[j]
				dot = (g_i * g_j).sum()
				if dot < 0:
					g_j_norm_sq = (g_j * g_j).sum().clamp(min=1e-12)
					projected[i] = g_i - (dot / g_j_norm_sq) * g_j
		return projected

	def _pi_backward_pcgrad(self, pi_q_loss, pi_shared_loss, zs, rho, anchor_obs):
		"""Backward pass for update_pi using PCGrad gradient surgery.

		The Q-maximization loss is split by target-force group.  Per-group
		gradients are PCGrad-projected before being summed.  The shared
		(BC / teacher-anchor) gradient is accumulated on top without projection.

		Args:
			pi_q_loss: Q-maximisation loss computed on the full batch (used as
				fallback when fewer than 2 groups have samples).
			pi_shared_loss: all auxiliary losses (pi_loss - pi_q_loss).
			zs: latent-state sequence (horizon+1, batch, latent_dim), detached.
			rho: discount weights over horizon steps, shape (horizon+1,).
			anchor_obs: raw observation tensor (horizon+1, batch, obs_dim).
		"""
		target_obs_idx = int(getattr(
			self.cfg, "pcgrad_target_obs_idx",
			getattr(self.cfg, "force_plan_target_obs_idx", 35),
		))
		target_values = [
			float(v)
			for v in str(getattr(self.cfg, "pcgrad_target_values", "5|8|12"))
				.replace(",", "|").split("|")
			if v.strip()
		]
		tolerance_n = float(getattr(self.cfg, "pcgrad_target_tolerance_n", 1.5))
		pi_params = list(self.model._pi.parameters())

		# Use the first time-step's target force observation to bin samples.
		target_obs_vec = anchor_obs[0, :, target_obs_idx].abs()  # (batch,)

		per_group_flat_grads = []

		for t_val in target_values:
			mask = torch.abs(target_obs_vec - t_val) <= tolerance_n
			if not mask.any():
				continue

			zs_k = zs[:, mask, :]
			action_k, info_k = self.model.pi(zs_k, task=None)
			qs_k = self.model.Q(zs_k.detach(), action_k, task=None, return_type="avg", detach=True)
			qs_k = self.scale(qs_k)
			pi_q_loss_k = (
				-(self.cfg.entropy_coef * info_k["scaled_entropy"] + qs_k)
				.mean(dim=(1, 2)) * rho
			).mean()

			# Collect gradient for this group on pi params only.
			for p in pi_params:
				if p.grad is not None:
					p.grad.zero_()
			pi_q_loss_k.backward()
			flat_k = torch.cat([
				p.grad.detach().flatten() if p.grad is not None
				else p.new_zeros(p.numel())
				for p in pi_params
			])
			per_group_flat_grads.append(flat_k)

		if len(per_group_flat_grads) >= 2:
			projected = self._pcgrad_project(per_group_flat_grads)
			total_flat = sum(projected)
		elif len(per_group_flat_grads) == 1:
			total_flat = per_group_flat_grads[0]
		else:
			# No samples matched any known target group — standard backward.
			pi_q_loss.backward(retain_graph=True)
			if pi_shared_loss.requires_grad:
				pi_shared_loss.backward()
			return

		# Write projected Q-gradient onto pi params.
		offset = 0
		for p in pi_params:
			n = p.numel()
			grad_slice = total_flat[offset:offset + n].view_as(p)
			if p.grad is None:
				p.grad = grad_slice.clone()
			else:
				p.grad.copy_(grad_slice)
			offset += n

		# Add shared (BC / teacher-anchor) loss gradients, accumulating onto pi params.
		if pi_shared_loss.requires_grad:
			pi_shared_loss.backward()

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _update(
		self,
		obs,
		action,
		reward,
		terminated,
		task=None,
		demo_obs=None,
		demo_action=None,
		corrective_obs=None,
		corrective_action=None,
	):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		if self.cfg.freeze_encoder_updates:
			with torch.no_grad():
				z = self.model.encode(obs[0], task)
		else:
			z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		if self.cfg.force_pred:
			force_preds = self.model.force(_zs, action, task)
			force_targets = obs[1:, :, self.cfg.force_obs_idx].unsqueeze(-1)
		if getattr(self.cfg, "envelope_pred", False):
			# Envelope target: max next-step force over the sampled slice's
			# remaining window (horizon steps = 150ms at 20Hz). Trained at
			# the slice root only — full window, replay distribution.
			envelope_pred = self.model.envelope(zs[0].detach() if self.cfg.freeze_encoder_updates else zs[0], action[0], task)
			envelope_target = obs[1:, :, self.cfg.force_obs_idx].max(dim=0).values.unsqueeze(-1)
		if self.cfg.force_regime_pred:
			force_regime_logits = self.model.force_regime(_zs, action, task)
			regime_force = obs[1:, :, self.cfg.force_obs_idx].abs()
			regime_target = obs[1:, :, self.cfg.force_regime_target_obs_idx].abs()
			regime_band_fraction = float(getattr(self.cfg, "force_regime_band_fraction", 0.0))
			if regime_band_fraction > 0.0:
				regime_band = regime_target * regime_band_fraction
			else:
				regime_band = regime_target.new_full(
					regime_target.shape, float(getattr(self.cfg, "force_regime_band_n", 2.0))
				)
			regime_low = torch.maximum(
				torch.full_like(regime_target, float(getattr(self.cfg, "force_regime_min_force_n", 3.0))),
				regime_target - regime_band,
			)
			regime_high = torch.minimum(
				torch.full_like(regime_target, float(getattr(self.cfg, "force_regime_max_force_n", 15.0))),
				regime_target + regime_band,
			)
			force_regime_targets = torch.ones_like(regime_force, dtype=torch.long)
			force_regime_targets = torch.where(
				regime_force < regime_low,
				torch.zeros_like(force_regime_targets),
				force_regime_targets,
			)
			force_regime_targets = torch.where(
				regime_force > regime_high,
				torch.full_like(force_regime_targets, 2),
				force_regime_targets,
			)
		if self.cfg.phase_policy:
			phase_obs = obs[:-1]
			phase_shape = phase_obs.shape[:2]
			phase_z = self.model.encode(
				phase_obs.reshape(-1, *phase_obs.shape[2:]),
				task=None,
			).reshape(*phase_shape, self.cfg.latent_dim)
			phase_logits = self.model.phase(phase_z, task)
			phase_targets = self.phase_labels_from_obs(phase_obs)
		if self.cfg.target_policy:
			target_policy_obs = obs[:-1]
			target_policy_shape = target_policy_obs.shape[:2]
			target_policy_z = self.model.encode(
				target_policy_obs.reshape(-1, *target_policy_obs.shape[2:]),
				task=None,
			).reshape(*target_policy_shape, self.cfg.latent_dim)
			target_policy_logits = self.model.target_gate(target_policy_z, task)
			target_policy_targets = self.target_policy_labels_from_obs(target_policy_obs)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.force_pred:
			rho = torch.pow(self.cfg.rho, torch.arange(self.cfg.horizon, device=self.device)).view(-1, 1, 1)
			force_error = F.smooth_l1_loss(force_preds, force_targets, reduction='none')
			if bool(getattr(self.cfg, "force_contact_balance", False)):
				weights = torch.ones_like(force_targets)
				contact_mask = force_targets >= float(getattr(self.cfg, "force_contact_threshold_n", 0.5))
				high_mask = force_targets >= float(getattr(self.cfg, "force_high_threshold_n", 10.0))
				weights = weights + contact_mask.float() * float(getattr(self.cfg, "force_contact_weight", 4.0))
				weights = weights + high_mask.float() * float(getattr(self.cfg, "force_high_weight", 8.0))
				weights = weights.clamp(max=float(getattr(self.cfg, "force_weight_clip", 12.0)))
				weighted = force_error * weights * rho
				normalizer = (weights * rho).sum().clamp(min=1.0)
				force_loss = weighted.sum() / normalizer
			else:
				force_loss = (force_error * rho).mean()
		else:
			force_loss = torch.tensor(0., device=self.device)
		if getattr(self.cfg, "envelope_pred", False):
			envelope_loss = F.smooth_l1_loss(envelope_pred, envelope_target)
		else:
			envelope_loss = torch.tensor(0., device=self.device)
		if self.cfg.force_regime_pred:
			flat_targets = force_regime_targets.reshape(-1)
			class_weights = None
			if bool(getattr(self.cfg, "force_regime_balance", True)):
				counts = torch.bincount(flat_targets, minlength=3).float()
				present = counts > 0
				class_weights = torch.zeros_like(counts)
				class_weights[present] = (
					flat_targets.numel()
					/ (present.sum().float() * counts[present])
				)
				class_weights = class_weights.clamp(
					max=float(getattr(self.cfg, "force_regime_weight_clip", 10.0))
				)
			regime_error = F.cross_entropy(
				force_regime_logits.reshape(-1, 3),
				flat_targets,
				weight=class_weights,
				reduction="none",
			).reshape(force_regime_targets.shape)
			regime_rho = torch.pow(
				self.cfg.rho,
				torch.arange(self.cfg.horizon, device=self.device),
			).view(-1, 1)
			force_regime_loss = (
				(regime_error * regime_rho).sum()
				/ regime_rho.expand_as(regime_error).sum().clamp(min=1.0)
			)
			force_regime_accuracy = (
				force_regime_logits.argmax(dim=-1) == force_regime_targets
			).float().mean()
		else:
			force_regime_loss = torch.tensor(0., device=self.device)
			force_regime_accuracy = torch.tensor(0., device=self.device)
		if self.cfg.phase_policy:
			flat_phase_targets = phase_targets.reshape(-1)
			phase_weights = None
			if bool(self.cfg.phase_policy_balance):
				phase_counts = torch.bincount(flat_phase_targets, minlength=3).float()
				phase_present = phase_counts > 0
				phase_weights = torch.zeros_like(phase_counts)
				phase_weights[phase_present] = (
					flat_phase_targets.numel()
					/ (phase_present.sum().float() * phase_counts[phase_present])
				)
				phase_weights = phase_weights.clamp(
					max=float(self.cfg.phase_policy_weight_clip)
				)
			phase_error = F.cross_entropy(
				phase_logits.reshape(-1, 3),
				flat_phase_targets,
				weight=phase_weights,
				reduction="none",
			).reshape(phase_targets.shape)
			phase_rho = torch.pow(
				self.cfg.rho,
				torch.arange(self.cfg.horizon, device=self.device),
			).view(-1, 1)
			phase_policy_loss = (
				(phase_error * phase_rho).sum()
				/ phase_rho.expand_as(phase_error).sum().clamp(min=1.0)
			)
			phase_policy_accuracy = (
				phase_logits.argmax(dim=-1) == phase_targets
			).float().mean()
		else:
			phase_policy_loss = torch.tensor(0., device=self.device)
			phase_policy_accuracy = torch.tensor(0., device=self.device)
		if self.cfg.target_policy:
			flat_target_policy_targets = target_policy_targets.reshape(-1)
			target_policy_weights = None
			if bool(self.cfg.target_policy_balance):
				target_policy_counts = torch.bincount(flat_target_policy_targets, minlength=3).float()
				target_policy_present = target_policy_counts > 0
				target_policy_weights = torch.zeros_like(target_policy_counts)
				target_policy_weights[target_policy_present] = (
					flat_target_policy_targets.numel()
					/ (target_policy_present.sum().float() * target_policy_counts[target_policy_present])
				)
				target_policy_weights = target_policy_weights.clamp(
					max=float(self.cfg.target_policy_weight_clip)
				)
			target_policy_error = F.cross_entropy(
				target_policy_logits.reshape(-1, 3),
				flat_target_policy_targets,
				weight=target_policy_weights,
				reduction="none",
			).reshape(target_policy_targets.shape)
			target_policy_rho = torch.pow(
				self.cfg.rho,
				torch.arange(self.cfg.horizon, device=self.device),
			).view(-1, 1)
			target_policy_loss = (
				(target_policy_error * target_policy_rho).sum()
				/ target_policy_rho.expand_as(target_policy_error).sum().clamp(min=1.0)
			)
			target_policy_accuracy = (
				target_policy_logits.argmax(dim=-1) == target_policy_targets
			).float().mean()
		else:
			target_policy_loss = torch.tensor(0., device=self.device)
			target_policy_accuracy = torch.tensor(0., device=self.device)
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.
		teacher_encoder_anchor_loss = torch.tensor(0., device=self.device)
		teacher_encoder_anchor_coef = float(getattr(self.cfg, "teacher_encoder_anchor_coef", 0.0))
		anchor_decay_updates = int(getattr(self.cfg, "teacher_anchor_decay_updates", 0))
		if anchor_decay_updates > 0:
			encoder_min_coef = float(getattr(self.cfg, "teacher_encoder_anchor_min_coef", teacher_encoder_anchor_coef))
			if encoder_min_coef >= 0.0:
				alpha = min(max(self._update_counter / max(anchor_decay_updates, 1), 0.0), 1.0)
				teacher_encoder_anchor_coef = teacher_encoder_anchor_coef + alpha * (encoder_min_coef - teacher_encoder_anchor_coef)
		if teacher_encoder_anchor_coef > 0 and self._has_bc_teacher():
			anchor_obs = obs[:-1].reshape(-1, *obs.shape[2:])
			if anchor_obs.shape[0] > int(self.cfg.teacher_anchor_batch_size):
				idx = torch.randperm(anchor_obs.shape[0], device=self.device)[:int(self.cfg.teacher_anchor_batch_size)]
				anchor_obs = anchor_obs[idx]
			with torch.no_grad():
				teacher_mean = self._teacher_pi_mean_from_obs(anchor_obs, task=None)
			anchor_z = self.model.encode(anchor_obs, task=None)
			_, anchor_info = self.model.pi(anchor_z, task=None)
			teacher_encoder_anchor_loss = F.mse_loss(anchor_info["mean"], teacher_mean)
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.force_coef * force_loss +
			float(getattr(self.cfg, 'envelope_coef', 0.0)) * envelope_loss +
			self.cfg.force_regime_coef * force_regime_loss +
			self.cfg.phase_policy_coef * phase_policy_loss +
			self.cfg.target_policy_coef * target_policy_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss +
			teacher_encoder_anchor_coef * teacher_encoder_anchor_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)
		self.pi_optim.zero_grad(set_to_none=True)

		# Update policy
		pi_update_every = max(1, int(getattr(self.cfg, "pi_update_every", 1)))
		do_pi_update = (self._update_counter % pi_update_every) == 0
		pi_warmup_updates = max(0, int(getattr(self.cfg, "pi_warmup_updates", 0)))
		in_pi_warmup = self._update_counter < pi_warmup_updates
		if self.cfg.freeze_pi_updates or in_pi_warmup or not do_pi_update:
			pi_info = TensorDict({
				"pi_loss": torch.tensor(0., device=self.device),
				"pi_grad_norm": torch.tensor(0., device=self.device),
				"pi_entropy": torch.tensor(0., device=self.device),
				"pi_scaled_entropy": torch.tensor(0., device=self.device),
				"pi_scale": self.scale.value,
				"demo_bc_online_loss": torch.tensor(0., device=self.device),
				"dagger_bc_loss": torch.tensor(0., device=self.device),
				"teacher_anchor_loss": torch.tensor(0., device=self.device),
				"force_regime_actor_loss": torch.tensor(0., device=self.device),
				"force_band_action_loss": torch.tensor(0., device=self.device),
				"late_forward_action_loss": torch.tensor(0., device=self.device),
				"late_forward_unsafe_action_loss": torch.tensor(0., device=self.device),
				"high_force_down_action_loss": torch.tensor(0., device=self.device),
			})
		else:
			pi_info = self.update_pi(
				zs.detach(),
				task,
				demo_obs=demo_obs,
				demo_action=demo_action,
				corrective_obs=corrective_obs,
				corrective_action=corrective_action,
				anchor_obs=obs,
			)

		# Update target Q-functions
		self.model.soft_update_target_Q()
		self.update_eval_ema()
		self._update_counter += 1

		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"force_loss": force_loss,
			"envelope_loss": envelope_loss,
			"force_regime_loss": force_regime_loss,
			"force_regime_accuracy": force_regime_accuracy,
			"phase_policy_loss": phase_policy_loss,
			"phase_policy_accuracy": phase_policy_accuracy,
			"target_policy_loss": target_policy_loss,
			"target_policy_accuracy": target_policy_accuracy,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
			"teacher_encoder_anchor_loss": teacher_encoder_anchor_loss,
		})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer, demo_batch=None, corrective_batch=None):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		if demo_batch is not None:
			kwargs["demo_obs"], kwargs["demo_action"] = demo_batch
		if corrective_batch is not None:
			kwargs["corrective_obs"], kwargs["corrective_action"] = corrective_batch
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated, **kwargs)
