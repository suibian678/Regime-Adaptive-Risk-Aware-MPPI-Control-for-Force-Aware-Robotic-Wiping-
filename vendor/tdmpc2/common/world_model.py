from copy import deepcopy

import torch
import torch.nn as nn

from common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._force = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.force_pred else None
		self._force_regime = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], 3) if cfg.force_regime_pred else None
		self._envelope = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if getattr(cfg, 'envelope_pred', False) else None
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		joint_target_phase_policy = bool(getattr(cfg, "joint_target_phase_policy", False))
		if cfg.phase_policy and cfg.target_policy and not joint_target_phase_policy:
			raise ValueError("phase_policy and target_policy are mutually exclusive unless joint_target_phase_policy=true")
		if joint_target_phase_policy and not (cfg.phase_policy and cfg.target_policy):
			raise ValueError("joint_target_phase_policy requires phase_policy=true and target_policy=true")
		use_independent_policy_experts = (
			(cfg.phase_policy and cfg.phase_policy_independent_experts)
			or (cfg.target_policy and cfg.target_policy_independent_experts)
		)
		use_policy_experts = cfg.phase_policy or cfg.target_policy
		num_policy_experts = 9 if joint_target_phase_policy else (3 if use_policy_experts else 1)
		if use_independent_policy_experts:
			self._pi = nn.ModuleList([
				layers.mlp(
					cfg.latent_dim + cfg.task_dim,
					2 * [cfg.mlp_dim],
					2 * cfg.action_dim,
				)
				for _ in range(num_policy_experts)
			])
		else:
			pi_output_dim = 2 * cfg.action_dim * num_policy_experts
			self._pi = layers.mlp(
				cfg.latent_dim + cfg.task_dim,
				2 * [cfg.mlp_dim],
				pi_output_dim,
			)
		self._phase = layers.mlp(cfg.latent_dim + cfg.task_dim, [cfg.mlp_dim], 3) if cfg.phase_policy else None
		self._target_gate = layers.mlp(cfg.latent_dim + cfg.task_dim, [cfg.mlp_dim], 3) if cfg.target_policy else None
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Force', 'Force regime', 'Termination', 'Policy prior', 'Phase gate', 'Target gate', 'Q-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._force, self._force_regime, self._termination, self._pi, self._phase, self._target_gate, self._Qs]):
			if m == self._force and not self.cfg.force_pred:
				continue
			if m == self._force_regime and not self.cfg.force_regime_pred:
				continue
			if m == self._termination and not self.cfg.episodic:
				continue
			if m == self._phase and not self.cfg.phase_policy:
				continue
			if m == self._target_gate and not self.cfg.target_policy:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)

	def force(self, z, a, task):
		"""
		Predicts the next-step contact force for force-aware tasks.
		"""
		assert self._force is not None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._force(z)

	def envelope(self, z, a, task):
		"""
		Predicts the max normal force over the next `horizon` steps (the
		transient envelope) given the current latent state and the action
		about to be taken. Pre-impact regression, not a forward rollout.
		"""
		assert self._envelope is not None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._envelope(z)

	def force_regime(self, z, a, task):
		"""
		Predicts whether next-step force is below, inside, or above the task band.
		"""
		assert self._force_regime is not None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._force_regime(z)
	
	def termination(self, z, task, unnormalized=False):
		"""
		Predicts termination signal.
		"""
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))
		

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior. Optional gates are learned from latent state
		# supervision and detached from the actor objective.
		phase_logits = None
		phase_probs = None
		target_logits = None
		target_probs = None
		joint_target_phase_policy = bool(getattr(self.cfg, "joint_target_phase_policy", False))
		use_independent_policy_experts = (
			(self.cfg.phase_policy and self.cfg.phase_policy_independent_experts)
			or (self.cfg.target_policy and self.cfg.target_policy_independent_experts)
		)
		if use_independent_policy_experts:
			pi_output = torch.stack([expert(z) for expert in self._pi], dim=-2)
		else:
			pi_output = self._pi(z)
		if joint_target_phase_policy:
			if use_independent_policy_experts:
				experts = pi_output.reshape(*pi_output.shape[:-2], 3, 3, 2 * self.cfg.action_dim)
			else:
				experts = pi_output.reshape(*pi_output.shape[:-1], 3, 3, 2 * self.cfg.action_dim)
			target_logits = self._target_gate(z)
			target_probs = torch.softmax(
				target_logits / float(self.cfg.target_policy_temperature),
				dim=-1,
			).detach()
			if bool(self.cfg.target_policy_hard_gate):
				target_probs = torch.nn.functional.one_hot(
					target_probs.argmax(dim=-1),
					num_classes=3,
				).to(experts.dtype)
			phase_logits = self._phase(z)
			phase_probs = torch.softmax(
				phase_logits / float(self.cfg.phase_policy_temperature),
				dim=-1,
			).detach()
			if bool(self.cfg.phase_policy_hard_gate):
				phase_probs = torch.nn.functional.one_hot(
					phase_probs.argmax(dim=-1),
					num_classes=3,
				).to(experts.dtype)
			joint_probs = target_probs.unsqueeze(-1) * phase_probs.unsqueeze(-2)
			pi_output = (experts * joint_probs.unsqueeze(-1)).sum(dim=(-3, -2))
		elif self.cfg.target_policy:
			if self.cfg.target_policy_independent_experts:
				experts = pi_output
			else:
				experts = pi_output.reshape(*pi_output.shape[:-1], 3, 2 * self.cfg.action_dim)
			target_logits = self._target_gate(z)
			target_probs = torch.softmax(
				target_logits / float(self.cfg.target_policy_temperature),
				dim=-1,
			).detach()
			if bool(self.cfg.target_policy_hard_gate):
				target_probs = torch.nn.functional.one_hot(
					target_probs.argmax(dim=-1),
					num_classes=3,
				).to(experts.dtype)
			pi_output = (experts * target_probs.unsqueeze(-1)).sum(dim=-2)
		elif self.cfg.phase_policy:
			if self.cfg.phase_policy_independent_experts:
				experts = pi_output
			else:
				experts = pi_output.reshape(*pi_output.shape[:-1], 3, 2 * self.cfg.action_dim)
			phase_logits = self._phase(z)
			phase_probs = torch.softmax(
				phase_logits / float(self.cfg.phase_policy_temperature),
				dim=-1,
			).detach()
			if bool(self.cfg.phase_policy_hard_gate):
				phase_probs = torch.nn.functional.one_hot(
					phase_probs.argmax(dim=-1),
					num_classes=3,
				).to(experts.dtype)
			pi_output = (experts * phase_probs.unsqueeze(-1)).sum(dim=-2)
		mean, log_std = pi_output.chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask: # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		if phase_logits is not None:
			info["phase_logits"] = phase_logits
			info["phase_probs"] = phase_probs
		if target_logits is not None:
			info["target_logits"] = target_logits
			info["target_probs"] = target_probs
		return action, info

	def target_gate(self, z, task):
		"""Predict low, mid, or high target-force policy expert from latent state."""
		assert self._target_gate is not None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		return self._target_gate(z)

	def phase(self, z, task):
		"""Predict low-force, in-band, or high-force phase from latent state."""
		assert self._phase is not None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		return self._phase(z)

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2
