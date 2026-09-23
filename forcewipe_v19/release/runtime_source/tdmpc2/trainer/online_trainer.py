import csv
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from trainer.base import Trainer

# Python-side env/wrapper counters that `get_state`/`set_state` do not cover.
# The `_best_*` counters are monotonic (wipe_progress IS _best_contact_progress),
# so restoring them is required for one-step candidate labeling: without it, a
# candidate step leaks its progress into every later candidate and into the
# ongoing episode (V344 audit fix). Module-level because tool scripts bind the
# snapshot methods onto shim namespaces without class attributes.
FORCE_WIPE_SNAPSHOT_ATTRS = (
	"_elapsed_steps",
	"_t",
	"_best_contact_progress",
	"_best_force_band_progress",
	"_force_phase_potential",
	"_force_filtered",
	"_force_delta",
	"_force_history",
	"_force_obs_initialized",
)


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._eval_diag_path = Path(self.cfg.work_dir) / "eval_diagnostics.csv"
		self._update_diag_path = Path(self.cfg.work_dir) / "update_diagnostics.csv"
		self._next_update_diag_step = 0
		self._bc_retention_path = Path(self.cfg.work_dir) / "bc_retention.csv"
		self._wipe_curriculum = self._parse_wipe_curriculum_schedule()
		self._wipe_curriculum_stage = None
		self._demo_plan_library_size = 0
		self._last_demo_blend_alpha = 0.0
		self._demo_trajectory_library_size = 0
		self._active_demo_trajectory = None
		self._active_demo_cursor = 0
		self._last_demo_trajectory_alpha = 0.0
		self._demo_reset_states = []
		self._demo_reset_count = 0
		self._dagger_obs = None
		self._dagger_action = None
		self._dagger_regime = None
		self._dagger_size = 0
		self._dagger_seen = 0
		self._dagger_phase_seen = np.zeros(3, dtype=np.int64)
		self._dagger_action_mse_sum = 0.0
		self._dagger_teacher_phase = "move_to_start"
		self._dagger_teacher_filtered_force = 0.0
		self._dagger_teacher_last_progress = 0.0
		self._best_eval_success = -np.inf
		self._best_eval_stage = -1
		self._hybrid_floor_filtered = None
		self._hybrid_floor_integral = 0.0
		self._best_eval_path = Path(self.cfg.work_dir) / "best_eval.csv"

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	@staticmethod
	def _scalar(value, default=np.nan):
		if value is None:
			return default
		if torch.is_tensor(value):
			return float(value.detach().cpu().flatten()[0])
		return float(value)

	@staticmethod
	def _to_numpy_obs(obs):
		if torch.is_tensor(obs):
			obs = obs.detach().cpu().numpy().astype(np.float32)
		else:
			obs = np.asarray(obs, dtype=np.float32)
		if obs.ndim == 2 and obs.shape[0] == 1:
			obs = obs[0]
		return obs

	def _write_csv_row(self, path, row):
		"""Append diagnostics with a stable header independent of the main logger."""
		path.parent.mkdir(parents=True, exist_ok=True)
		write_header = not path.exists()
		with path.open("a", newline="", encoding="utf-8") as f:
			writer = csv.DictWriter(f, fieldnames=list(row.keys()))
			if write_header:
				writer.writeheader()
			writer.writerow(row)

	@torch.no_grad()
	def _demo_bc_mse(self):
		"""Measure whether the policy prior is drifting away from scripted demos."""
		if not hasattr(self, "_demo_tds"):
			return np.nan
		self._prepare_demo_bc_data()
		if len(self._demo_action) == 0:
			return np.nan
		total_loss, total_count = 0.0, 0
		batch_size = min(512, len(self._demo_action))
		self.agent.model.eval()
		for start in range(0, len(self._demo_action), batch_size):
			end = min(start + batch_size, len(self._demo_action))
			obs = self._demo_obs[start:end]
			action = self._demo_action[start:end]
			z = self.agent.model.encode(obs, task=None)
			_, info = self.agent.model.pi(z, task=None)
			loss = F.mse_loss(info["mean"], action, reduction="sum")
			total_loss += float(loss.detach().cpu())
			total_count += int(action.numel())
		return total_loss / max(total_count, 1)

	@torch.no_grad()
	def _demo_bc_mse_by_phase(self):
		"""Measure policy drift separately in low, in-band, and high-force states."""
		if not bool(getattr(self.cfg, "phase_policy", False)):
			return (np.nan, np.nan, np.nan)
		if not hasattr(self, "_demo_tds"):
			return (np.nan, np.nan, np.nan)
		self._prepare_demo_bc_data()
		if len(self._demo_action) == 0:
			return (np.nan, np.nan, np.nan)
		phase_loss = np.zeros(3, dtype=np.float64)
		phase_count = np.zeros(3, dtype=np.int64)
		batch_size = min(512, len(self._demo_action))
		self.agent.model.eval()
		for start in range(0, len(self._demo_action), batch_size):
			end = min(start + batch_size, len(self._demo_action))
			obs = self._demo_obs[start:end]
			action = self._demo_action[start:end]
			labels = self.agent.phase_labels_from_obs(obs)
			z = self.agent.model.encode(obs, task=None)
			_, info = self.agent.model.pi(z, task=None)
			sample_loss = F.mse_loss(
				info["mean"], action, reduction="none"
			).mean(dim=-1)
			for phase in range(3):
				mask = labels == phase
				if mask.any():
					phase_loss[phase] += float(sample_loss[mask].sum().detach().cpu())
					phase_count[phase] += int(mask.sum().detach().cpu())
		return tuple(
			phase_loss[i] / phase_count[i] if phase_count[i] else np.nan
			for i in range(3)
		)

	def _failure_reason(self, info):
		"""Classify final eval failure using task-specific success conditions."""
		if self._scalar(info.get("success"), 0.0) >= 0.5:
			return "success"
		if self._scalar(info.get("fail"), 0.0) >= 0.5:
			return "force_safety_fail"
		if self.cfg.task != "force-wipe":
			raw_env = self.env.unwrapped
			normal_force = self._scalar(info.get("normal_force"))
			force_error = self._scalar(info.get("force_error"))
			xy_error = self._scalar(info.get("xy_error"))
			if normal_force > float(getattr(raw_env, "max_safe_force_n", np.inf)):
				return "too_much_contact_force"
			if force_error > float(getattr(raw_env, "force_success_band_n", np.inf)):
				return "force_error_too_large"
			if xy_error >= 0.06:
				return "xy_error_too_large"
			return "unknown"
		progress = self._scalar(info.get("wipe_progress"))
		normal_force = self._scalar(info.get("normal_force"))
		force_error = self._scalar(info.get("force_error"))
		y_error = self._scalar(info.get("y_error"))
		raw_env = self.env.unwrapped
		if progress <= float(getattr(raw_env, "success_progress_threshold", 0.92)):
			return "progress_below_threshold"
		if normal_force < float(raw_env.min_wipe_force_n):
			return "not_enough_contact_force"
		if normal_force > float(raw_env.max_safe_force_n):
			return "too_much_contact_force"
		if force_error > float(raw_env.force_success_band_n):
			return "force_error_too_large"
		if y_error >= 0.045:
			return "y_error_too_large"
		return "unknown"

	def _parse_wipe_curriculum_schedule(self):
		"""Parse 'step:path_end_x:success_progress' curriculum stages."""
		schedule = str(getattr(self.cfg, "wipe_curriculum_schedule", "") or "").strip()
		if self.cfg.task != "force-wipe" or not schedule:
			return []
		stages = []
		for item in schedule.replace("|", ",").split(","):
			parts = [p.strip() for p in item.split(":")]
			if len(parts) != 3:
				raise ValueError(
					"wipe_curriculum_schedule entries must be step:path_end_x:success_progress"
				)
			stages.append((int(parts[0]), float(parts[1]), float(parts[2])))
		stages.sort(key=lambda x: x[0])
		return stages

	def _apply_wipe_curriculum(self):
		"""Update ForceWipe path length and success threshold for the current training step."""
		if not self._wipe_curriculum:
			return
		stage_idx = 0
		for idx, (start_step, _, _) in enumerate(self._wipe_curriculum):
			if self._step >= start_step:
				stage_idx = idx
		if self._wipe_curriculum_stage == stage_idx:
			return
		start_step, path_end_x, success_progress = self._wipe_curriculum[stage_idx]
		raw_env = self.env.unwrapped
		raw_env.path_end_xy = (path_end_x, float(raw_env.path_end_xy[1]))
		raw_env.success_progress_threshold = success_progress
		self._wipe_curriculum_stage = stage_idx
		print(
			" curriculum      "
			f"I: {self._step:,} stage: {stage_idx} "
			f"start: {start_step} path_end_x: {path_end_x:.3f} "
			f"success_progress: {success_progress:.3f}"
		)

	def _reset_hybrid_floor(self):
		self._hybrid_floor_filtered = None
		self._hybrid_floor_integral = 0.0
		self._hybrid_floor_prev_raw = None
		# V438 continuous-preload state (inactive unless hybrid_floor_preload)
		self._hybrid_floor_steady_steps = 0
		self._hybrid_floor_preload_authority = 0.0
		self._hybrid_floor_steps_no_contact = 10**6
		self._hybrid_floor_had_contact = False
		self._hybrid_floor_x_limit = 1.0
		self._hybrid_floor_step_count = 0
		self._hybrid_floor_contact_t = None
		self._hybrid_floor_prev_tcp_x = None
		self._hybrid_floor_last_x_cmd = 0.0
		self._hybrid_floor_x_gain_ema = None
		self._hybrid_floor_f_hist = None
		self._hybrid_floor_alarm_ttl = 0
		self._hybrid_floor_hover_steps = 0
		self._hybrid_floor_hover_boost = 0.0
		self._hybrid_floor_prev_gap = None

	def _apply_hybrid_force_floor(self, obs, action):
		"""Hybrid force/position floor: the servo owns z, the policy owns x/y.

		Reactive-only by design (V424: contact-force transients are not
		predictable from observations): pre-contact z follows the z-gap
		approach governor; in contact z is a PI servo on observed force toward
		the target force, so contact retention and force-band tracking are the
		same controller. All z commands are clamped by the validated per-step
		rate limits; the over-force lift override stays on top.
		"""
		if not bool(getattr(self.cfg, "hybrid_force_floor", False)):
			return action
		if self.cfg.task != "force-wipe" or isinstance(obs, dict):
			return action
		obs_flat = obs.detach().flatten()
		if obs_flat.numel() <= self.cfg.force_plan_target_obs_idx:
			return action
		raw_env = self.env.unwrapped
		normal_force = float(obs_flat[self.cfg.force_obs_idx].cpu())
		target_force = float(obs_flat[self.cfg.force_plan_target_obs_idx].cpu())
		action = action.clone()
		self._hybrid_floor_step_count = getattr(self, "_hybrid_floor_step_count", 0) + 1

		alpha = float(getattr(self.cfg, "hybrid_floor_filter_alpha", 0.65))
		prev = getattr(self, "_hybrid_floor_filtered", None)
		prev_raw = getattr(self, "_hybrid_floor_prev_raw", None)
		filtered = normal_force if prev is None else alpha * prev + (1.0 - alpha) * normal_force
		self._hybrid_floor_filtered = filtered
		force_rate = 0.0 if prev_raw is None else normal_force - prev_raw
		self._hybrid_floor_prev_raw = normal_force
		# Safety-side force estimate: the filter lags on impact transients
		# (V424: they move ~7N/step), so every descent/lift decision uses the
		# most pessimistic of filtered and raw.
		safety_force = max(filtered, normal_force)

		tcp_z = self._scalar(raw_env.agent.tcp.pose.p[:, 2])
		pad_top_z = self._scalar(raw_env.wipe_pad.pose.p[:, 2]) + float(raw_env.pad_half_size[2])
		z_gap = tcp_z - pad_top_z

		contact_n = float(getattr(self.cfg, "hybrid_floor_contact_force_n", 0.5))
		gap_contact = float(getattr(self.cfg, "hybrid_floor_gap_contact_m", 0.013))
		max_lift = float(self.cfg.force_safe_max_lift)
		# Reactive contact gate on x: inside the wipe region, do not advance
		# while below the minimum wipe force — wait for the servo to reacquire.
		if bool(getattr(self.cfg, "hybrid_floor_contact_gate_x", True)):
			tcp_x = self._scalar(raw_env.agent.tcp.pose.p[:, 0])
			start_x = float(raw_env.path_start_xy[0])
			end_x = float(raw_env.path_end_xy[0])
			tcp_prog = (tcp_x - start_x) / max(end_x - start_x, 1e-6)
			gate_start = float(getattr(self.cfg, "hybrid_floor_contact_gate_progress", 0.02))
			if tcp_prog >= gate_start and filtered < float(raw_env.min_wipe_force_n):
				action[..., 0] = torch.clamp(action[..., 0], max=0.0)
		# V442 dip-precursor alarm (config-gated, default OFF): a linear
		# logistic on physical observables (force history, rate, pad lateral
		# state — all inside the 40-dim observation), calibrated on DEV and
		# FROZEN. Fires 1-2 steps before an impact-recoil contact loss
		# (Step-1: linear AUC 0.87 vs reactive rate-spike 0.71). Response is
		# purely classical: brief x slowdown + a recoil-catch descent floor;
		# every safety clamp (taper, re-acq guard, cap guards, pessimistic
		# safety_force) applies ON TOP, unchanged.
		alarm_on = bool(getattr(self.cfg, "hybrid_floor_alarm", False))
		alarm_active = False
		if alarm_on:
			hist = getattr(self, "_hybrid_floor_f_hist", None)
			if hist is None:
				hist = [normal_force] * 4
			hist = [normal_force] + hist[:3]
			self._hybrid_floor_f_hist = hist
			pad_p = raw_env.wipe_pad.pose.p.detach().cpu().numpy()[0]
			pad_v = raw_env.wipe_pad.linear_velocity.detach().cpu().numpy()[0]
			mu_n = float(getattr(self.cfg, "hybrid_floor_alarm_mu", 0.276))
			ratio = 180.0 * float(np.hypot(pad_p[0], pad_p[1])) / max(normal_force, 0.5) / max(mu_n, 1e-6)
			last_x = float(getattr(self, "_hybrid_floor_last_x_cmd", 0.0))
			recoil = -np.sign(last_x + 1e-9) * float(pad_v[0])
			feats = np.array(hist + [force_rate, ratio, recoil,
									 float(pad_p[0]), float(pad_v[0]), last_x])
			wstr = str(getattr(self.cfg, "hybrid_floor_alarm_w", "") or "")
			ws = [float(v) for v in wstr.replace(",", "|").split("|") if v.strip()]
			if len(ws) == 11:
				score = float(np.dot(ws[:10], feats) + ws[10])
				if score > float(getattr(self.cfg, "hybrid_floor_alarm_threshold", 0.0)):
					self._hybrid_floor_alarm_ttl = int(getattr(self.cfg, "hybrid_floor_alarm_window", 3))
			ttl = getattr(self, "_hybrid_floor_alarm_ttl", 0)
			alarm_active = ttl > 0
			if alarm_active:
				self._hybrid_floor_alarm_ttl = ttl - 1

		# V438 continuous-preload extension (config-gated; defaults preserve
		# the historical floor exactly). Diagnosis (diag_v438): losses are
		# (a) a sub-3N equilibrium trap created by the re-acquisition cap
		# (steady ~2.2N < 3N contact line, never crosses the 5N release), and
		# (b) impact-recoil bounces while descending with force rising fast.
		preload_on = bool(getattr(self.cfg, "hybrid_floor_preload", False))
		if preload_on:
			if normal_force < contact_n:
				self._hybrid_floor_steps_no_contact = getattr(
					self, "_hybrid_floor_steps_no_contact", 10**6) + 1
			else:
				self._hybrid_floor_steps_no_contact = 0
				if normal_force >= float(raw_env.min_wipe_force_n):
					if not getattr(self, "_hybrid_floor_had_contact", False):
						self._hybrid_floor_contact_t = getattr(self, "_hybrid_floor_step_count", 0)
					self._hybrid_floor_had_contact = True

		# V444 (lifted-setting support): per-step z rate scale for finer
		# control rates — wall-clock z speeds are preserved by scaling the
		# per-step commands; default 1.0 leaves 20Hz behavior untouched.
		zrs = float(getattr(self.cfg, "hybrid_floor_z_rate_scale", 1.0))
		if normal_force < contact_n or z_gap > gap_contact:
			# Pre-contact: validated z-gap approach governor (V406c/V416),
			# passed through unclamped — its steps are the approach rate limit.
			z_cmd = self._force_wipe_approach_z_from_gap(z_gap) * zrs
			self._hybrid_floor_integral = 0.0
			if preload_on:
				# Steady-contact authority resets on any full contact loss.
				self._hybrid_floor_steady_steps = 0
				self._hybrid_floor_preload_authority = 0.0
				lost_window = int(getattr(self.cfg, "hybrid_floor_lost_window", 8))
				lost_down = float(getattr(self.cfg, "hybrid_floor_lost_reapproach_down", 0.006))
				ref_t_l = float(getattr(self.cfg, "hybrid_floor_reacq_ref_target", 0.0))
				if ref_t_l > 0:
					lost_down = lost_down * min(1.0, ref_t_l / max(target_force, 1e-6))
				if getattr(self, "_hybrid_floor_had_contact", False):
					# Post-bounce: gentler-than-historical re-approach for a
					# short window (V428 measured +14.5N re-impacts from free
					# bounces under −0.010; adversarial x/y proposals reach
					# these states), then the normal governor reacquires.
					# First approach of the episode is NOT clamped (a −0.020
					# first-approach boost was tried and produced 23N
					# first-impact peaks — reverted).
					if getattr(self, "_hybrid_floor_steps_no_contact", 10**6) <= lost_window and z_gap <= 0.025:
						z_cmd = max(z_cmd, -lost_down)
				# Hover breaker: the plant tracks small z deltas weakly, so a
				# constant −0.010 near-gap approach can hover above the pad
				# for a whole episode. Escalate the approach ONLY when the
				# gap is demonstrably not closing (realized velocity ≈ 0, so
				# escalation adds no impact energy); de-escalate the moment
				# the gap starts closing. FIRST ACQUISITION ONLY: from a
				# post-bounce state the recoiling pad fakes a "stuck gap"
				# and the escalation slams the re-impact (measured 19.9N
				# peaks, 13/25 over-force) — post-loss states keep the
				# gentle lost-window clamp instead.
				hover_after = int(getattr(self.cfg, "hybrid_floor_hover_steps", 6))
				hover_step = float(getattr(self.cfg, "hybrid_floor_hover_escalate", 0.002))
				hover_max = float(getattr(self.cfg, "hybrid_floor_hover_max_down", 0.024))
				if (hover_after > 0 and z_gap <= gap_contact
						and not getattr(self, "_hybrid_floor_had_contact", False)):
					prev_gap = getattr(self, "_hybrid_floor_prev_gap", None)
					closing = prev_gap is not None and (prev_gap - z_gap) > 0.0008
					if closing:
						self._hybrid_floor_hover_steps = 0
						self._hybrid_floor_hover_boost = 0.0
					else:
						self._hybrid_floor_hover_steps = getattr(self, "_hybrid_floor_hover_steps", 0) + 1
						if self._hybrid_floor_hover_steps >= hover_after:
							self._hybrid_floor_hover_boost = min(
								getattr(self, "_hybrid_floor_hover_boost", 0.0) + hover_step,
								hover_max - 0.010,
							)
					boost = getattr(self, "_hybrid_floor_hover_boost", 0.0)
					if boost > 0:
						z_cmd = min(z_cmd, -(0.010 * zrs + boost))
				else:
					self._hybrid_floor_hover_steps = 0
					self._hybrid_floor_hover_boost = 0.0
			self._hybrid_floor_prev_gap = z_gap
		else:
			self._hybrid_floor_prev_gap = z_gap
			if preload_on:
				self._hybrid_floor_hover_steps = 0
				self._hybrid_floor_hover_boost = 0.0
			# In contact: PI servo on observed force toward the target force.
			kp = float(getattr(self.cfg, "hybrid_floor_kp", 0.004))
			ki = float(getattr(self.cfg, "hybrid_floor_ki", 0.0002))
			i_clip = float(getattr(self.cfg, "hybrid_floor_integral_clip", 30.0))
			# V438 setpoint bias: track a point below the target inside the
			# band, leaving headroom for the wipe's natural force spikes —
			# the spikes then stay inside the band instead of triggering
			# recoil unloads. Smoothly parameterized in units of the
			# per-target band (no per-tier constants).
			bias = float(getattr(self.cfg, "hybrid_floor_setpoint_bias_band", 0.0)) if preload_on else 0.0
			band_n_i = float(getattr(raw_env, "force_success_band_n", 2.0))
			servo_target = target_force - bias * band_n_i
			# V443-F1 (config-gated): when the success band STRADDLES the 3N
			# contact line (light targets), tracking the raw target parks the
			# force astride the line — a metric-geometry contact ceiling of
			# ~0.5. Serve the upper half of the band instead: still in band,
			# above the contact line. Smooth in target (only affects targets
			# with target − band < min_wipe; 5/8/12N unchanged).
			if preload_on and bool(getattr(self.cfg, "hybrid_floor_servo_target_floor", False)):
				# Oscillation-margin form: bias the setpoint up by half the
				# deficit between (contact line + the ~2N 20Hz force-noise
				# floor, V424) and the band's lower edge, clamped to the
				# upper half of the band. One physical constant, smooth in
				# target; no-op for tiers whose band clears the margin.
				osc_n = float(getattr(self.cfg, "hybrid_floor_osc_margin_n", 2.0))
				mw_sp = float(raw_env.min_wipe_force_n)
				deficit = max(0.0, (mw_sp + osc_n) - (target_force - band_n_i))
				servo_target = max(servo_target, min(target_force + 0.5 * band_n_i,
													  target_force + 0.5 * deficit))
			if preload_on and getattr(self, "_hybrid_floor_steady_steps", 0) >= int(
					getattr(self.cfg, "hybrid_floor_preload_min_steps", 3)):
				low_edge_i = max(float(raw_env.min_wipe_force_n), target_force - band_n_i)
				if normal_force < low_edge_i:
					# V438: proportional droop leaves the servo hovering
					# below the band (kp·error balances the plant's
					# command→force map around ~4N). In verified steady
					# contact below the band the integral may wind further —
					# the COMMAND stays inside the unchanged taper/cap
					# limits, so safety authority is untouched.
					i_clip = float(getattr(self.cfg, "hybrid_floor_preload_i_clip", 90.0))
			error = filtered - servo_target
			self._hybrid_floor_integral = float(
				np.clip(self._hybrid_floor_integral + error, -i_clip, i_clip)
			)
			kd = float(getattr(self.cfg, "hybrid_floor_kd", 0.0015))
			z_cmd = kp * error + ki * self._hybrid_floor_integral + kd * force_rate
			# Descent rate limit with continuous taper: teacher-scale descent
			# (±0.018, over-force 0 in all demo quality gates) far below the
			# band, smoothly reduced to the conservative validated cap at the
			# band lower edge. The taper avoids the bang-bang overshoot/lift
			# limit cycle of a hard two-stage cap.
			# Descent authority tapers with headroom to the safety cap, not
			# with distance to the band: full teacher-scale authority anywhere
			# the cap is far (band tracking is the servo's job), conservative
			# descent only where a transient could reach the cap.
			acq_down = float(getattr(self.cfg, "hybrid_floor_acq_max_down", 0.018))
			taper_n = float(getattr(self.cfg, "hybrid_floor_acq_taper_n", 3.0))
			cap_ref = float(getattr(raw_env, "max_safe_force_n", 15.0)) - float(getattr(self.cfg, "hybrid_floor_cap_guard_n", 1.5))
			headroom = cap_ref - safety_force
			frac = float(np.clip(headroom / max(taper_n, 1e-6), 0.0, 1.0))
			if preload_on:
				# V438 continuous-preload authority state machine. Plant fact
				# (diag_v438): steady contact force is roughly a function of
				# the commanded z-delta magnitude (−0.010 ⇔ ~2.2N,
				# −0.016 ⇔ ~4.4N under weak per-step delta tracking), so
				# holding an 8N band requires sustained commands BEYOND the
				# historical caps — but only ever in verified steady contact,
				# where the press is quasi-static and carries no bounce
				# momentum. Authority ramps up slowly while contact is steady
				# (|rate| small), holds through legitimate pressure build
				# (moderate rates), and is zeroed instantly by an impact
				# transient (rate above the brake threshold) or contact loss.
				# The cap-headroom taper and the cap-zone guards below apply
				# to the extended authority unchanged — the safety clamp
				# stays decoupled from the tracking mechanism.
				steady_rate = float(getattr(self.cfg, "hybrid_floor_preload_steady_rate_n", 1.0))
				min_steps = int(getattr(self.cfg, "hybrid_floor_preload_min_steps", 3))
				ramp = float(getattr(self.cfg, "hybrid_floor_preload_ramp", 0.002))
				press_max = float(getattr(self.cfg, "hybrid_floor_press_max_down", 0.030))
				brake_rate = float(getattr(self.cfg, "hybrid_floor_impact_brake_rate_n", 2.5))
				if force_rate > brake_rate:
					self._hybrid_floor_steady_steps = 0
					self._hybrid_floor_preload_authority = 0.0
				elif abs(force_rate) <= steady_rate:
					self._hybrid_floor_steady_steps = getattr(self, "_hybrid_floor_steady_steps", 0) + 1
					if self._hybrid_floor_steady_steps >= min_steps:
						self._hybrid_floor_preload_authority = min(
							getattr(self, "_hybrid_floor_preload_authority", 0.0) + ramp,
							max(0.0, press_max - acq_down),
						)
				# moderate rates: hold current authority (pressure building)
				acq_down = acq_down + getattr(self, "_hybrid_floor_preload_authority", 0.0)
			max_down = float(self.cfg.force_safe_max_down) + frac * (acq_down - float(self.cfg.force_safe_max_down))
			# Reacquisition impact guard: when contact has (nearly) dropped the
			# force reading is low, so the cap-headroom taper grants full
			# descent authority right before an unpredictable re-impact (V428:
			# every observed cap violation had this bounce signature). Cap the
			# descent at the validated near-gap approach rate instead.
			reacq_force_n = float(getattr(self.cfg, "hybrid_floor_reacq_force_n", 0.0)) or float(raw_env.min_wipe_force_n)
			if normal_force < reacq_force_n:
				reacq_down = float(getattr(self.cfg, "hybrid_floor_reacq_max_down", 0.010))
				# V443-F2 (config-gated): re-impact energy scales with the
				# re-approach rate, and the cap headroom shrinks with the
				# target — scale the re-acq descent by ref/target above the
				# reference tier (12N: x2/3). Smooth in target.
				ref_t = float(getattr(self.cfg, "hybrid_floor_reacq_ref_target", 0.0))
				if ref_t > 0:
					reacq_down = reacq_down * min(1.0, ref_t / max(target_force, 1e-6))
				if preload_on:
					# The re-acq cap alone creates a sub-3N equilibrium trap
					# (steady ~2.2N, never crossing the release threshold):
					# in verified steady light contact the preload authority
					# lifts this cap too, so pressure can build without an
					# impact. Bounce states (zero-force steps / rate spikes)
					# have authority 0 and keep the historical guard.
					reacq_down = reacq_down + getattr(self, "_hybrid_floor_preload_authority", 0.0)
				max_down = min(max_down, reacq_down)
			z_cmd = float(np.clip(z_cmd, -max_down, max_lift))
			# V438 predictive impact brake (config-gated): stop descending
			# only when the NEXT step is projected to overshoot the band top
			# (force already at/above the band's low edge AND
			# force + force_rate beyond the high edge) — the soft catch that
			# prevents the pad-recoil unload without braking legitimate
			# pressure build-up below the band. A raw-rate brake stalls the
			# build (measured: band 0.37→0.07 limit cycle at ~5N); V428's
			# rejection of rate braking is resolved by the preload ramp
			# providing the non-impact re-acquisition path.
			if preload_on:
				brake_rate = float(getattr(self.cfg, "hybrid_floor_impact_brake_rate_n", 2.5))
				band_n = float(getattr(raw_env, "force_success_band_n", 2.0))
				low_edge = max(float(raw_env.min_wipe_force_n), target_force - band_n)
				high_edge = target_force + band_n
				if (brake_rate > 0 and force_rate > brake_rate
						and normal_force >= low_edge
						and normal_force + force_rate > high_edge):
					z_cmd = max(z_cmd, 0.0)
			# V442 recoil catch: inside an alarm window, once the recoil
			# starts (force falling), descend at least at the validated
			# re-acq rate so the finger follows the receding pad instead of
			# losing contact. Interlocked away from the cap zone; the
			# max_down clamp and all cap guards below still apply.
			if alarm_active and force_rate < 0 and safety_force < cap_ref - 1.0:
				catch = min(float(getattr(self.cfg, "hybrid_floor_alarm_catch_down", 0.010)), max_down)
				z_cmd = min(z_cmd, -catch)
			# Reactive over-force lift override, cap-aware: for high targets
			# (12N) target+band touches the 15N cap, so the lift reference is
			# the tighter of band top and cap-guard margin (validated
			# target-adaptive ramp behavior).
			max_safe = float(getattr(raw_env, "max_safe_force_n", 15.0))
			guard_n = float(getattr(self.cfg, "hybrid_floor_cap_guard_n", 1.5))
			# Safety lift guards the 15N cap only. Band-top excursions are the
			# servo's job — an eager lift at target+band fights the servo and
			# causes full unload/reacquire cycles (V426 round-1 failure mode).
			lift_ref = max_safe - guard_n
			over_force = safety_force - lift_ref
			if over_force > 0:
				lift = min(max_lift, max(0.012 * zrs, float(self.cfg.force_safe_lift_gain) * over_force))
				z_cmd = max(z_cmd, lift)
				action[..., 0] = action[..., 0] * float(self.cfg.force_safe_slow_x)
			# Rate brake: an observed fast force rise means an impact transient
			# is in progress (V424: these move ~7N/step); stop descending now.
			rate_brake = float(getattr(self.cfg, "hybrid_floor_rate_brake_n", 1.5))
			if rate_brake > 0 and force_rate > rate_brake and normal_force >= float(raw_env.min_wipe_force_n):
				z_cmd = max(z_cmd, 0.0)
			# Hard cap-zone guard: no descent inside the last 2N below the cap
			# regardless of servo state; small mandatory lift.
			if safety_force >= max_safe - 2.0:
				z_cmd = max(z_cmd, 0.008 * zrs)
		# V438 proportional x governor (config-gated): advance at full speed
		# only with real in-band contact; slow down (never hard-stop — the
		# V427 hard x gate caused stick-slip) while contact is light or
		# lost, so the wipe does not outrun the servo. Thresholds are
		# band-relative (smooth parameterization across targets).
		if preload_on and bool(getattr(self.cfg, "hybrid_floor_x_governor", False)):
			tcp_x_g = self._scalar(raw_env.agent.tcp.pose.p[:, 0])
			start_x_g = float(raw_env.path_start_xy[0])
			end_x_g = float(raw_env.path_end_xy[0])
			tcp_prog_g = (tcp_x_g - start_x_g) / max(end_x_g - start_x_g, 1e-6)
			if tcp_prog_g >= float(getattr(self.cfg, "hybrid_floor_contact_gate_progress", 0.02)):
				# Adaptive wipe-speed limiter: per-episode contact ratio is
				# strongly anti-correlated with feed speed (r=-0.80 on DEV) —
				# fast feeds skip (impact-recoil at the control rate). The
				# limiter decays on every contact-quality event (dip below
				# the contact line or an impact-rate spike) and recovers
				# slowly during steady contact, so each feed converges to
				# the speed the pad can sustain instead of paying a fixed
				# slowdown. Static tier: hard slow while contact is lost.
				min_wipe_g = float(raw_env.min_wipe_force_n)
				# V443-F1 (config-gated): at light targets the absolute 3N
				# governor threshold sits at/above the achievable steady
				# force, so the limiter decays permanently and kills
				# progress. Make the threshold target-relative (smooth;
				# only changes targets whose band straddles the line).
				if bool(getattr(self.cfg, "hybrid_floor_xgov_band_thr", False)):
					band_gv = float(getattr(raw_env, "force_success_band_n", 2.0))
					min_wipe_g = min(min_wipe_g, target_force - 0.5 * band_gv)
				brake_g = float(getattr(self.cfg, "hybrid_floor_impact_brake_rate_n", 2.5))
				decay = float(getattr(self.cfg, "hybrid_floor_x_adapt_decay", 0.85))
				# V443-F1 (config-gated): the speed->skipping mechanism the
				# limiter exists for is friction-load driven and vanishes at
				# light targets (corr(contact, x_speed): 8N -0.70, 5N -0.08)
				# — there the decay is pure progress cost. Fade the decay
				# with the band's distance to the contact line (smooth in
				# target: 3N no decay, 5N ~none, 8/12N unchanged).
				if bool(getattr(self.cfg, "hybrid_floor_xgov_light_fade", False)):
					band_lf = float(getattr(raw_env, "force_success_band_n", 2.0))
					mw_lf = float(raw_env.min_wipe_force_n)
					# band-straddle indicator: engage ONLY where the success
					# band straddles the contact line (graded 5N fade
					# regressed CI-significantly on held-out; removed).
					w_lf = 0.0 if (target_force - band_lf) < mw_lf else 1.0
					decay = 1.0 - (1.0 - decay) * w_lf
				recover = float(getattr(self.cfg, "hybrid_floor_x_adapt_recover", 0.02))
				floor_lim = float(getattr(self.cfg, "hybrid_floor_x_adapt_floor", 0.35))
				# Schedule-aware floor: slowing for continuity must not cost
				# completion — when behind the pace that finishes the path by
				# the budget fraction of the episode, the limiter floor rises
				# toward 1 (completion first). Target-independent.
				budget = float(getattr(self.cfg, "hybrid_floor_x_pace_budget", 0.90))
				ep_len = float(getattr(self.cfg, "episode_length", 160) or 160)
				step_i = getattr(self, "_hybrid_floor_step_count", 0)
				# Online command->speed gain estimate (the plant tracks x
				# deltas only partially); the limiter floor is then exactly
				# the multiplier that still finishes the path in budget —
				# throttled fast feeds keep completing, slow feeds are
				# never throttled below their completion speed.
				x_prop = float(action.flatten()[0])
				prev_x = getattr(self, "_hybrid_floor_prev_tcp_x", None)
				last_cmd = getattr(self, "_hybrid_floor_last_x_cmd", 0.0)
				if prev_x is not None and last_cmd > 1e-4:
					g_inst = max(tcp_x_g - prev_x, 0.0) / last_cmd
					g_ema = getattr(self, "_hybrid_floor_x_gain_ema", None)
					self._hybrid_floor_x_gain_ema = g_inst if g_ema is None else 0.9 * g_ema + 0.1 * g_inst
				self._hybrid_floor_prev_tcp_x = tcp_x_g
				g_ema = getattr(self, "_hybrid_floor_x_gain_ema", None)
				remaining_steps = max(budget * ep_len - step_i, 1.0)
				required_speed = max(end_x_g - tcp_x_g, 0.0) / remaining_steps
				min_cmd = float(getattr(self.cfg, "hybrid_floor_x_min_cmd", 0.0))
				if budget > 0 and g_ema is not None and g_ema > 1e-4 and x_prop > 1e-4:
					needed_cmd = required_speed / g_ema
					floor_dyn = float(np.clip(needed_cmd / x_prop, floor_lim, 1.0))
				elif min_cmd > 0 and x_prop > 1e-4:
					# Command-aware completion floor: never throttle the
					# executed command below the (path-property, target-
					# independent) completion command; slow feeds are never
					# throttled at all, fast feeds keep finishing on time.
					floor_dyn = float(np.clip(min_cmd / x_prop, floor_lim, 1.0))
				else:
					floor_dyn = floor_lim
				lim = float(getattr(self, "_hybrid_floor_x_limit", 1.0))
				if normal_force < min_wipe_g or force_rate > brake_g:
					lim = lim * decay
				else:
					lim = lim + recover
				lim = float(np.clip(lim, floor_dyn, 1.0))
				self._hybrid_floor_x_limit = lim
				scale = lim
				if filtered < min_wipe_g:
					scale = min(scale, float(getattr(self.cfg, "hybrid_floor_x_slow_no_contact", 0.4)))
				if alarm_active:
					# V442: don't feed more drag energy into the incoming
					# recoil — brief proportional slowdown, never a stop.
					scale = min(scale, float(getattr(self.cfg, "hybrid_floor_alarm_slow_x", 0.3)))
				if bool(getattr(self.cfg, "hybrid_floor_xgov_light_fade", False)):
					# V443-F1: PARTIAL light-target fade for the static
					# no-contact slowdown. Full removal was tried and is
					# UNSAFE (5N 17.4N re-impact + returning hover episodes)
					# — the slowdown is load-bearing for light-contact
					# safety; fade its strength only up to a floor.
					band_lf2 = float(getattr(raw_env, "force_success_band_n", 2.0))
					mw_lf2 = float(raw_env.min_wipe_force_n)
					w_lf2 = 0.0 if (target_force - band_lf2) < mw_lf2 else 1.0
					slow_hi = float(getattr(self.cfg, "hybrid_floor_xgov_light_slow", 0.65))
					scale_min = scale + (slow_hi - scale) * (1.0 - w_lf2) if scale < slow_hi else scale
					scale = min(max(scale, scale_min), 1.0)
				if scale < 1.0:
					pos = torch.clamp(action[..., 0], min=0.0) * scale
					neg = torch.clamp(action[..., 0], max=0.0)
					action[..., 0] = pos + neg
				self._hybrid_floor_last_x_cmd = max(float(action.flatten()[0]), 0.0)
		action[..., 2] = torch.tensor(z_cmd, dtype=action.dtype, device=action.device)
		return action

	def _apply_force_safe_action(self, obs, action):
		"""Project contact actions into a conservative force-safe set."""
		if not self.cfg.force_safe_action or isinstance(obs, dict):
			return action
		if obs.numel() <= self.cfg.force_plan_target_obs_idx:
			return action
		obs_flat = obs.detach().flatten()
		normal_force = float(obs_flat[self.cfg.force_obs_idx].cpu())
		target_force = float(obs_flat[self.cfg.force_plan_target_obs_idx].cpu())
		raw_env = self.env.unwrapped
		action = action.clone()

		if self.cfg.task == "force-press":
			max_lift = float(self.cfg.force_safe_max_lift)
			max_safe = float(getattr(raw_env, "max_safe_force_n", target_force + 2.0))
			if obs_flat.numel() > 29:
				action[..., 0] = torch.tensor(
					float(np.clip(4.0 * float(obs_flat[28].cpu()), -0.05, 0.05)),
					dtype=action.dtype,
					device=action.device,
				)
				action[..., 1] = torch.tensor(
					float(np.clip(4.0 * float(obs_flat[29].cpu()), -0.05, 0.05)),
					dtype=action.dtype,
					device=action.device,
				)
			max_down = (
				float(self.cfg.force_safe_approach_max_down)
				if normal_force < 0.2
				else float(self.cfg.force_safe_max_down)
			)
			action[..., 2] = torch.clamp(action[..., 2], min=-max_down, max=max_lift)
			if bool(getattr(self.cfg, "force_safe_pi_enabled", False)):
				prev_force_raw = getattr(self, "_force_press_filtered_force", None)
				prev_force = normal_force if prev_force_raw is None else float(prev_force_raw)
				alpha = float(getattr(self.cfg, "force_safe_pi_filter_alpha", 0.70))
				filtered_force = alpha * prev_force + (1.0 - alpha) * normal_force
				self._force_press_filtered_force = filtered_force
				if normal_force < 0.2:
					self._force_press_pi_integral = 0.0
				else:
					i_max = float(getattr(self.cfg, "force_safe_pi_i_max", 18.0))
					prev_i = float(getattr(self, "_force_press_pi_integral", 0.0))
					self._force_press_pi_integral = float(np.clip(prev_i + (filtered_force - target_force), -i_max, i_max))
				kp = float(getattr(self.cfg, "force_safe_pi_kp", 0.006))
				ki = float(getattr(self.cfg, "force_safe_pi_ki", 0.0008))
				correction = kp * (filtered_force - target_force) + ki * float(getattr(self, "_force_press_pi_integral", 0.0))
				z_cmd = torch.clamp(
					action[..., 2] + torch.tensor(correction, dtype=action.dtype, device=action.device),
					min=-max_down,
					max=max_lift,
				)
				under_target = (target_force - float(self.cfg.force_safe_target_band)) - normal_force
				if under_target > 0:
					press = min(max_down, max(0.003, 0.004 * under_target))
					z_cmd = torch.minimum(
						z_cmd,
						torch.tensor(-press, dtype=action.dtype, device=action.device),
					)
				over_target = normal_force - (target_force + float(self.cfg.force_safe_target_band))
				if over_target > 0:
					lift = min(max_lift, max(0.006, kp * over_target))
					z_cmd = torch.maximum(
						z_cmd,
						torch.tensor(lift, dtype=action.dtype, device=action.device),
					)
				near_safe = normal_force - (max_safe - float(self.cfg.force_safe_target_band))
				if near_safe > 0:
					lift = min(max_lift, max(0.02, float(self.cfg.force_safe_lift_gain) * near_safe))
					z_cmd = torch.maximum(
						z_cmd,
						torch.tensor(lift, dtype=action.dtype, device=action.device),
					)
				action[..., 2] = z_cmd
				return action
			under_target = (target_force - float(self.cfg.force_safe_target_band)) - normal_force
			if under_target > 0:
				press = min(max_down, max(0.004, 0.006 * under_target))
				action[..., 2] = torch.minimum(
					action[..., 2],
					torch.tensor(-press, dtype=action.dtype, device=action.device),
				)
			over_target = normal_force - (target_force + float(self.cfg.force_safe_target_band))
			near_safe = normal_force - (max_safe - float(self.cfg.force_safe_target_band))
			if near_safe > 0:
				lift = min(max_lift, max(0.02, float(self.cfg.force_safe_lift_gain) * near_safe))
				action[..., 2] = torch.maximum(
					action[..., 2],
					torch.tensor(lift, dtype=action.dtype, device=action.device),
				)
			elif over_target > 0:
				lift = min(max_lift, max(0.008, float(self.cfg.force_safe_lift_gain) * over_target))
				action[..., 2] = torch.maximum(
					action[..., 2],
					torch.tensor(lift, dtype=action.dtype, device=action.device),
				)
			elif normal_force > target_force:
				action[..., 2] = torch.maximum(
					action[..., 2],
					torch.tensor(0.004, dtype=action.dtype, device=action.device),
				)
			return action

		if self.cfg.task != "force-wipe":
			return action
		max_down = (
			float(self.cfg.force_safe_approach_max_down)
			if normal_force < 0.2
			else float(self.cfg.force_safe_max_down)
		)
		max_lift = float(self.cfg.force_safe_max_lift)
		action[..., 2] = torch.clamp(action[..., 2], min=-max_down, max=max_lift)
		over_force = normal_force - (target_force + float(self.cfg.force_safe_target_band))
		if over_force > 0:
			lift = min(max_lift, max(0.012, float(self.cfg.force_safe_lift_gain) * over_force))
			lift_t = torch.tensor(lift, dtype=action.dtype, device=action.device)
			action[..., 2] = torch.maximum(action[..., 2], lift_t)
			action[..., 0] = action[..., 0] * float(self.cfg.force_safe_slow_x)
		elif normal_force > target_force:
			lift_t = torch.tensor(0.004, dtype=action.dtype, device=action.device)
			action[..., 2] = torch.maximum(action[..., 2], lift_t)
		elif normal_force > 0.2 and normal_force < float(raw_env.min_wipe_force_n):
			action[..., 2] = torch.clamp(action[..., 2], min=-0.01)
		return action

	def _reset_force_press_pi(self):
		if self.cfg.task == "force-press":
			self._force_press_pi_integral = 0.0
			self._force_press_filtered_force = None

	def _demo_blend_alpha(self, eval_mode=False):
		if not self.cfg.demo_action_blend:
			return 0.0
		if eval_mode and not self.cfg.demo_action_blend_eval:
			return 0.0
		start = float(self.cfg.demo_action_blend_alpha)
		min_alpha = float(self.cfg.demo_action_blend_min_alpha)
		decay_steps = max(float(self.cfg.demo_action_blend_decay_steps), 1.0)
		frac = min(float(self._step) / decay_steps, 1.0)
		return max(min_alpha, start * (1.0 - frac))

	def _demo_trajectory_alpha(self, eval_mode=False):
		if not self.cfg.demo_trajectory_blend:
			return 0.0
		if eval_mode and not self.cfg.demo_trajectory_blend_eval:
			return 0.0
		start = float(self.cfg.demo_trajectory_blend_alpha)
		min_alpha = float(self.cfg.demo_trajectory_blend_min_alpha)
		decay_steps = max(float(self.cfg.demo_trajectory_blend_decay_steps), 1.0)
		frac = min(float(self._step) / decay_steps, 1.0)
		return max(min_alpha, start * (1.0 - frac))

	def _nearest_demo_action(self, obs):
		if not hasattr(self, "_demo_obs") or not hasattr(self, "_demo_action") or len(self._demo_action) == 0:
			return None
		obs_flat = obs.detach().flatten().to(self.agent.device)
		dist = float(self.cfg.demo_action_blend_full_obs_weight) * (
			self._demo_obs - obs_flat
		).square().mean(dim=-1)
		if obs_flat.numel() > 39:
			idx = torch.tensor([self.cfg.force_obs_idx, 37, 38, 39], device=self.agent.device)
			weights = torch.tensor([
				self.cfg.demo_action_blend_force_weight,
				self.cfg.demo_action_blend_progress_weight,
				self.cfg.demo_action_blend_progress_weight,
				self.cfg.demo_action_blend_progress_weight,
			], device=self.agent.device)
			dist = dist + ((self._demo_obs[:, idx] - obs_flat[idx]) * weights).square().sum(dim=-1)
		nearest = torch.argmin(dist)
		return self._demo_action[nearest].detach().cpu()

	def _apply_demo_action_blend(self, obs, action, eval_mode=False):
		"""Blend policy/MPPI action with the nearest successful demo action."""
		alpha = self._demo_blend_alpha(eval_mode=eval_mode)
		self._last_demo_blend_alpha = alpha
		if alpha <= 0.0 or self.cfg.task != "force-wipe" or isinstance(obs, dict):
			return action
		if not hasattr(self, "_demo_action") and hasattr(self, "_demo_tds"):
			self._prepare_demo_bc_data()
		if not hasattr(self, "_demo_action"):
			return action
		demo_action = self._nearest_demo_action(obs)
		if demo_action is None:
			return action
		demo_action = demo_action.to(dtype=action.dtype, device=action.device)
		return ((1.0 - alpha) * action + alpha * demo_action).clamp(-1, 1)

	def _reset_demo_trajectory(self, obs):
		"""Select one successful demo trajectory for this episode."""
		self._active_demo_trajectory = None
		self._active_demo_cursor = 0
		if (
			not self.cfg.demo_trajectory_blend
			or not hasattr(self, "_demo_trajectories")
			or len(self._demo_trajectories) == 0
			or isinstance(obs, dict)
		):
			return
		obs_flat = obs.detach().flatten().cpu()
		distances = []
		for traj in self._demo_trajectories:
			distances.append(torch.mean((traj["obs"][0] - obs_flat) ** 2))
		idx = int(torch.argmin(torch.stack(distances)))
		self._active_demo_trajectory = self._demo_trajectories[idx]

	def _trajectory_demo_action(self, obs):
		"""Return the next time-aligned action from the active demo trajectory."""
		if self._active_demo_trajectory is None or isinstance(obs, dict):
			return None
		traj = self._active_demo_trajectory
		if len(traj["action"]) == 0:
			return None
		obs_flat = obs.detach().flatten().cpu()
		current_progress = float(obs_flat[38]) if obs_flat.numel() > 38 else 0.0
		normal_force = float(obs_flat[self.cfg.force_obs_idx]) if obs_flat.numel() > self.cfg.force_obs_idx else 0.0
		progress = traj["progress"]
		lag = float(self.cfg.demo_trajectory_progress_lag)
		contact_threshold = float(self.env.unwrapped.min_wipe_force_n)
		if normal_force < contact_threshold:
			pre_wipe_limit = max(int(traj["contact_start"]), int(traj["wipe_start"]) - 1)
			self._active_demo_cursor = min(self._active_demo_cursor, pre_wipe_limit)
		else:
			while (
				self._active_demo_cursor + 1 < len(progress)
				and float(progress[self._active_demo_cursor]) < current_progress - lag
			):
				self._active_demo_cursor += 1
		cursor = min(self._active_demo_cursor, len(traj["action"]) - 1)
		demo_action = traj["action"][cursor]
		if normal_force < contact_threshold:
			pre_wipe_limit = max(int(traj["contact_start"]), int(traj["wipe_start"]) - 1)
			if self._active_demo_cursor < pre_wipe_limit:
				self._active_demo_cursor += 1
		elif self._active_demo_cursor + 1 < len(traj["action"]):
			self._active_demo_cursor += 1
		return demo_action

	def _apply_demo_trajectory_blend(self, obs, action, eval_mode=False):
		"""Blend with a temporally coherent successful demo trajectory."""
		alpha = self._demo_trajectory_alpha(eval_mode=eval_mode)
		self._last_demo_trajectory_alpha = alpha
		if alpha <= 0.0 or self.cfg.task != "force-wipe" or isinstance(obs, dict):
			return action
		if not hasattr(self, "_demo_trajectories") and hasattr(self, "_demo_tds"):
			self._prepare_demo_trajectory_data()
		if self._active_demo_trajectory is None:
			self._reset_demo_trajectory(obs)
		demo_action = self._trajectory_demo_action(obs)
		if demo_action is None:
			return action
		demo_action = demo_action.to(dtype=action.dtype, device=action.device)
		return ((1.0 - alpha) * action + alpha * demo_action).clamp(-1, 1)

	def _apply_contact_force_control(self, obs, action):
		"""Use measured normal force to make z action a hard contact-stability prior."""
		if not self.cfg.contact_force_control or self.cfg.task != "force-wipe" or isinstance(obs, dict):
			return action
		if obs.numel() <= self.cfg.force_plan_target_obs_idx:
			return action
		obs_flat = obs.detach().flatten()
		normal_force = float(obs_flat[self.cfg.force_obs_idx].cpu())
		target_force = float(obs_flat[self.cfg.force_plan_target_obs_idx].cpu())
		if normal_force < float(self.cfg.contact_force_control_start_n):
			return action
		action = action.clone()
		raw_env = self.env.unwrapped
		force_error = normal_force - target_force
		z_cmd = float(self.cfg.contact_force_control_gain) * force_error
		if normal_force > float(raw_env.max_safe_force_n):
			z_cmd = float(self.cfg.contact_force_control_max_z)
		z_cmd = float(np.clip(
			z_cmd,
			float(self.cfg.contact_force_control_min_z),
			float(self.cfg.contact_force_control_max_z),
		))
		z_cmd_t = torch.tensor(z_cmd, dtype=action.dtype, device=action.device)
		alpha = float(self.cfg.contact_force_control_alpha)
		action[..., 2] = (1.0 - alpha) * action[..., 2] + alpha * z_cmd_t
		if normal_force > target_force + float(self.cfg.contact_force_control_slow_x_band):
			action[..., 0] = action[..., 0] * float(self.cfg.contact_force_control_slow_x)
		return action.clamp(-1, 1)

	def _apply_residual_force_scaffold(self, obs, action):
		"""Target-conditioned residual force scaffold used during train and eval.

		The actor still proposes the action; this layer only enforces the
		contact-regulation interface that the policy is trained under. It is
		deliberately target-normalized so the same rule applies to 5/8/12N.
		"""
		if not bool(getattr(self.cfg, "residual_force_scaffold", False)):
			return action
		if self.cfg.task != "force-wipe" or isinstance(obs, dict):
			return action
		if obs.numel() <= max(int(self.cfg.force_obs_idx), int(self.cfg.force_plan_target_obs_idx)):
			return action
		obs_flat = obs.detach().flatten()
		normal_force = abs(float(obs_flat[int(self.cfg.force_obs_idx)].cpu()))
		target_force = abs(float(obs_flat[int(self.cfg.force_plan_target_obs_idx)].cpu()))
		if target_force <= 1e-6:
			return action
		raw_env = self.env.unwrapped
		min_force = max(
			float(getattr(raw_env, "min_wipe_force_n", 0.0)),
			float(getattr(self.cfg, "residual_force_min_force_n", 3.0)),
		)
		max_safe = min(
			float(getattr(raw_env, "max_safe_force_n", np.inf)),
			float(getattr(self.cfg, "residual_force_max_safe_n", 15.0)),
		)
		band = max(
			float(getattr(self.cfg, "residual_force_min_band_n", 0.5)),
			float(getattr(self.cfg, "residual_force_band_fraction", 0.25)) * target_force,
		)
		low_edge = max(min_force, target_force - band)
		high_edge = min(max_safe, target_force + band)
		tcp_progress = float(obs_flat[38].cpu()) if obs_flat.numel() > 38 else 0.0
		wipe_progress = float(obs_flat[37].cpu()) if obs_flat.numel() > 37 else 0.0
		tcp = raw_env.agent.tcp.pose.p.detach().cpu().numpy()[0]
		start_x = float(raw_env.path_start_xy[0])
		start_y = float(raw_env.path_start_xy[1])
		at_start = (
			abs(float(tcp[0]) - start_x) <= float(getattr(self.cfg, "residual_force_start_tolerance_x", 0.025))
			and abs(float(tcp[1]) - start_y) <= float(getattr(self.cfg, "residual_force_start_tolerance_y", 0.035))
		)
		started_wipe = (
			at_start
			or normal_force > 0.2
			or tcp_progress >= float(getattr(self.cfg, "residual_force_start_progress", 0.02))
			or wipe_progress >= float(getattr(self.cfg, "residual_force_start_progress", 0.02))
		)
		if not started_wipe:
			return action
		action = action.clone()
		alpha = float(getattr(self.cfg, "residual_force_action_alpha", 1.0))

		def blend_axis(axis, value):
			target = torch.tensor(value, dtype=action.dtype, device=action.device)
			action[..., axis] = (1.0 - alpha) * action[..., axis] + alpha * target

		if normal_force < min_force:
			deficit = min_force - normal_force
			down_z = min(
				abs(float(getattr(self.cfg, "residual_force_low_down_z", -0.020))),
				abs(float(getattr(self.cfg, "residual_force_low_gain", 0.004))) * max(deficit, 1.0),
			)
			action[..., 0] = action[..., 0] * float(getattr(self.cfg, "residual_force_low_slow_x", 0.0))
			blend_axis(2, -down_z)
		elif normal_force < low_edge:
			action[..., 0] = action[..., 0] * float(getattr(self.cfg, "residual_force_below_band_slow_x", 0.35))
			blend_axis(2, float(getattr(self.cfg, "residual_force_below_band_down_z", -0.010)))
		elif normal_force > max_safe:
			action[..., 0] = action[..., 0] * float(getattr(self.cfg, "residual_force_near_safe_slow_x", 0.0))
			blend_axis(2, float(getattr(self.cfg, "residual_force_near_safe_lift_z", 0.030)))
		elif normal_force > high_edge:
			over = normal_force - high_edge
			lift_z = min(
				float(getattr(self.cfg, "residual_force_over_lift_z", 0.018)),
				float(getattr(self.cfg, "residual_force_over_gain", 0.008)) * max(over, 1.0),
			)
			action[..., 0] = action[..., 0] * float(getattr(self.cfg, "residual_force_over_slow_x", 0.15))
			blend_axis(2, lift_z)
		return action.clamp(-1, 1)

	def _apply_end_contact_recovery(self, obs, action):
		"""Recover contact near the end of the wipe path before declaring success."""
		if not self.cfg.end_contact_recovery or self.cfg.task != "force-wipe" or isinstance(obs, dict):
			return action
		if obs.numel() <= max(self.cfg.force_obs_idx, 38):
			return action
		obs_flat = obs.detach().flatten()
		normal_force = float(obs_flat[self.cfg.force_obs_idx].cpu())
		tcp_progress = float(obs_flat[38].cpu())
		if tcp_progress < float(self.cfg.end_contact_recovery_progress):
			return action
		action = action.clone()
		if normal_force < float(self.cfg.end_contact_recovery_force_n):
			action[..., 0] = action[..., 0] * float(self.cfg.end_contact_recovery_slow_x)
			action[..., 2] = torch.minimum(
				action[..., 2],
				torch.tensor(float(self.cfg.end_contact_recovery_down_z), dtype=action.dtype, device=action.device),
			)
		elif normal_force > float(self.cfg.end_contact_recovery_max_force_n):
			action[..., 0] = action[..., 0] * float(self.cfg.end_contact_recovery_slow_x)
			action[..., 2] = torch.maximum(
				action[..., 2],
				torch.tensor(float(self.cfg.end_contact_recovery_lift_z), dtype=action.dtype, device=action.device),
			)
		return action.clamp(-1, 1)

	@staticmethod
	def _longest_true_run(mask):
		longest = current = 0
		for value in np.asarray(mask, dtype=bool):
			if value:
				current += 1
				longest = max(longest, current)
			else:
				current = 0
		return int(longest)

	@staticmethod
	def _wipe_failure_phase(
		success,
		ever_over_force,
		contact_acquisition_step,
		final_progress,
		progress_threshold,
		final_force_error,
		force_band,
		final_y_error,
	):
		if success:
			return "success"
		if ever_over_force:
			return "over_force"
		if contact_acquisition_step is None:
			return "contact_acquisition"
		if final_progress <= 0.05:
			return "contact_acquisition"
		if final_progress <= progress_threshold:
			return "wipe_progress_or_retention"
		if final_force_error > force_band:
			return "terminal_force_regulation"
		if final_y_error >= 0.045:
			return "path_alignment"
		return "timeout_or_other"

	def eval(self):
		"""Evaluate a TD-MPC2 agent and write ForceWipe diagnostics."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		ep_success_once, ep_safe_success_once, ep_safe_success_at_end = [], [], []
		ep_wipe_progress, ep_tcp_progress, ep_force_errors = [], [], []
		ep_force_band_progress, ep_instant_force_band_progress = [], []
		ep_normal_force, ep_y_error = [], []
		ep_force_rmse, ep_force_nrmse, ep_force_p95_abs_error = [], [], []
		ep_force_band_rate, ep_no_contact_rate, ep_over_force_rate = [], [], []
		ep_peak_wipe_force, ep_wipe_overshoot = [], []
		ep_contact_acquisition_step, ep_force_settling_steps = [], []
		ep_approach_impact_peak = []
		ep_no_contact_steps, ep_longest_no_contact = [], []
		ep_over_force_steps, ep_longest_over_force = [], []
		ep_potential_shaping_return = []
		bc_mse = self._demo_bc_mse()
		bc_mse_low, bc_mse_in_band, bc_mse_high = self._demo_bc_mse_by_phase()
		dagger_bc_mse = self._dagger_bc_mse()
		raw_env = self.env.unwrapped
		eval_seed_start = int(getattr(self.cfg, "eval_seed_start", -1))
		metrics_progress_start = float(getattr(self.cfg, "eval_force_metrics_progress_start", 0.05))
		settle_dwell_steps = max(int(getattr(self.cfg, "eval_force_settle_dwell_steps", 3)), 1)
		for i in range(self.cfg.eval_episodes):
			eval_seed = eval_seed_start + i if eval_seed_start >= 0 else -1
			obs = self.env.reset(seed=eval_seed) if eval_seed >= 0 else self.env.reset()
			done, ep_reward, t = False, 0, 0
			wipe_forces, wipe_force_errors = [], []
			all_forces = []
			approach_forces = []
			success_once = False
			ever_over_force = False
			contact_acquisition_step = None
			force_settling_step = None
			force_band_streak = 0
			potential_shaping_rewards = []
			self._reset_demo_trajectory(obs)
			self._reset_force_press_pi()
			self._reset_hybrid_floor()
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				action = self._apply_demo_trajectory_blend(obs, action, eval_mode=True)
				action = self._apply_demo_action_blend(obs, action, eval_mode=True)
				action = self._apply_contact_force_control(obs, action)
				action = self._apply_residual_force_scaffold(obs, action)
				action = self._apply_end_contact_recovery(obs, action)
				action = self._apply_hybrid_force_floor(obs, action)
				action = self._apply_force_safe_action(obs, action)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				step_tcp_progress = self._scalar(info.get("tcp_progress"), 0.0)
				step_force = self._scalar(info.get("normal_force"), 0.0)
				step_force_error = self._scalar(info.get("force_error"), np.nan)
				all_forces.append(step_force)
				potential_shaping_rewards.append(
					self._scalar(info.get("potential_shaping_reward"), 0.0)
				)
				success_once = success_once or self._scalar(info.get("success"), 0.0) > 0.5
				ever_over_force = ever_over_force or step_force > float(raw_env.max_safe_force_n)
				if contact_acquisition_step is None and step_force >= float(raw_env.min_wipe_force_n):
					contact_acquisition_step = t - 1
				if step_tcp_progress < metrics_progress_start:
					approach_forces.append(step_force)
				in_force_band = (
					step_tcp_progress >= metrics_progress_start
					and step_force >= float(raw_env.min_wipe_force_n)
					and step_force <= float(raw_env.max_safe_force_n)
					and step_force_error <= float(raw_env.force_success_band_n)
				)
				force_band_streak = force_band_streak + 1 if in_force_band else 0
				if force_settling_step is None and force_band_streak >= settle_dwell_steps:
					force_settling_step = t - settle_dwell_steps
				if step_tcp_progress >= metrics_progress_start:
					wipe_forces.append(step_force)
					wipe_force_errors.append(step_force_error)
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(self._scalar(ep_reward))
			success_at_end = self._scalar(info.get("success"), 0.0)
			safe_success_once = float(success_once and not ever_over_force)
			safe_success_at_end = float(success_at_end > 0.5 and not ever_over_force)
			ep_successes.append(success_at_end)
			ep_success_once.append(float(success_once))
			ep_safe_success_once.append(safe_success_once)
			ep_safe_success_at_end.append(safe_success_at_end)
			ep_lengths.append(t)
			final_wipe_progress = self._scalar(info.get("wipe_progress"))
			final_tcp_progress = self._scalar(info.get("tcp_progress"))
			final_force_band_progress = self._scalar(info.get("force_band_progress"), 0.0)
			final_instant_force_band_progress = self._scalar(info.get("instant_force_band_progress"), 0.0)
			final_force_error = self._scalar(info.get("force_error"))
			final_normal_force = self._scalar(info.get("normal_force"))
			final_y_error = self._scalar(info.get("y_error", info.get("xy_error")))
			ep_wipe_progress.append(final_wipe_progress)
			ep_tcp_progress.append(final_tcp_progress)
			ep_force_band_progress.append(final_force_band_progress)
			ep_instant_force_band_progress.append(final_instant_force_band_progress)
			ep_force_errors.append(final_force_error)
			ep_normal_force.append(final_normal_force)
			ep_y_error.append(final_y_error)
			force_arr = np.asarray(wipe_forces, dtype=np.float64)
			error_arr = np.asarray(wipe_force_errors, dtype=np.float64)
			if force_arr.size:
				force_rmse = float(np.sqrt(np.nanmean(np.square(error_arr))))
				force_nrmse = force_rmse / max(float(raw_env.target_force_n), 1e-6)
				force_p95_abs_error = float(np.nanpercentile(error_arr, 95))
				force_band_rate = float(np.nanmean(error_arr <= float(raw_env.force_success_band_n)))
				no_contact_rate = float(np.nanmean(force_arr < float(raw_env.min_wipe_force_n)))
				over_force_rate = float(np.nanmean(force_arr > float(raw_env.max_safe_force_n)))
				peak_wipe_force = float(np.nanmax(force_arr))
				wipe_overshoot = max(0.0, peak_wipe_force - float(raw_env.target_force_n))
				no_contact_mask = force_arr < float(raw_env.min_wipe_force_n)
				over_force_mask = force_arr > float(raw_env.max_safe_force_n)
				no_contact_steps = int(np.sum(no_contact_mask))
				over_force_steps = int(np.sum(over_force_mask))
				longest_no_contact = self._longest_true_run(no_contact_mask)
				longest_over_force = self._longest_true_run(over_force_mask)
			else:
				force_rmse = force_nrmse = force_p95_abs_error = np.nan
				force_band_rate = no_contact_rate = over_force_rate = np.nan
				peak_wipe_force = np.nan
				wipe_overshoot = np.nan
				no_contact_steps = over_force_steps = 0
				longest_no_contact = longest_over_force = 0
			approach_impact_peak = (
				float(np.max(np.asarray(approach_forces, dtype=np.float64)))
				if approach_forces
				else np.nan
			)
			contact_step_value = (
				float(contact_acquisition_step)
				if contact_acquisition_step is not None
				else np.nan
			)
			settling_steps_value = (
				float(force_settling_step - contact_acquisition_step)
				if force_settling_step is not None and contact_acquisition_step is not None
				else np.nan
			)
			failure_phase = self._wipe_failure_phase(
				success_at_end > 0.5,
				ever_over_force,
				contact_acquisition_step,
				final_wipe_progress,
				float(raw_env.success_progress_threshold),
				final_force_error,
				float(raw_env.force_success_band_n),
				final_y_error,
			)
			ep_force_rmse.append(force_rmse)
			ep_force_nrmse.append(force_nrmse)
			ep_force_p95_abs_error.append(force_p95_abs_error)
			ep_force_band_rate.append(force_band_rate)
			ep_no_contact_rate.append(no_contact_rate)
			ep_over_force_rate.append(over_force_rate)
			ep_peak_wipe_force.append(peak_wipe_force)
			ep_wipe_overshoot.append(wipe_overshoot)
			ep_contact_acquisition_step.append(contact_step_value)
			ep_force_settling_steps.append(settling_steps_value)
			ep_approach_impact_peak.append(approach_impact_peak)
			ep_no_contact_steps.append(no_contact_steps)
			ep_longest_no_contact.append(longest_no_contact)
			ep_over_force_steps.append(over_force_steps)
			ep_longest_over_force.append(longest_over_force)
			potential_shaping_return = float(np.sum(potential_shaping_rewards))
			ep_potential_shaping_return.append(potential_shaping_return)
			self._write_csv_row(self._eval_diag_path, {
				"step": self._step,
				"eval_episode": i,
				"eval_seed": eval_seed,
				"episode_reward": self._scalar(ep_reward),
				"success": success_at_end,
				"success_once": float(success_once),
				"safe_success_once": safe_success_once,
				"safe_success_at_end": safe_success_at_end,
				"ever_over_force": int(ever_over_force),
				"episode_length": t,
				"final_wipe_progress": final_wipe_progress,
				"final_tcp_progress": final_tcp_progress,
				"final_force_band_progress": final_force_band_progress,
				"final_instant_force_band_progress": final_instant_force_band_progress,
				"final_force_error": final_force_error,
				"final_normal_force": final_normal_force,
				"final_y_error": final_y_error,
				"wipe_force_samples": int(force_arr.size),
				"wipe_force_rmse": force_rmse,
				"wipe_force_nrmse": force_nrmse,
				"wipe_force_p95_abs_error": force_p95_abs_error,
				"wipe_force_band_rate": force_band_rate,
				"wipe_no_contact_rate": no_contact_rate,
				"wipe_over_force_rate": over_force_rate,
				"wipe_peak_force": peak_wipe_force,
				"wipe_overshoot_n": wipe_overshoot,
				"contact_acquisition_step": contact_step_value,
				"force_settling_steps": settling_steps_value,
				"settle_dwell_steps": settle_dwell_steps,
				"approach_impact_peak_n": approach_impact_peak,
				"wipe_no_contact_steps": no_contact_steps,
				"wipe_longest_no_contact_steps": longest_no_contact,
				"wipe_over_force_steps": over_force_steps,
				"wipe_longest_over_force_steps": longest_over_force,
				"potential_shaping_return": potential_shaping_return,
				"final_force_phase_potential": self._scalar(
					info.get("force_phase_potential"), 0.0
				),
				"potential_shaping_weight": float(
					getattr(self.cfg, "wipe_potential_shaping_weight", 0.0)
				),
				"failure_phase": failure_phase,
				"failure_reason": self._failure_reason(info),
				"demo_bc_mse": bc_mse,
				"demo_bc_mse_low": bc_mse_low,
				"demo_bc_mse_in_band": bc_mse_in_band,
				"demo_bc_mse_high": bc_mse_high,
				"dagger_samples": self._dagger_size,
				"dagger_states_seen": self._dagger_seen,
				"dagger_bc_mse": dagger_bc_mse,
				"dagger_low_seen": int(self._dagger_phase_seen[0]),
				"dagger_in_band_seen": int(self._dagger_phase_seen[1]),
				"dagger_high_seen": int(self._dagger_phase_seen[2]),
				"curriculum_stage": -1 if self._wipe_curriculum_stage is None else self._wipe_curriculum_stage,
				"path_end_x": float(getattr(raw_env, "path_end_xy", (np.nan, np.nan))[0]),
				"success_progress_threshold": float(getattr(raw_env, "success_progress_threshold", np.nan)),
				"target_force_n": float(getattr(raw_env, "target_force_n", np.nan)),
				"force_success_band_n": float(getattr(raw_env, "force_success_band_n", np.nan)),
				"min_wipe_force_n": float(getattr(raw_env, "min_wipe_force_n", np.nan)),
				"max_safe_force_n": float(getattr(raw_env, "max_safe_force_n", np.nan)),
				"demo_plan_library_size": self._demo_plan_library_size,
				"demo_blend_alpha": self._last_demo_blend_alpha,
				"demo_trajectory_alpha": self._last_demo_trajectory_alpha,
				"demo_trajectory_library_size": self._demo_trajectory_library_size,
				"contact_force_control": int(bool(self.cfg.contact_force_control)),
				"end_contact_recovery": int(bool(self.cfg.end_contact_recovery)),
			})
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		self._write_csv_row(self._bc_retention_path, {
			"step": self._step,
			"demo_bc_mse": bc_mse,
			"demo_bc_mse_low": bc_mse_low,
			"demo_bc_mse_in_band": bc_mse_in_band,
			"demo_bc_mse_high": bc_mse_high,
			"dagger_samples": self._dagger_size,
			"dagger_states_seen": self._dagger_seen,
			"dagger_bc_mse": dagger_bc_mse,
			"dagger_low_seen": int(self._dagger_phase_seen[0]),
			"dagger_in_band_seen": int(self._dagger_phase_seen[1]),
			"dagger_high_seen": int(self._dagger_phase_seen[2]),
			"eval_reward": float(np.nanmean(ep_rewards)),
			"eval_success": float(np.nanmean(ep_successes)),
			"eval_success_once": float(np.nanmean(ep_success_once)),
			"eval_safe_success_once": float(np.nanmean(ep_safe_success_once)),
			"eval_safe_success_at_end": float(np.nanmean(ep_safe_success_at_end)),
			"final_wipe_progress": float(np.nanmean(ep_wipe_progress)),
			"final_force_band_progress": float(np.nanmean(ep_force_band_progress)),
			"final_force_error": float(np.nanmean(ep_force_errors)),
			"wipe_force_rmse": float(np.nanmean(ep_force_rmse)),
			"wipe_force_nrmse": float(np.nanmean(ep_force_nrmse)),
			"wipe_force_p95_abs_error": float(np.nanmean(ep_force_p95_abs_error)),
			"wipe_force_band_rate": float(np.nanmean(ep_force_band_rate)),
			"wipe_no_contact_rate": float(np.nanmean(ep_no_contact_rate)),
			"wipe_over_force_rate": float(np.nanmean(ep_over_force_rate)),
			"wipe_peak_force": float(np.nanmean(ep_peak_wipe_force)),
			"wipe_overshoot_n": float(np.nanmean(ep_wipe_overshoot)),
			"contact_acquisition_step": float(np.nanmean(ep_contact_acquisition_step)),
			"force_settling_steps": float(np.nanmean(ep_force_settling_steps)),
			"approach_impact_peak_n": float(np.nanmean(ep_approach_impact_peak)),
			"wipe_no_contact_steps": float(np.nanmean(ep_no_contact_steps)),
			"wipe_longest_no_contact_steps": float(np.nanmean(ep_longest_no_contact)),
			"wipe_over_force_steps": float(np.nanmean(ep_over_force_steps)),
			"wipe_longest_over_force_steps": float(np.nanmean(ep_longest_over_force)),
			"potential_shaping_return": float(np.nanmean(ep_potential_shaping_return)),
			"curriculum_stage": -1 if self._wipe_curriculum_stage is None else self._wipe_curriculum_stage,
			"path_end_x": float(getattr(raw_env, "path_end_xy", (np.nan, np.nan))[0]),
			"success_progress_threshold": float(getattr(raw_env, "success_progress_threshold", np.nan)),
			"demo_plan_library_size": self._demo_plan_library_size,
			"demo_blend_alpha": self._last_demo_blend_alpha,
			"demo_trajectory_alpha": self._last_demo_trajectory_alpha,
			"demo_trajectory_library_size": self._demo_trajectory_library_size,
			"contact_force_control": int(bool(self.cfg.contact_force_control)),
			"end_contact_recovery": int(bool(self.cfg.end_contact_recovery)),
		})
		print(
			" eval_diag        "
			f"I: {self._step:,} "
			f"progress: {np.nanmean(ep_wipe_progress):.3f} "
			f"safe: {np.nanmean(ep_safe_success_at_end):.3f} "
			f"band_progress: {np.nanmean(ep_force_band_progress):.3f} "
			f"tcp: {np.nanmean(ep_tcp_progress):.3f} "
			f"force_err: {np.nanmean(ep_force_errors):.3f} "
			f"force: {np.nanmean(ep_normal_force):.3f} "
			f"nrmse: {np.nanmean(ep_force_nrmse):.3f} "
			f"band_rate: {np.nanmean(ep_force_band_rate):.3f} "
			f"y_err: {np.nanmean(ep_y_error):.3f} "
			f"bc_mse: {bc_mse:.6f} "
			f"phase_bc: {bc_mse_low:.6f}/{bc_mse_in_band:.6f}/{bc_mse_high:.6f} "
			f"dagger: {self._dagger_size}/{dagger_bc_mse:.6f}"
		)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_success_once=np.nanmean(ep_success_once),
			episode_safe_success_once=np.nanmean(ep_safe_success_once),
			episode_safe_success_at_end=np.nanmean(ep_safe_success_at_end),
			episode_length=np.nanmean(ep_lengths),
			final_wipe_progress=np.nanmean(ep_wipe_progress),
			final_tcp_progress=np.nanmean(ep_tcp_progress),
			final_force_band_progress=np.nanmean(ep_force_band_progress),
			final_force_error=np.nanmean(ep_force_errors),
			final_normal_force=np.nanmean(ep_normal_force),
			final_y_error=np.nanmean(ep_y_error),
			wipe_force_rmse=np.nanmean(ep_force_rmse),
			wipe_force_nrmse=np.nanmean(ep_force_nrmse),
			wipe_force_p95_abs_error=np.nanmean(ep_force_p95_abs_error),
			wipe_force_band_rate=np.nanmean(ep_force_band_rate),
			wipe_no_contact_rate=np.nanmean(ep_no_contact_rate),
			wipe_over_force_rate=np.nanmean(ep_over_force_rate),
			wipe_peak_force=np.nanmean(ep_peak_wipe_force),
			wipe_overshoot_n=np.nanmean(ep_wipe_overshoot),
			contact_acquisition_step=np.nanmean(ep_contact_acquisition_step),
			force_settling_steps=np.nanmean(ep_force_settling_steps),
			approach_impact_peak_n=np.nanmean(ep_approach_impact_peak),
			wipe_no_contact_steps=np.nanmean(ep_no_contact_steps),
			wipe_longest_no_contact_steps=np.nanmean(ep_longest_no_contact),
			wipe_over_force_steps=np.nanmean(ep_over_force_steps),
			wipe_longest_over_force_steps=np.nanmean(ep_longest_over_force),
			potential_shaping_return=np.nanmean(ep_potential_shaping_return),
			demo_bc_mse=bc_mse,
			demo_bc_mse_low=bc_mse_low,
			demo_bc_mse_in_band=bc_mse_in_band,
			demo_bc_mse_high=bc_mse_high,
			dagger_samples=self._dagger_size,
			dagger_states_seen=self._dagger_seen,
			dagger_bc_mse=dagger_bc_mse,
		)
	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	@staticmethod
	def _force_wipe_approach_z_from_gap(z_gap):
		if z_gap > 0.055:
			return -0.065
		if z_gap > 0.025:
			return -0.040
		if z_gap > 0.013:
			return -0.020
		return -0.010

	def _maybe_convert_force_wipe_action_to_basis(self, action):
		"""Convert scripted raw z-delta commands into the configured ForceWipe action basis."""
		if self.cfg.task != "force-wipe":
			return action
		basis = str(getattr(self.cfg, "wipe_action_basis", "delta_pos"))
		if basis not in {"force_admittance_z", "phase_admittance_z", "native_admittance_z"}:
			return action
		raw_env = self.env.unwrapped
		if not hasattr(raw_env, "_normal_force"):
			return action
		if torch.is_tensor(action):
			action_np = action.detach().cpu().numpy().astype(np.float32).copy()
		else:
			action_np = np.asarray(action, dtype=np.float32).copy()
		normal_force = float(raw_env._normal_force().detach().cpu().flatten()[0])
		target_force = float(getattr(raw_env, "target_force_n", 0.0))
		raw_z = float(action_np[..., 2])
		if basis == "force_admittance_z":
			gain = float(getattr(self.cfg, "wipe_force_admittance_z_gain", 0.003))
			residual_scale = max(float(getattr(self.cfg, "wipe_force_admittance_z_residual_scale", 0.020)), 1e-6)
			base_z = gain * (normal_force - target_force)
			action_np[..., 2] = np.clip((raw_z - base_z) / residual_scale, -1.0, 1.0)
			return torch.from_numpy(action_np)

		if basis == "native_admittance_z":
			target_offset_n = max(float(getattr(self.cfg, "wipe_native_admittance_target_offset_n", 3.0)), 1e-6)
			gain_mid = 0.5 * (
				float(getattr(self.cfg, "wipe_native_admittance_kp_min", 0.0015))
				+ float(getattr(self.cfg, "wipe_native_admittance_kp_max", 0.0060))
			)
			gain_mid = max(gain_mid, 1e-6)
			target_setpoint = normal_force - raw_z / gain_mid
			action_np[..., 2] = np.clip((target_setpoint - target_force) / target_offset_n, -1.0, 1.0)
			action_np[..., -1] = 0.0
			return torch.from_numpy(action_np)

		tcp_z = float(raw_env.agent.tcp.pose.p[:, 2].detach().cpu().flatten()[0])
		pad_top_z = float(raw_env.wipe_pad.pose.p[:, 2].detach().cpu().flatten()[0]) + float(raw_env.pad_half_size[2])
		z_gap = tcp_z - pad_top_z
		target_offset_n = max(float(getattr(self.cfg, "wipe_phase_admittance_target_offset_n", 3.0)), 1e-6)
		gain_mid = 0.5 * (
			float(getattr(self.cfg, "wipe_phase_admittance_gain_min", 0.0015))
			+ float(getattr(self.cfg, "wipe_phase_admittance_gain_max", 0.0060))
		)
		gain_mid = max(gain_mid, 1e-6)
		if normal_force < 0.5 or z_gap > 0.013:
			base_z = self._force_wipe_approach_z_from_gap(z_gap)
			residual_scale = max(float(getattr(self.cfg, "wipe_phase_admittance_acq_residual_scale", 0.010)), 1e-6)
			action_np[..., 2] = np.clip((raw_z - base_z) / residual_scale, -1.0, 1.0)
		else:
			band = float(getattr(raw_env, "force_success_band_n", 0.25 * target_force))
			min_force = float(getattr(raw_env, "min_wipe_force_n", 3.0))
			max_force = float(getattr(raw_env, "max_safe_force_n", 15.0))
			low_edge = max(min_force, target_force - band)
			high_edge = min(max_force, target_force + band)
			adjust = -0.004 if normal_force < low_edge else 0.004 if normal_force > high_edge else 0.0
			target_setpoint = normal_force - (raw_z - adjust) / gain_mid
			action_np[..., 2] = np.clip((target_setpoint - target_force) / target_offset_n, -1.0, 1.0)
		action_np[..., -1] = 0.0
		return torch.from_numpy(action_np)

	def _force_wipe_native_admittance_action(self, phase, filtered_force, last_progress):
		"""Native semantic teacher for the ForceWipe admittance action basis."""
		raw_env = self.env.unwrapped
		tcp = raw_env.agent.tcp.pose.p.detach().cpu().numpy()[0]
		target_force = float(raw_env.target_force_n)
		start_x = float(raw_env.path_start_xy[0])
		end_x = float(raw_env.path_end_xy[0])
		y_target = float(raw_env.path_start_xy[1])
		band = float(raw_env.force_success_band_n)
		low_edge = max(float(raw_env.min_wipe_force_n), target_force - band)
		high_edge = min(float(raw_env.max_safe_force_n), target_force + band)
		action = np.zeros(self.env.action_space.shape, dtype=np.float32)
		# Semantic z = target force offset; semantic last dim = admittance gain.
		action[..., 2] = 0.0
		action[..., -1] = 0.0
		def clipped(value, limit):
			return float(np.clip(value, -limit, limit))
		low_force_boundary = target_force <= float(getattr(self.cfg, "wipe_scripted_low_force_boundary_n", 4.0))
		high_force_boundary = target_force >= float(getattr(self.cfg, "wipe_scripted_high_force_boundary_n", 14.0))
		if low_force_boundary:
			base_speed = 0.080
		elif high_force_boundary:
			base_speed = 0.060
		else:
			base_speed = 0.052
		if phase == "move_to_start":
			action[0] = clipped(4.0 * (start_x - tcp[0]), 0.08)
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.08)
			if abs(tcp[0] - start_x) < 0.018 and abs(tcp[1] - y_target) < 0.025:
				phase = "approach"
		elif phase == "approach":
			action[0] = clipped(3.0 * (start_x - tcp[0]), 0.06)
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.06)
			action[..., -1] = 0.25
			if filtered_force >= raw_env.min_wipe_force_n:
				phase = "wipe"
		else:
			in_band = low_edge <= filtered_force <= high_edge
			end_phase = last_progress > raw_env.success_progress_threshold
			if filtered_force < float(raw_env.min_wipe_force_n):
				action[0] = 0.0
				action[2] = 0.75
				action[..., -1] = 0.5
			elif filtered_force < low_edge:
				action[0] = (0.018 if end_phase else 0.5 * base_speed) if tcp[0] < end_x else 0.0
				action[2] = 0.5
				action[..., -1] = 0.25
			elif filtered_force > high_edge:
				action[0] = (0.010 if not end_phase else 0.0) if tcp[0] < end_x else 0.0
				action[2] = -0.5
				action[..., -1] = 0.5
			else:
				action[0] = (0.020 if end_phase else base_speed) if tcp[0] < end_x else 0.0
				action[2] = 0.0
				action[..., -1] = 0.0
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.05)
			if end_phase and not in_band:
				action[0] = 0.0
		return torch.from_numpy(action), phase

	def _force_wipe_scripted_action(self, phase, filtered_force, last_progress):
		"""Scripted ForceWipe controller used only for optional replay warm start."""
		basis = str(getattr(self.cfg, "wipe_action_basis", "delta_pos"))
		if basis == "native_admittance_z" and str(getattr(self.cfg, "wipe_native_teacher_mode", "semantic")) != "inverse_delta":
			return self._force_wipe_native_admittance_action(phase, filtered_force, last_progress)
		raw_env = self.env.unwrapped
		tcp = raw_env.agent.tcp.pose.p.detach().cpu().numpy()[0]
		pad_top_z = float(raw_env.wipe_pad.pose.p[:, 2].detach().cpu().flatten()[0]) + float(raw_env.pad_half_size[2])
		z_gap = float(tcp[2] - pad_top_z)
		target_force = float(raw_env.target_force_n)
		start_x = float(raw_env.path_start_xy[0])
		end_x = float(raw_env.path_end_xy[0])
		y_target = float(raw_env.path_start_xy[1])
		action = np.zeros(self.env.action_space.shape, dtype=np.float32)
		action[..., -1] = -1.0
		def clipped(value, limit):
			return float(np.clip(value, -limit, limit))
		low_force_boundary = target_force <= float(getattr(self.cfg, "wipe_scripted_low_force_boundary_n", 4.0))
		high_force_boundary = target_force >= float(getattr(self.cfg, "wipe_scripted_high_force_boundary_n", 14.0))
		light5_teacher = bool(getattr(self.cfg, "wipe_scripted_light5_teacher", False)) and target_force <= float(getattr(self.cfg, "wipe_scripted_light5_threshold_n", 5.5))
		if light5_teacher:
			low_edge = max(float(raw_env.min_wipe_force_n), target_force - float(raw_env.force_success_band_n))
			high_edge = min(float(raw_env.max_safe_force_n), target_force + float(raw_env.force_success_band_n))
			if phase == "move_to_start":
				action[0] = clipped(4.0 * (start_x - tcp[0]), 0.08)
				action[1] = clipped(4.0 * (y_target - tcp[1]), 0.08)
				action[2] = 0.0
				if abs(tcp[0] - start_x) < 0.018 and abs(tcp[1] - y_target) < 0.025:
					phase = "approach"
			elif phase == "approach":
				action[0] = clipped(3.0 * (start_x - tcp[0]), 0.05)
				action[1] = clipped(4.0 * (y_target - tcp[1]), 0.05)
				if filtered_force < 0.5:
					if z_gap > 0.055:
						action[2] = -0.060
					elif z_gap > 0.025:
						action[2] = -0.035
					elif z_gap > 0.013:
						action[2] = -0.018
					else:
						action[2] = -0.008
				else:
					action[2] = clipped(0.0045 * (filtered_force - target_force), 0.018)
				if filtered_force >= low_edge:
					phase = "wipe"
			else:
				end_phase = last_progress > raw_env.success_progress_threshold
				in_band = low_edge <= filtered_force <= high_edge
				if filtered_force < low_edge:
					wipe_speed = 0.0
				elif filtered_force > high_edge:
					wipe_speed = 0.018
				else:
					wipe_speed = 0.026 if end_phase else 0.092
				action[0] = wipe_speed if tcp[0] < end_x else 0.0
				action[1] = clipped(4.0 * (y_target - tcp[1]), 0.045)
				z_cmd = 0.0040 * (filtered_force - target_force)
				if filtered_force < low_edge:
					z_cmd -= 0.010
				if filtered_force > high_edge:
					z_cmd += 0.004
				if end_phase and not in_band:
					action[0] = 0.0
				action[2] = clipped(z_cmd, 0.018)
			return self._maybe_convert_force_wipe_action_to_basis(torch.from_numpy(action)), phase
		if low_force_boundary:
			approach_switch_force = raw_env.min_wipe_force_n
			force_gate = raw_env.min_wipe_force_n
			approach_gain, approach_limit = 0.006, 0.020
			wipe_gain, wipe_limit, base_speed = 0.0026, 0.012, 0.140
		elif high_force_boundary:
			approach_switch_force = 2.5
			force_gate = max(raw_env.min_wipe_force_n, target_force - 3.0 * raw_env.force_success_band_n)
			approach_gain, approach_limit = 0.005, 0.030
			wipe_gain, wipe_limit, base_speed = 0.0040, 0.024, 0.080
		else:
			approach_switch_force, force_gate = 2.5, 2.0
			# The 5N/8N/12N benchmark uses one controller structure and one
			# gain set; only the target-force setpoint is allowed to change.
			approach_gain, approach_limit = 0.005, 0.030
			wipe_gain, wipe_limit, base_speed = 0.0035, 0.018, 0.052
		if phase == "move_to_start":
			action[0] = clipped(4.0 * (start_x - tcp[0]), 0.08)
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.08)
			action[2] = 0.0
			if abs(tcp[0] - start_x) < 0.018 and abs(tcp[1] - y_target) < 0.025:
				phase = "approach"
		elif phase == "approach":
			action[0] = clipped(3.0 * (start_x - tcp[0]), 0.06)
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.06)
			if filtered_force < 0.5:
				if z_gap > 0.055:
					action[2] = -0.065
				elif z_gap > 0.025:
					action[2] = -0.040
				elif z_gap > 0.013:
					action[2] = -0.020
				else:
					action[2] = -0.010
			else:
				action[2] = clipped(approach_gain * (filtered_force - target_force), approach_limit)
			if filtered_force >= approach_switch_force:
				phase = "wipe"
		else:
			force_tune_phase = last_progress > 0.88
			end_phase = last_progress > raw_env.success_progress_threshold
			wipe_speed = 0.022 if end_phase else base_speed
			if end_phase and abs(filtered_force - target_force) > raw_env.force_success_band_n:
				wipe_speed = 0.0
			action[0] = wipe_speed if (filtered_force >= force_gate and tcp[0] < end_x) else 0.0
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.05)
			z_cmd = wipe_gain * (filtered_force - target_force)
			if filtered_force < raw_env.min_wipe_force_n:
				z_cmd -= 0.004
			end_force_floor = max(raw_env.min_wipe_force_n, target_force - 0.5 * raw_env.force_success_band_n)
			if low_force_boundary:
				end_force_floor = max(raw_env.min_wipe_force_n, target_force - raw_env.force_success_band_n)
			if force_tune_phase and filtered_force < end_force_floor:
				z_cmd -= 0.014 if high_force_boundary else 0.004 if low_force_boundary else 0.006
			if high_force_boundary and filtered_force < target_force - raw_env.force_success_band_n:
				z_cmd -= 0.003
			action[2] = clipped(z_cmd, wipe_limit)
		return self._maybe_convert_force_wipe_action_to_basis(torch.from_numpy(action)), phase

	def _force_wipe_aux_light5_contact_action(self, phase, filtered_force, last_progress, integral, dwell):
		"""Strict-band 5N contact-quality teacher for auxiliary BC demos only."""
		raw_env = self.env.unwrapped
		tcp = raw_env.agent.tcp.pose.p.detach().cpu().numpy()[0]
		pad_top_z = float(raw_env.wipe_pad.pose.p[:, 2].detach().cpu().flatten()[0]) + float(raw_env.pad_half_size[2])
		z_gap = float(tcp[2] - pad_top_z)
		target_force = float(raw_env.target_force_n)
		start_x = float(raw_env.path_start_xy[0])
		end_x = float(raw_env.path_end_xy[0])
		y_target = float(raw_env.path_start_xy[1])
		low_edge = max(float(raw_env.min_wipe_force_n), target_force - float(raw_env.force_success_band_n))
		high_edge = min(float(raw_env.max_safe_force_n), target_force + float(raw_env.force_success_band_n))
		action = np.zeros(self.env.action_space.shape, dtype=np.float32)
		action[..., -1] = -1.0
		def clipped(value, limit):
			return float(np.clip(value, -limit, limit))
		def force_pi(base_down=0.0, max_abs=0.014):
			err = filtered_force - target_force
			integral_next = float(np.clip(integral + err, -80.0, 80.0))
			cmd = 0.0032 * err + 0.00008 * integral_next - base_down
			if filtered_force < low_edge:
				cmd -= 0.0045
			if filtered_force > high_edge:
				cmd += 0.006
			return clipped(cmd, max_abs), integral_next
		if phase == "move_to_start":
			action[0] = clipped(4.0 * (start_x - tcp[0]), 0.08)
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.08)
			action[2] = 0.0
			if abs(tcp[0] - start_x) < 0.018 and abs(tcp[1] - y_target) < 0.025:
				phase = "approach"
		elif phase == "approach":
			action[0] = clipped(3.0 * (start_x - tcp[0]), 0.05)
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.05)
			if filtered_force < 0.4:
				if z_gap > 0.055:
					action[2] = -0.050
				elif z_gap > 0.025:
					action[2] = -0.030
				elif z_gap > 0.012:
					action[2] = -0.014
				else:
					action[2] = -0.006
			else:
				action[2], integral = force_pi(base_down=0.001, max_abs=0.014)
			if filtered_force >= low_edge:
				phase = "hold"
				integral = 0.0
				dwell = 0
		elif phase == "hold":
			action[0] = 0.0
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.04)
			action[2], integral = force_pi(base_down=0.0, max_abs=0.012)
			if low_edge <= filtered_force <= high_edge:
				dwell += 1
			else:
				dwell = 0
			if dwell >= 3:
				phase = "wipe"
				dwell = 0
		else:
			in_band = low_edge <= filtered_force <= high_edge
			end_phase = last_progress > raw_env.success_progress_threshold
			if in_band:
				x_speed = 0.064 if not end_phase else 0.018
			elif filtered_force < low_edge:
				x_speed = 0.0
			else:
				x_speed = 0.014
			action[0] = x_speed if tcp[0] < end_x else 0.0
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.035)
			action[2], integral = force_pi(base_down=0.0, max_abs=0.012)
			if end_phase and not in_band:
				action[0] = 0.0
		return self._maybe_convert_force_wipe_action_to_basis(torch.from_numpy(action)), phase, integral, dwell

	def _force_wipe_teacher_v2_action(self, phase, filtered_force, last_progress, integral, dwell):
		"""Target-conditioned PI force-holding scripted teacher (wipe_teacher_v2=true).

		Unlike _force_wipe_scripted_action's default branch, which shares one gain set
		across all targets (see its comment), every gain here scales with the per-target
		band (force_success_band_n = 0.25 * target), so a tight 5N window (+/-1.25N) gets
		a gentler/slower loop than a loose 12N window (+/-3N). Adds: anti-windup (integral
		resets whenever contact is lost, i.e. filtered_force < min_wipe_force_n), a
		preemptive anti-overshoot term that eases off before force reaches target+band
		(not just reactively once past it), a "hold" phase that dwells in-band for a few
		steps before x-advance starts, and an x-advance gate that only moves at full speed
		while in-band.
		"""
		raw_env = self.env.unwrapped
		tcp = raw_env.agent.tcp.pose.p.detach().cpu().numpy()[0]
		pad_top_z = float(raw_env.wipe_pad.pose.p[:, 2].detach().cpu().flatten()[0]) + float(raw_env.pad_half_size[2])
		z_gap = float(tcp[2] - pad_top_z)
		target_force = float(raw_env.target_force_n)
		band = float(raw_env.force_success_band_n)
		min_force = float(raw_env.min_wipe_force_n)
		max_force = float(raw_env.max_safe_force_n)
		start_x = float(raw_env.path_start_xy[0])
		end_x = float(raw_env.path_end_xy[0])
		y_target = float(raw_env.path_start_xy[1])
		low_edge = max(min_force, target_force - band)
		high_edge = min(max_force, target_force + band)
		action = np.zeros(self.env.action_space.shape, dtype=np.float32)
		action[..., -1] = -1.0
		def clipped(value, limit):
			return float(np.clip(value, -limit, limit))

		# Feedforward + small PI trim, not a from-zero PI. Calibration (static z-command
		# vs. steady-state force, x=0) showed the z-delta-to-force plant is steep and
		# roughly linear (~300N of steady-state force per unit of z-command magnitude),
		# so a from-zero PI has to integrate for many steps before it even reaches a
		# large target like 12N, and once it does the integral term is deeply wound up
		# and overshoots. Feedforward starts the command near the right steady-state
		# depth for target_force directly; the PI below only trims the residual error,
		# which is target-conditioned via `band` (tighter band -> gentler trim).
		plant_gain = float(getattr(self.cfg, "wipe_teacher_v2_plant_gain_n_per_unit_z", 300.0))
		# When target+band is capped by max_safe_force_n (e.g. 12N target, 15N cap ->
		# high_edge=15 with ~0 nominal margin), single-step force noise (observed to be
		# +/-1-2N even under a static hold) can spike past the cap and end the episode
		# in a fail regardless of mean-tracking quality. Bias the regulation setpoint
		# toward the low side of the band in that case to buy margin against spikes.
		ff_target = target_force
		cap_bias_frac = float(getattr(self.cfg, "wipe_teacher_v2_cap_bias_fraction", 0.0))
		if cap_bias_frac > 0.0 and target_force + band >= max_force - 1e-6:
			ff_target = target_force - cap_bias_frac * band
		z_ff = -ff_target / max(plant_gain, 1e-6)
		kp = float(getattr(self.cfg, "wipe_teacher_v2_kp_per_band", 0.00035)) * band
		ki = float(getattr(self.cfg, "wipe_teacher_v2_ki_per_band", 0.000012)) * band
		trim_limit = float(np.clip(
			float(getattr(self.cfg, "wipe_teacher_v2_trim_limit_per_band", 0.0060)) * band,
			float(getattr(self.cfg, "wipe_teacher_v2_trim_limit_min", 0.006)),
			float(getattr(self.cfg, "wipe_teacher_v2_trim_limit_max", 0.016)),
		))
		z_cmd_floor = -float(getattr(self.cfg, "wipe_teacher_v2_z_max_down", 0.042))
		z_cmd_ceiling = float(getattr(self.cfg, "wipe_teacher_v2_z_max_up", 0.050))
		integral_clip = float(getattr(self.cfg, "wipe_teacher_v2_integral_clip_per_band", 20.0)) * band
		base_speed = float(np.clip(
			float(getattr(self.cfg, "wipe_teacher_v2_speed_per_band", 0.012)) * band,
			float(getattr(self.cfg, "wipe_teacher_v2_speed_min", 0.018)),
			float(getattr(self.cfg, "wipe_teacher_v2_speed_max", 0.055)),
		))
		dwell_target = int(getattr(self.cfg, "wipe_teacher_v2_dwell_steps", 4))
		anti_overshoot_gain = float(getattr(self.cfg, "wipe_teacher_v2_anti_overshoot_gain", 0.030))
		caution_gain = float(getattr(self.cfg, "wipe_teacher_v2_caution_gain", 0.012))

		def force_pi(state_integral):
			err = filtered_force - target_force
			if filtered_force < min_force:
				# Anti-windup: no contact -> no integral memory, so we don't slam
				# down on re-contact with a stale accumulated term.
				next_integral = 0.0
			else:
				next_integral = float(np.clip(state_integral + err, -integral_clip, integral_clip))
			trim = kp * err + ki * next_integral
			if filtered_force < low_edge:
				trim -= 0.35 * kp * band
			if filtered_force > high_edge:
				# Reactive: already past the edge, push back hard.
				trim += anti_overshoot_gain * (filtered_force - high_edge)
			elif filtered_force > target_force:
				# Preemptive: ease off before reaching the edge at all.
				proximity = (filtered_force - target_force) / max(band, 1e-6)
				trim += caution_gain * proximity * band
			trim = clipped(trim, trim_limit)
			cmd = z_ff + trim
			# Hard safety backoff, independent of target+band (which caps at
			# max_force for high targets like 12N and so has ~0 margin left by
			# the time high_edge is reached): always start lifting well before
			# the safety cap regardless of how close target+band is to it.
			safety_start = max_force - float(getattr(self.cfg, "wipe_teacher_v2_safety_margin_n", 2.5))
			if filtered_force > safety_start:
				safety_gain = float(getattr(self.cfg, "wipe_teacher_v2_safety_gain", 0.025))
				cmd += safety_gain * (filtered_force - safety_start)
			cmd = float(np.clip(cmd, z_cmd_floor, z_cmd_ceiling))
			return cmd, next_integral

		if phase == "move_to_start":
			action[0] = clipped(4.0 * (start_x - tcp[0]), 0.08)
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.08)
			action[2] = 0.0
			integral = 0.0
			if abs(tcp[0] - start_x) < 0.018 and abs(tcp[1] - y_target) < 0.025:
				phase = "approach"
		elif phase == "approach":
			# Contact acquisition uses the same flat, known-reliable descent as the
			# default teacher (v1): the V290 audit found contact acquisition itself
			# was not the broken part (most demos DO make contact); regulation once
			# in contact was. So approach gains are NOT band-scaled -- only the
			# hold/wipe force-holding PI below is target/band-conditioned.
			action[0] = clipped(3.0 * (start_x - tcp[0]), 0.06)
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.06)
			integral = 0.0  # anti-windup: no accumulation before contact is made
			descent_scale = float(getattr(self.cfg, "wipe_teacher_v2_descent_scale", 1.0))
			if filtered_force < 0.5:
				# Only the outer (still-far-from-contact) tiers are sped up; the
				# final approach tier stays at the gentle v1 rate so the last bit
				# of descent into contact is still a soft touch, not a hard impact.
				if z_gap > 0.055:
					action[2] = -0.065 * descent_scale
				elif z_gap > 0.025:
					action[2] = -0.040 * descent_scale
				elif z_gap > 0.013:
					action[2] = -0.020 * min(descent_scale, 1.5)
				else:
					action[2] = -0.010
			else:
				action[2] = clipped(0.005 * (filtered_force - target_force), 0.030)
			if filtered_force >= 2.5:
				phase = "hold"
				integral = 0.0
				dwell = 0
		elif phase == "hold":
			# DWELL: settle force into band before allowing forward advance.
			action[0] = 0.0
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.04)
			action[2], integral = force_pi(integral)
			if filtered_force < 0.4:
				phase = "approach"  # lost contact entirely -> reacquire
				integral = 0.0
				dwell = 0
			elif low_edge <= filtered_force <= high_edge:
				dwell += 1
			else:
				dwell = 0
			if dwell >= dwell_target:
				phase = "wipe"
				dwell = 0
		else:
			# PROGRESS GATE: advance whenever force is inside or near the band; only
			# slow/stop when force is critically out of range (near no-contact or
			# near the safety cap), since fully stopping on every small excursion
			# both stalls progress and does not itself resettle force any faster
			# than the PI term already does while still moving.
			near_low = max(min_force, low_edge - 0.5 * band)
			near_high = min(max_force, high_edge + 0.5 * band)
			in_band = low_edge <= filtered_force <= high_edge
			near_band = near_low <= filtered_force <= near_high
			end_phase = last_progress > raw_env.success_progress_threshold
			if in_band:
				x_speed = base_speed if not end_phase else 0.35 * base_speed
			elif near_band:
				x_speed = 0.75 * base_speed if not end_phase else 0.20 * base_speed
			elif filtered_force < min_force:
				x_speed = 0.0
			else:
				x_speed = 0.35 * base_speed if not end_phase else 0.0
			action[0] = x_speed if tcp[0] < end_x else 0.0
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.035)
			action[2], integral = force_pi(integral)
			if end_phase and not in_band:
				action[0] = 0.0
			if filtered_force < 0.4:
				phase = "hold"  # lost contact -> settle/dwell again before resuming
				integral = 0.0
				dwell = 0
		return self._maybe_convert_force_wipe_action_to_basis(torch.from_numpy(action)), phase, integral, dwell


	def _force_wipe_oracle_snapshot(self):
		"""Snapshot simulator and wrapper counters for V293 one-step lookahead."""
		raw_state = self.env.unwrapped.get_state().detach().clone()
		attrs = []
		current, seen = self.env, set()
		while current is not None and id(current) not in seen:
			seen.add(id(current))
			values = {}
			for name in FORCE_WIPE_SNAPSHOT_ATTRS:
				if hasattr(current, name):
					value = getattr(current, name)
					if torch.is_tensor(value):
						value = value.detach().clone()
					values[name] = value
			attrs.append((current, values))
			current = getattr(current, "env", None)
		return raw_state, attrs

	def _force_wipe_oracle_restore(self, snapshot):
		"""Restore a V293 oracle snapshot without sharing mutable tensor counters."""
		raw_state, attrs = snapshot
		self.env.unwrapped.set_state(raw_state)
		for wrapper, values in attrs:
			for name, value in values.items():
				if torch.is_tensor(value):
					value = value.detach().clone()
				setattr(wrapper, name, value)

	def _parse_force_wipe_oracle_values(self, name, default):
		value = str(getattr(self.cfg, name, default))
		return [float(v) for v in value.replace(",", "|").split("|") if v.strip()]

	def _force_wipe_oracle_base_action(self):
		action = np.zeros(self.env.action_space.shape, dtype=np.float32)
		action[..., -1] = -1.0
		return action

	def _force_wipe_oracle_precontact_action(self, phase):
		"""Move to the start and acquire first contact before oracle wiping."""
		raw_env = self.env.unwrapped
		tcp = raw_env.agent.tcp.pose.p.detach().cpu().numpy()[0]
		pad_top_z = float(raw_env.wipe_pad.pose.p[:, 2].detach().cpu().flatten()[0]) + float(raw_env.pad_half_size[2])
		z_gap = float(tcp[2] - pad_top_z)
		start_x = float(raw_env.path_start_xy[0])
		y_target = float(raw_env.path_start_xy[1])
		action = self._force_wipe_oracle_base_action()
		def clipped(value, limit):
			return float(np.clip(value, -limit, limit))
		if phase == "move_to_start":
			action[0] = clipped(4.0 * (start_x - tcp[0]), 0.08)
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.08)
			if abs(float(tcp[0] - start_x)) < 0.018 and abs(float(tcp[1] - y_target)) < 0.025:
				phase = "approach"
		else:
			action[0] = clipped(3.0 * (start_x - tcp[0]), 0.05)
			action[1] = clipped(4.0 * (y_target - tcp[1]), 0.05)
			if z_gap > 0.055:
				action[2] = -0.075
			elif z_gap > 0.025:
				action[2] = -0.050
			elif z_gap > 0.013:
				action[2] = -0.023
			else:
				action[2] = -0.008
		return torch.from_numpy(action), phase

	def _force_wipe_oracle_candidate_actions(self, last_progress=0.0):
		"""Candidate action grid for the V293 privileged one-step teacher."""
		raw_env = self.env.unwrapped
		tcp = raw_env.agent.tcp.pose.p.detach().cpu().numpy()[0]
		y_target = float(raw_env.path_start_xy[1])
		end_x = float(raw_env.path_end_xy[0])
		target = float(raw_env.target_force_n)
		def clipped(value, limit):
			return float(np.clip(value, -limit, limit))
		y_err = float(y_target - tcp[1])
		y_limit = 0.055 if abs(y_err) > 0.035 else 0.035
		action_y = clipped(4.0 * y_err, y_limit)
		success_progress = float(getattr(raw_env, "success_progress", 0.92))
		if tcp[0] >= end_x and float(last_progress) >= success_progress:
			xs = [0.0]
		else:
			xs = self._parse_force_wipe_oracle_values(
				"wipe_teacher_v3_candidates_x", "0|0.025|0.05|0.08|0.11"
			)
		zs = self._parse_force_wipe_oracle_values(
			"wipe_teacher_v3_candidates_z", "-0.018|-0.012|-0.006|0|0.006|0.012|0.020"
		)
		if target <= 6.0:
			zs = [-0.018, -0.014, -0.012, -0.011, -0.010, -0.009, -0.008, -0.007, -0.006, 0.0, 0.006, 0.012, 0.020]
		elif target >= 10.0:
			zs = [-0.036, -0.028, -0.022] + zs
		for x in xs:
			for z in zs:
				action = self._force_wipe_oracle_base_action()
				action[0] = float(x)
				action[1] = action_y
				action[2] = float(z)
				yield torch.from_numpy(action)
	def _force_wipe_oracle_score(self, info, prev_progress, action):
		"""Score a one-step candidate by force quality, progress, and safety."""
		raw_env = self.env.unwrapped
		target = float(raw_env.target_force_n)
		band = float(raw_env.force_success_band_n)
		min_force = float(raw_env.min_wipe_force_n)
		max_force = float(raw_env.max_safe_force_n)
		low_edge = max(min_force, target - band)
		high_edge = min(max_force, target + band)
		inner_margin = max(
			0.0,
			float(getattr(self.cfg, "wipe_teacher_v3_inner_band_margin_n", 0.0)),
			float(getattr(self.cfg, "wipe_teacher_v3_inner_band_margin_fraction", 0.0)) * band,
		)
		inner_low_edge = min(target, low_edge + inner_margin)
		inner_high_edge = max(target, high_edge - inner_margin)
		if inner_low_edge > inner_high_edge:
			inner_low_edge = target
			inner_high_edge = target
		force = float(info.get("normal_force", 0.0))
		progress = float(info.get("wipe_progress", 0.0))
		fail = float(info.get("fail", 0.0)) > 0.5
		success = float(info.get("success", 0.0)) > 0.5
		force_err = 0.0
		if force < inner_low_edge:
			force_err = inner_low_edge - force
		elif force > inner_high_edge:
			force_err = force - inner_high_edge
		delta_progress = max(0.0, progress - prev_progress)
		score = 80.0 * delta_progress + 5.0 * progress - 1.5 * force_err * force_err
		if inner_low_edge <= force <= inner_high_edge:
			score += 3.0
		if force < min_force:
			score -= 5.0
		if force > max_force:
			score -= 100.0
		if fail:
			score -= 200.0
		if success:
			score += 100.0
		action_np = action.detach().cpu().numpy()
		if force < min_force or force > inner_high_edge:
			score -= 2.0 * max(0.0, float(action_np[0]))
		return float(score)

	def _force_wipe_teacher_v3_oracle_action(self, phase, filtered_force, last_progress):
		"""Privileged one-step lookahead teacher for high-quality demo generation only."""
		raw_env = self.env.unwrapped
		min_force = float(raw_env.min_wipe_force_n)
		tcp = raw_env.agent.tcp.pose.p.detach().cpu().numpy()[0]
		start_x = float(raw_env.path_start_xy[0])
		start_y = float(raw_env.path_start_xy[1])
		start_misaligned = abs(float(tcp[0] - start_x)) > 0.030 or abs(float(tcp[1] - start_y)) > 0.035
		if phase in {"move_to_start", "approach"} and (filtered_force < 0.7 * min_force or start_misaligned):
			return self._force_wipe_oracle_precontact_action(phase)
		best_action, best_score = None, -1e9
		snapshot = self._force_wipe_oracle_snapshot()
		for action in self._force_wipe_oracle_candidate_actions(last_progress):
			_, _, _, info = self.env.step(action)
			score = self._force_wipe_oracle_score(info, last_progress, action)
			self._force_wipe_oracle_restore(snapshot)
			if score > best_score:
				best_action, best_score = action, score
		return best_action, "oracle_wipe"
	def _force_wipe_demo_quality_metrics(self, forces, progresses, phases):
		"""Compute per-demo force-quality metrics before keeping a demo."""
		raw_env = self.env.unwrapped
		forces = np.asarray(forces, dtype=np.float32)
		progresses = np.asarray(progresses, dtype=np.float32)
		if forces.size == 0:
			return dict(force_band_rate=0.0, no_contact_rate=1.0, over_force_rate=1.0, final_progress=0.0, peak_force=0.0, p95_force=0.0)
		target = float(raw_env.target_force_n)
		band = float(raw_env.force_success_band_n)
		min_force = float(raw_env.min_wipe_force_n)
		max_force = float(raw_env.max_safe_force_n)
		low_edge = max(min_force, target - band)
		high_edge = min(max_force, target + band)
		active_names = {"hold", "wipe", "force_stabilize", "force_gated_wipe", "oracle_wipe", "recovery_low_force", "recovery_over_force"}
		active_mask = np.asarray([p in active_names for p in phases], dtype=bool)
		active_forces = forces[active_mask] if active_mask.any() else forces
		return dict(
			force_band_rate=float(((active_forces >= low_edge) & (active_forces <= high_edge)).mean()),
			no_contact_rate=float((active_forces < min_force).mean()),
			over_force_rate=float((forces > max_force).mean()),
			final_progress=float(progresses[-1]) if progresses.size else 0.0,
			peak_force=float(forces.max()),
			p95_force=float(np.percentile(forces, 95)),
		)

	def _force_wipe_demo_quality_pass(self, metrics):
		if not bool(getattr(self.cfg, "demo_quality_gate", False)):
			return True
		return (
			metrics["force_band_rate"] >= float(getattr(self.cfg, "demo_min_force_band_rate", 0.80))
			and metrics["no_contact_rate"] <= float(getattr(self.cfg, "demo_max_no_contact_rate", 0.10))
			and metrics["over_force_rate"] <= float(getattr(self.cfg, "demo_max_over_force_rate", 0.0))
		)

	def _force_wipe_corrective_action(self, obs):
		"""Label an actor-visited state without executing the scripted action."""
		if self.cfg.task != "force-wipe" or isinstance(obs, dict):
			return None, None
		obs_flat = obs.detach().flatten().cpu()
		if obs_flat.numel() <= max(int(self.cfg.force_obs_idx), 38):
			return None, None
		raw_env = self.env.unwrapped
		normal_force = abs(float(obs_flat[int(self.cfg.force_obs_idx)]))
		wipe_progress = float(obs_flat[37])
		tcp_progress = float(obs_flat[38])
		tcp = raw_env.agent.tcp.pose.p.detach().cpu().numpy()[0]
		start_x = float(raw_env.path_start_xy[0])
		start_y = float(raw_env.path_start_xy[1])
		at_start = (
			abs(float(tcp[0]) - start_x) < 0.018
			and abs(float(tcp[1]) - start_y) < 0.025
		)
		if bool(getattr(self.cfg, "dagger_teacher_v3_oracle", False)):
			self._dagger_teacher_filtered_force = (
				0.65 * self._dagger_teacher_filtered_force
				+ 0.35 * normal_force
			)
			self._dagger_teacher_last_progress = max(
				self._dagger_teacher_last_progress,
				wipe_progress,
			)
			if not bool(getattr(self.cfg, "dagger_stateful_teacher", False)):
				if wipe_progress > 0.01 or normal_force >= float(raw_env.min_wipe_force_n):
					self._dagger_teacher_phase = "wipe"
				elif not at_start or tcp_progress > 0.10:
					self._dagger_teacher_phase = "move_to_start"
				else:
					self._dagger_teacher_phase = "approach"
			action, self._dagger_teacher_phase = self._force_wipe_teacher_v3_oracle_action(
				self._dagger_teacher_phase,
				self._dagger_teacher_filtered_force,
				self._dagger_teacher_last_progress,
			)
		elif bool(getattr(self.cfg, "dagger_stateful_teacher", False)):
			self._dagger_teacher_filtered_force = (
				0.65 * self._dagger_teacher_filtered_force
				+ 0.35 * normal_force
			)
			self._dagger_teacher_last_progress = max(
				self._dagger_teacher_last_progress,
				wipe_progress,
			)
			action, self._dagger_teacher_phase = self._force_wipe_scripted_action(
				self._dagger_teacher_phase,
				self._dagger_teacher_filtered_force,
				self._dagger_teacher_last_progress,
			)
		else:
			if wipe_progress > 0.01 or normal_force >= float(raw_env.min_wipe_force_n):
				phase = "wipe"
			elif not at_start or tcp_progress > 0.10:
				phase = "move_to_start"
			else:
				phase = "approach"
			action, _ = self._force_wipe_scripted_action(
				phase,
				normal_force,
				wipe_progress,
			)
		target_force = float(raw_env.target_force_n)
		band = float(raw_env.force_success_band_n)
		low_edge = max(float(raw_env.min_wipe_force_n), target_force - band)
		high_edge = min(float(raw_env.max_safe_force_n), target_force + band)
		regime = 0 if normal_force < low_edge else 2 if normal_force > high_edge else 1
		return action, regime

	def _reset_dagger_teacher(self):
		self._dagger_teacher_phase = "move_to_start"
		self._dagger_teacher_filtered_force = 0.0
		self._dagger_teacher_last_progress = 0.0

	def _maybe_collect_dagger_correction(self, obs, policy_action):
		"""Aggregate corrective labels on states induced by the learned policy."""
		if (
			not bool(getattr(self.cfg, "dagger_correction", False))
			or self.cfg.task != "force-wipe"
			or self._step < int(getattr(self.cfg, "dagger_start_step", 0))
		):
			return
		collect_every = max(int(getattr(self.cfg, "dagger_collect_every", 1)), 1)
		if self._step % collect_every != 0:
			return
		corrective_action, regime = self._force_wipe_corrective_action(obs)
		if corrective_action is None:
			return
		obs_cpu = obs.detach().flatten().to(dtype=torch.float32, device="cpu")
		action_cpu = corrective_action.detach().flatten().to(dtype=torch.float32, device="cpu")
		policy_cpu = policy_action.detach().flatten().to(dtype=torch.float32, device="cpu")
		action_mse = float(F.mse_loss(policy_cpu, action_cpu))
		if action_mse < float(getattr(self.cfg, "dagger_min_action_mse", 0.0)):
			return
		max_samples = max(int(getattr(self.cfg, "dagger_max_samples", 10000)), 1)
		if self._dagger_obs is None:
			self._dagger_obs = torch.empty(
				(max_samples, obs_cpu.numel()),
				dtype=torch.float32,
				device="cpu",
			)
			self._dagger_action = torch.empty(
				(max_samples, action_cpu.numel()),
				dtype=torch.float32,
				device="cpu",
			)
			self._dagger_regime = torch.empty(
				(max_samples,),
				dtype=torch.long,
				device="cpu",
			)
		self._dagger_seen += 1
		self._dagger_phase_seen[int(regime)] += 1
		self._dagger_action_mse_sum += action_mse
		if self._dagger_size < max_samples:
			slot = self._dagger_size
			self._dagger_size += 1
		else:
			slot = int(np.random.randint(0, self._dagger_seen))
			if slot >= max_samples:
				return
		self._dagger_obs[slot].copy_(obs_cpu)
		self._dagger_action[slot].copy_(action_cpu)
		self._dagger_regime[slot] = int(regime)

	def _sample_dagger_batch(self):
		"""Sample aggregated actor-visited states with corrective teacher labels."""
		if (
			not bool(getattr(self.cfg, "dagger_correction", False))
			or float(getattr(self.cfg, "dagger_bc_coef", 0.0)) <= 0
			or self._dagger_size == 0
		):
			return None
		batch_size = min(
			int(getattr(self.cfg, "dagger_batch_size", 256)),
			self._dagger_size,
		)
		if bool(getattr(self.cfg, "dagger_phase_balance", False)):
			present = [
				phase
				for phase in range(3)
				if bool((self._dagger_regime[:self._dagger_size] == phase).any())
			]
			per_phase = batch_size // max(len(present), 1)
			remainder = batch_size - per_phase * len(present)
			parts = []
			for order, phase in enumerate(present):
				phase_idx = torch.nonzero(
					self._dagger_regime[:self._dagger_size] == phase,
					as_tuple=False,
				).flatten()
				count = per_phase + (1 if order < remainder else 0)
				draw = torch.randint(len(phase_idx), (count,), device="cpu")
				parts.append(phase_idx[draw])
			idx = torch.cat(parts) if parts else torch.randint(
				self._dagger_size,
				(batch_size,),
				device="cpu",
			)
		else:
			idx = torch.randint(self._dagger_size, (batch_size,), device="cpu")
		return (
			self._dagger_obs[idx].to(self.agent.device),
			self._dagger_action[idx].to(self.agent.device),
		)

	@torch.no_grad()
	def _dagger_bc_mse(self):
		"""Measure the current learned actor on its aggregated correction states."""
		if self._dagger_size == 0:
			return np.nan
		count = min(self._dagger_size, 1024)
		idx = torch.linspace(
			0,
			self._dagger_size - 1,
			steps=count,
			dtype=torch.long,
			device="cpu",
		)
		obs = self._dagger_obs[idx].to(self.agent.device)
		action = self._dagger_action[idx].to(self.agent.device)
		self.agent.model.eval()
		z = self.agent.model.encode(obs, task=None)
		_, info = self.agent.model.pi(z, task=None)
		return float(F.mse_loss(info["mean"], action).detach().cpu())

	def _force_press_scripted_action(self, normal_force, phase="approach"):
		"""Phase-aware scripted ForcePress controller for replay warm start.

		The first ForcePress teacher stopped as soon as the force entered the
		success band. That produced demos that looked like "press until contact"
		rather than "approach, regulate force, and avoid over-pressure." This
		version deliberately spends time below the success threshold before
		entering the target band, so BC sees hold/lift labels near contact.
		"""
		raw_env = self.env.unwrapped
		tcp = raw_env.agent.tcp.pose.p.detach().cpu().numpy()[0]
		pad = raw_env.press_pad.pose.p.detach().cpu().numpy()[0]
		pad_top_z = float(pad[2]) + float(raw_env.pad_half_size[2])
		z_gap = float(tcp[2] - pad_top_z)
		target_force = float(raw_env.target_force_n)
		prehold_force = max(0.5, target_force - float(raw_env.force_success_band_n) - 0.8)
		action = np.zeros(self.env.action_space.shape, dtype=np.float32)
		action[..., -1] = -1.0

		def clipped(value, limit):
			return float(np.clip(value, -limit, limit))

		action[0] = clipped(4.0 * (pad[0] - tcp[0]), 0.05)
		action[1] = clipped(4.0 * (pad[1] - tcp[1]), 0.05)
		if phase == "approach":
			if normal_force < 0.5:
				if z_gap > 0.060:
					action[2] = -0.090
				elif z_gap > 0.030:
					action[2] = -0.060
				elif z_gap > 0.014:
					action[2] = -0.035
				else:
					action[2] = -0.014
			else:
				phase = "prehold"

		if phase == "prehold":
			err = normal_force - prehold_force
			if abs(err) < 0.35:
				action[2] = 0.0
			else:
				action[2] = clipped(0.0045 * err, 0.016)
			if normal_force > target_force + 0.5:
				action[2] = max(action[2], 0.012)
		elif phase == "target":
			err = normal_force - target_force
			if normal_force < target_force - float(raw_env.force_success_band_n):
				action[2] = clipped(0.0080 * err - 0.004, 0.030)
			elif abs(err) < 0.35:
				action[2] = 0.0
			else:
				action[2] = clipped(0.0045 * err, 0.020)
			if normal_force > target_force + float(raw_env.force_success_band_n):
				action[2] = max(action[2], 0.014)
		return torch.from_numpy(action), phase

	def _is_force_wipe_demo_reset_candidate(self, info):
		"""Return whether a scripted ForceWipe state is useful for jump-start resets."""
		if not bool(getattr(self.cfg, "demo_state_reset", False)):
			return False
		progress = self._scalar(info.get("wipe_progress"), 0.0)
		normal_force = self._scalar(info.get("normal_force"), 0.0)
		force_error = self._scalar(info.get("force_error"), np.inf)
		raw_env = self.env.unwrapped
		if progress < float(getattr(self.cfg, "demo_state_reset_min_progress", 0.05)):
			return False
		if progress > float(getattr(self.cfg, "demo_state_reset_max_progress", 0.85)):
			return False
		if normal_force < float(getattr(self.cfg, "demo_state_reset_min_force_n", raw_env.min_wipe_force_n)):
			return False
		if bool(getattr(self.cfg, "demo_state_reset_force_band_only", True)):
			band_fraction = float(getattr(self.cfg, "demo_state_reset_force_band_fraction", 0.0))
			if band_fraction > 0.0:
				# Target-proportional band: consistent with the runtime success condition.
				reset_band = band_fraction * float(raw_env.target_force_n)
			else:
				# Prefer runtime value; cfg override only when fraction is disabled.
				reset_band = float(getattr(self.cfg, "demo_state_reset_force_band_n", raw_env.force_success_band_n))
			if force_error > reset_band:
				return False
		return True

	def _maybe_demo_state_reset(self, obs):
		"""Reset online training episodes from successful demo contact states."""
		if (
			self.cfg.task != "force-wipe"
			or not bool(getattr(self.cfg, "demo_state_reset", False))
			or float(getattr(self.cfg, "demo_state_reset_prob", 0.0)) <= 0
			or not self._demo_reset_states
		):
			return obs
		if self._step < int(getattr(self.cfg, "demo_state_reset_start_step", 0)):
			return obs
		start_step = int(getattr(self.cfg, "demo_state_reset_start_step", 0))
		decay_steps = max(int(getattr(self.cfg, "demo_state_reset_decay_steps", 0)), 0)
		max_prob = float(getattr(self.cfg, "demo_state_reset_prob", 0.0))
		min_prob = float(getattr(self.cfg, "demo_state_reset_min_prob", max_prob))
		if decay_steps > 0:
			alpha = np.clip((self._step - start_step) / max(decay_steps, 1), 0.0, 1.0)
			reset_prob = max_prob + alpha * (min_prob - max_prob)
		else:
			reset_prob = max_prob
		raw_env = self.env.unwrapped
		current_target = float(getattr(raw_env, "target_force_n", np.nan))
		low_threshold = float(getattr(self.cfg, "demo_state_reset_low_target_threshold_n", 0.0))
		low_prob = float(getattr(self.cfg, "demo_state_reset_low_target_prob", 0.0))
		if low_threshold > 0 and np.isfinite(current_target) and current_target <= low_threshold:
			reset_prob = max(reset_prob, low_prob)
		if np.random.rand() >= reset_prob:
			return obs
		candidate_indices = np.arange(len(self._demo_reset_states))
		reset_targets = getattr(self, "_demo_reset_targets", [])
		if reset_targets and len(reset_targets) == len(self._demo_reset_states):
			tolerance = float(getattr(self.cfg, "demo_state_reset_target_tolerance_n", 0.25))
			candidate_indices = np.asarray([
				i for i, target in enumerate(reset_targets)
				if abs(float(target) - current_target) <= tolerance
			], dtype=np.int64)
			if candidate_indices.size == 0:
				if not bool(getattr(self.cfg, "demo_state_reset_fallback_any", False)):
					return obs
				candidate_indices = np.arange(len(self._demo_reset_states))
		idx = int(np.random.choice(candidate_indices))
		state = self._demo_reset_states[idx].clone()
		device = getattr(raw_env, "device", "cpu")
		raw_env.set_state(state.to(device))
		# Refresh task-internal progress trackers after restoring the simulator state.
		raw_env.evaluate()
		self._demo_reset_count += 1
		obs = raw_env.get_obs()
		if torch.is_tensor(obs) and obs.ndim == 2 and obs.shape[0] == 1:
			obs = obs[0]
		return obs

	def _prefill_force_wipe_aux_light5_demos(self):
		"""Append strict-band 5N contact-quality auxiliary demos for BC only."""
		want = int(getattr(self.cfg, "demo_aux_light5_contact_episodes", 0))
		if self.cfg.task != "force-wipe" or want <= 0:
			return
		target_n = float(getattr(self.cfg, "demo_aux_light5_target_n", 5.0))
		tol = float(getattr(self.cfg, "demo_aux_light5_target_tolerance_n", 0.05))
		band_fraction = float(getattr(self.cfg, "demo_aux_light5_band_fraction", 0.25))
		min_band_rate = float(getattr(self.cfg, "demo_aux_light5_min_band_rate", 0.35))
		max_over_rate = float(getattr(self.cfg, "demo_aux_light5_max_over_force_rate", 0.02))
		max_attempts = int(getattr(self.cfg, "demo_aux_light5_max_attempts", 0)) or max(4 * want, want)
		print(
			"Collecting auxiliary strict-band 5N contact demos "
			f"target={target_n:.2f}N band_fraction={band_fraction:.2f} "
			f"want={want} max_attempts={max_attempts}..."
		)
		kept = 0
		attempts = 0
		band_rates, no_contact_rates, over_rates, progresses = [], [], [], []
		if not hasattr(self, "_demo_tds"):
			self._demo_tds = []
		while kept < want and attempts < max_attempts:
			attempts += 1
			obs = self.env.reset()
			raw_env = self.env.unwrapped
			current_target = float(raw_env.target_force_n)
			if abs(current_target - target_n) > tol:
				continue
			expected_band = band_fraction * target_n
			if abs(float(raw_env.force_success_band_n) - expected_band) > 1e-4:
				raise RuntimeError(
					"Aux 5N demo protocol mismatch: "
					f"target={current_target:.3f}, band={float(raw_env.force_success_band_n):.3f}, "
					f"expected_band={expected_band:.3f}."
				)
			low_edge = max(float(raw_env.min_wipe_force_n), current_target - float(raw_env.force_success_band_n))
			high_edge = min(float(raw_env.max_safe_force_n), current_target + float(raw_env.force_success_band_n))
			tds = [self.to_td(obs)]
			phase = "move_to_start"
			filtered_force = 0.0
			last_progress = 0.0
			integral = 0.0
			dwell = 0
			forces, progresses_ep, phases_ep = [], [], []
			info = {"terminated": torch.tensor(0.0), "success": 0.0}
			for _ in range(self.cfg.episode_length):
				action, phase, integral, dwell = self._force_wipe_aux_light5_contact_action(
					phase, filtered_force, last_progress, integral, dwell
				)
				obs, reward, done, info = self.env.step(action)
				normal_force = float(info.get("normal_force", 0.0))
				filtered_force = 0.65 * filtered_force + 0.35 * normal_force
				last_progress = float(info.get("wipe_progress", last_progress))
				forces.append(normal_force)
				progresses_ep.append(last_progress)
				phases_ep.append(phase)
				tds.append(self.to_td(obs, action, reward, info["terminated"]))
				if done:
					break
			forces_arr = np.asarray(forces, dtype=np.float32)
			if forces_arr.size == 0:
				continue
			active_mask = np.asarray([phase in ("hold", "wipe") for phase in phases_ep], dtype=bool)
			active_forces = forces_arr[active_mask] if active_mask.any() else forces_arr
			band_rate = float(((active_forces >= low_edge) & (active_forces <= high_edge)).mean())
			no_contact = float((active_forces < float(raw_env.min_wipe_force_n)).mean())
			over_rate = float((active_forces > float(raw_env.max_safe_force_n)).mean())
			progress = float(progresses_ep[-1]) if progresses_ep else 0.0
			if band_rate < min_band_rate or over_rate > max_over_rate:
				continue
			demo_td = torch.cat(tds)
			if self._demo_tds and "episode" in self._demo_tds[0].keys() and "episode" not in demo_td.keys():
				demo_td["episode"] = torch.zeros(demo_td.batch_size, dtype=torch.int64)
			self._demo_tds.append(demo_td)
			kept += 1
			band_rates.append(band_rate)
			no_contact_rates.append(no_contact)
			over_rates.append(over_rate)
			progresses.append(progress)
		print(
			"Aux 5N contact demos "
			f"kept={kept}/{attempts}, "
			f"band_rate={np.mean(band_rates) if band_rates else float('nan'):.3f}, "
			f"no_contact={np.mean(no_contact_rates) if no_contact_rates else float('nan'):.3f}, "
			f"over_force={np.mean(over_rates) if over_rates else float('nan'):.3f}, "
			f"progress={np.mean(progresses) if progresses else float('nan'):.3f}"
		)

	def _assert_wipe_band_consistency(self):
		"""Assert that config band constants agree with the runtime env band."""
		if self.cfg.task != "force-wipe":
			return
		raw_env = self.env.unwrapped
		band_fraction = float(getattr(self.cfg, "wipe_target_force_band_fraction", 0.0))
		if band_fraction <= 0.0:
			return
		target = float(raw_env.target_force_n)
		expected_band = band_fraction * target
		actual_band = float(raw_env.force_success_band_n)
		if abs(actual_band - expected_band) > 1e-4:
			raise RuntimeError(
				f"Band consistency check failed: "
				f"raw_env.force_success_band_n={actual_band:.4f} "
				f"but wipe_target_force_band_fraction({band_fraction}) * target({target}) = {expected_band:.4f}. "
				f"Check wipe_force_success_band_n vs wipe_target_force_band_fraction."
			)
		# Warn (not error) when phase_policy_band_n doesn't match and fraction not set.
		phase_fraction = float(getattr(self.cfg, "phase_policy_band_fraction", 0.0))
		if bool(getattr(self.cfg, "phase_policy", False)) and phase_fraction <= 0.0:
			phase_band = float(getattr(self.cfg, "phase_policy_band_n", 2.0))
			if abs(phase_band - expected_band) > 1e-4:
				print(
					f"[WARN] band_consistency: phase_policy_band_n={phase_band:.3f} "
					f"!= expected {expected_band:.3f} for target={target:.1f}N. "
					f"Set phase_policy_band_fraction={band_fraction} to align phase labels "
					f"with the eval success band."
				)

	def _prefill_force_wipe_demos(self):
		"""Add scripted successful ForceWipe trajectories to replay before online data."""
		if self.cfg.task != "force-wipe" or self.cfg.demo_prefill_episodes <= 0:
			return
		target_values = [
			float(value)
			for value in str(
				getattr(self.cfg, "wipe_target_force_values", "") or ""
			).replace(",", "|").split("|")
			if value.strip()
		]
		per_force_target = int(
			getattr(self.cfg, "demo_target_successes_per_force", 0)
		)
		target_kept = {
			float(target): 0 for target in target_values
		} if per_force_target > 0 else {}
		target_successes = int(getattr(self.cfg, "demo_target_successes", 0))
		if target_kept:
			target_successes = per_force_target * len(target_kept)
		max_attempts = int(getattr(self.cfg, "demo_max_attempts", 0))
		if target_successes > 0:
			attempt_limit = max_attempts if max_attempts > 0 else max(
				self.cfg.demo_prefill_episodes,
				4 * target_successes,
			)
			print(
				"Prefilling replay until "
				f"{target_successes} successful scripted ForceWipe demos "
				f"(max_attempts={attempt_limit})..."
			)
		else:
			attempt_limit = self.cfg.demo_prefill_episodes
			print(f"Prefilling replay with {attempt_limit} scripted ForceWipe demos...")
		successes = 0
		kept = 0
		attempts = 0
		self._demo_tds = []
		self._demo_reset_states = []
		self._demo_reset_targets = []
		max_reset_states = int(getattr(self.cfg, "demo_state_reset_max_states", 5000))
		quality_path = Path(self.cfg.work_dir) / str(getattr(self.cfg, "demo_prefill_quality_csv", "demo_prefill_quality.csv"))
		for _ in range(attempt_limit):
			if target_successes > 0 and kept >= target_successes:
				break
			attempts += 1
			obs = self.env.reset()
			self._assert_wipe_band_consistency()
			episode_target_force = float(self.env.unwrapped.target_force_n)
			tds = [self.to_td(obs)]
			reset_candidates = []
			done = False
			phase = "move_to_start"
			filtered_force = 0.0
			last_progress = 0.0
			teacher_v3_oracle = bool(getattr(self.cfg, "wipe_teacher_v3_oracle", False))
			teacher_v2 = bool(getattr(self.cfg, "wipe_teacher_v2", False))
			teacher_v2_integral = 0.0
			teacher_v2_dwell = 0
			steps = 0
			forces, progresses, phases = [], [], []
			info = {"terminated": torch.tensor(0.0), "success": 0.0}
			while not done:
				if teacher_v3_oracle:
					action, phase = self._force_wipe_teacher_v3_oracle_action(phase, filtered_force, last_progress)
				elif teacher_v2:
					action, phase, teacher_v2_integral, teacher_v2_dwell = self._force_wipe_teacher_v2_action(
						phase, filtered_force, last_progress, teacher_v2_integral, teacher_v2_dwell
					)
				else:
					action, phase = self._force_wipe_scripted_action(phase, filtered_force, last_progress)
				obs, reward, done, info = self.env.step(action)
				normal_force = float(info.get("normal_force", 0.0))
				filtered_force = 0.65 * filtered_force + 0.35 * normal_force
				last_progress = float(info.get("wipe_progress", last_progress))
				forces.append(normal_force)
				progresses.append(last_progress)
				phases.append(phase)
				tds.append(self.to_td(obs, action, reward, info["terminated"]))
				if self._is_force_wipe_demo_reset_candidate(info):
					reset_candidates.append(self.env.unwrapped.get_state().detach().cpu())
				steps += 1
				if steps >= self.cfg.episode_length:
					break
			is_success = int(float(info.get("success", 0.0)) > 0.5)
			successes += is_success
			quality = self._force_wipe_demo_quality_metrics(forces, progresses, phases)
			quality_pass = self._force_wipe_demo_quality_pass(quality)
			target_key = None
			if target_kept:
				target_key = min(
					target_kept,
					key=lambda value: abs(value - episode_target_force),
				)
			drop_reason = "keep"
			if self.cfg.demo_success_only and not is_success:
				drop_reason = "not_success"
			elif not quality_pass:
				drop_reason = "quality_gate"
			elif target_kept and target_kept[target_key] >= per_force_target:
				drop_reason = "target_full"
			keep_demo = drop_reason == "keep"
			self._write_csv_row(quality_path, dict(
				attempt=attempts,
				target_force=episode_target_force,
				target_key=float(target_key) if target_key is not None else np.nan,
				success=is_success,
				kept=int(keep_demo),
				drop_reason=drop_reason,
				steps=steps,
				final_progress=quality["final_progress"],
				force_band_rate=quality["force_band_rate"],
				no_contact_rate=quality["no_contact_rate"],
				over_force_rate=quality["over_force_rate"],
				peak_force=quality["peak_force"],
				p95_force=quality["p95_force"],
			))
			if not keep_demo:
				continue
			demo_td = torch.cat(tds)
			self._demo_tds.append(demo_td)
			self.buffer.add(demo_td)
			if len(self._demo_reset_states) < max_reset_states:
				remaining = max_reset_states - len(self._demo_reset_states)
				selected_reset_candidates = reset_candidates[:remaining]
				self._demo_reset_states.extend(selected_reset_candidates)
				self._demo_reset_targets.extend(
					[episode_target_force] * len(selected_reset_candidates)
				)
			kept += 1
			if target_kept:
				target_kept[target_key] += 1
		print(f"Scripted demo prefill success={successes}/{attempts}, kept={kept}")
		if target_kept:
			print(
				"Scripted demo targets kept="
				+ ", ".join(
					f"{target:g}N:{count}"
					for target, count in sorted(target_kept.items())
				)
			)
		if target_successes > 0 and kept < target_successes:
			raise RuntimeError(
				"ForceWipe demo collection failed: "
				f"required {target_successes} successful demos, kept {kept} "
				f"after {attempts} attempts."
			)
		if bool(getattr(self.cfg, "demo_state_reset", False)):
			target_counts = {}
			for target in getattr(self, "_demo_reset_targets", []):
				target_key = f"{float(target):g}N"
				target_counts[target_key] = target_counts.get(target_key, 0) + 1
			print(
				"Demo state reset library "
				f"states={len(self._demo_reset_states)}, "
				f"prob={float(getattr(self.cfg, 'demo_state_reset_prob', 0.0)):.3f}"
			)
			if target_counts:
				print(
					"Demo state reset targets="
					+ ", ".join(
						f"{target}:{count}"
						for target, count in sorted(target_counts.items())
					)
				)
		if kept == 0:
			print("No scripted demos kept; skipping demo BC.")
			return
		self._prefill_force_wipe_aux_light5_demos()
		self._prepare_demo_plan_data()
		self._prepare_demo_trajectory_data()
		self._pretrain_policy_on_demos()

	def _prefill_force_press_demos(self):
		"""Add scripted successful ForcePress trajectories to replay before online data."""
		if self.cfg.task != "force-press" or self.cfg.demo_prefill_episodes <= 0:
			return
		print(f"Prefilling replay with {self.cfg.demo_prefill_episodes} scripted ForcePress demos...")
		successes = 0
		kept = 0
		self._demo_tds = []
		for _ in range(self.cfg.demo_prefill_episodes):
			obs = self.env.reset()
			tds = [self.to_td(obs)]
			done = False
			normal_force = 0.0
			filtered_force = 0.0
			phase = "approach"
			prehold_steps = 0
			steps = 0
			info = {"terminated": torch.tensor(0.0), "success": 0.0}
			while not done:
				if phase == "prehold":
					prehold_steps += 1
					if prehold_steps >= 10:
						phase = "target"
				action, phase = self._force_press_scripted_action(filtered_force, phase)
				obs, reward, done, info = self.env.step(action)
				normal_force = float(info.get("normal_force", 0.0))
				filtered_force = 0.65 * filtered_force + 0.35 * normal_force
				tds.append(self.to_td(obs, action, reward, info["terminated"]))
				steps += 1
				if steps >= self.cfg.episode_length:
					break
			is_success = int(float(info.get("success", 0.0)) > 0.5)
			successes += is_success
			if self.cfg.demo_success_only and not is_success:
				continue
			demo_td = torch.cat(tds)
			self._demo_tds.append(demo_td)
			self.buffer.add(demo_td)
			kept += 1
		print(f"Scripted demo prefill success={successes}/{self.cfg.demo_prefill_episodes}, kept={kept}")
		if kept == 0:
			print("No scripted demos kept; skipping demo BC.")
			return
		self._pretrain_policy_on_demos()

	def _pretrain_policy_on_demos(self):
		"""Behavior-clone the policy prior from scripted demo transitions."""
		if self.cfg.demo_bc_steps <= 0 or not hasattr(self, "_demo_tds"):
			return
		self._prepare_demo_bc_data()
		obs = self._demo_obs
		action = self._demo_action
		if len(action) == 0:
			print("Skipping demo BC: no valid scripted actions.")
			return
		batch_size = min(int(self.cfg.demo_bc_batch_size), len(action))
		print(f"Pretraining policy prior on {len(action)} demo transitions for {self.cfg.demo_bc_steps} BC steps...")
		self.agent.model.train()
		if bool(getattr(self.cfg, "phase_policy", False)):
			gate_steps = max(0, int(getattr(self.cfg, "phase_policy_gate_bc_steps", 200)))
			phase_labels = self.agent.phase_labels_from_obs(obs)
			phase_counts = torch.bincount(phase_labels, minlength=3).float()
			phase_present = phase_counts > 0
			phase_weights = torch.zeros_like(phase_counts)
			phase_weights[phase_present] = (
				len(phase_labels)
				/ (phase_present.sum().float() * phase_counts[phase_present])
			)
			phase_weights = phase_weights.clamp(
				max=float(getattr(self.cfg, "phase_policy_weight_clip", 10.0))
			)
			last_gate_loss = None
			for _ in range(gate_steps):
				idx = torch.randint(len(action), (batch_size,), device=self.agent.device)
				with torch.no_grad():
					z = self.agent.model.encode(obs[idx], task=None)
				logits = self.agent.model.phase(z, task=None)
				gate_loss = F.cross_entropy(
					logits,
					phase_labels[idx],
					weight=phase_weights,
				)
				self.agent.optim.zero_grad(set_to_none=True)
				gate_loss.backward()
				torch.nn.utils.clip_grad_norm_(
					self.agent.model._phase.parameters(),
					self.cfg.grad_clip_norm,
				)
				self.agent.optim.step()
				last_gate_loss = gate_loss.detach()
			with torch.no_grad():
				all_z = self.agent.model.encode(obs, task=None)
				gate_accuracy = (
					self.agent.model.phase(all_z, task=None).argmax(dim=-1)
					== phase_labels
				).float().mean()
			print(
				"Phase gate BC "
				f"steps={gate_steps}, "
				f"loss={float(last_gate_loss) if last_gate_loss is not None else float('nan'):.6f}, "
				f"accuracy={float(gate_accuracy):.3f}, "
				f"counts={phase_counts.int().tolist()}"
			)
		if bool(getattr(self.cfg, "target_policy", False)):
			gate_steps = max(0, int(getattr(self.cfg, "target_policy_gate_bc_steps", getattr(self.cfg, "phase_policy_gate_bc_steps", 200))))
			target_labels = self.agent.target_policy_labels_from_obs(obs)
			target_counts = torch.bincount(target_labels, minlength=3).float()
			target_present = target_counts > 0
			target_weights = torch.zeros_like(target_counts)
			target_weights[target_present] = (
				len(target_labels)
				/ (target_present.sum().float() * target_counts[target_present])
			)
			target_weights = target_weights.clamp(
				max=float(getattr(self.cfg, "target_policy_weight_clip", 10.0))
			)
			last_target_gate_loss = None
			for _ in range(gate_steps):
				idx = torch.randint(len(action), (batch_size,), device=self.agent.device)
				with torch.no_grad():
					z = self.agent.model.encode(obs[idx], task=None)
				logits = self.agent.model.target_gate(z, task=None)
				target_gate_loss = F.cross_entropy(
					logits,
					target_labels[idx],
					weight=target_weights,
				)
				self.agent.optim.zero_grad(set_to_none=True)
				target_gate_loss.backward()
				torch.nn.utils.clip_grad_norm_(
					self.agent.model._target_gate.parameters(),
					self.cfg.grad_clip_norm,
				)
				self.agent.optim.step()
				last_target_gate_loss = target_gate_loss.detach()
			with torch.no_grad():
				all_z = self.agent.model.encode(obs, task=None)
				target_gate_accuracy = (
					self.agent.model.target_gate(all_z, task=None).argmax(dim=-1)
					== target_labels
				).float().mean()
			print(
				"Target gate BC "
				f"steps={gate_steps}, "
				f"loss={float(last_target_gate_loss) if last_target_gate_loss is not None else float('nan'):.6f}, "
				f"accuracy={float(target_gate_accuracy):.3f}, "
				f"counts={target_counts.int().tolist()}"
			)
		last_loss = None
		for _ in range(int(self.cfg.demo_bc_steps)):
			if getattr(self, "_demo_bc_weight", None) is not None:
				idx = torch.multinomial(self._demo_bc_weight, batch_size, replacement=True)
			else:
				idx = torch.randint(len(action), (batch_size,), device=self.agent.device)
			with torch.no_grad():
				z = self.agent.model.encode(obs[idx], task=None)
			_, info = self.agent.model.pi(z, task=None)
			bc_loss = F.mse_loss(info["mean"], action[idx])
			self.agent.pi_optim.zero_grad(set_to_none=True)
			bc_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.agent.model._pi.parameters(), self.cfg.grad_clip_norm)
			self.agent.pi_optim.step()
			last_loss = bc_loss.detach()
		self.agent.model.eval()
		print(f"Demo BC final_loss={float(last_loss):.6f}")
		if bool(getattr(self.cfg, "eval_ema", False)):
			self.agent.capture_eval_ema()
			print("Captured evaluation EMA model after demo BC.")
		if (
			float(getattr(self.cfg, "teacher_anchor_coef", 0.0)) > 0
			or float(getattr(self.cfg, "teacher_encoder_anchor_coef", 0.0)) > 0
		):
			teacher_checkpoint = getattr(self.cfg, "teacher_checkpoint", None)
			if teacher_checkpoint is not None and str(teacher_checkpoint).lower() not in ("", "none", "null"):
				self.agent.load_bc_teacher(str(teacher_checkpoint))
				print(f"Loaded frozen external teacher checkpoint: {teacher_checkpoint}")
			else:
				self.agent.capture_bc_teacher()
				print("Captured frozen BC teacher for on-policy anchoring.")
		if bool(getattr(self.cfg, "save_post_bc_checkpoint", False)):
			post_bc_dir = Path(self.cfg.work_dir) / "models"
			post_bc_dir.mkdir(parents=True, exist_ok=True)
			post_bc_path = post_bc_dir / "post_bc.pt"
			self.agent.save(post_bc_path)
			print(f"Saved post-BC checkpoint: {post_bc_path}")

	def _prepare_demo_bc_data(self):
		"""Cache valid scripted demo state-action pairs on the agent device."""
		if hasattr(self, "_demo_obs") and hasattr(self, "_demo_action"):
			return
		demo = torch.cat(self._demo_tds)
		obs = demo["obs"][:-1].to(self.agent.device)
		action = demo["action"][1:].to(self.agent.device)
		valid = ~torch.isnan(action).any(dim=-1)
		self._demo_obs = obs[valid]
		self._demo_action = action[valid]
		self._demo_bc_weight = None
		if (
			bool(getattr(self.cfg, "demo_bc_contact_weighted", False))
			or bool(getattr(self.cfg, "demo_bc_force_band_weighted", False))
			or bool(getattr(self.cfg, "demo_bc_low_force_acquisition_weighted", False))
		):
			weights = torch.ones(len(self._demo_action), device=self.agent.device)
			force_idx = int(getattr(self.cfg, "force_obs_idx", -1))
			if 0 <= force_idx < self._demo_obs.shape[-1]:
				force = self._demo_obs[:, force_idx].abs()
				if bool(getattr(self.cfg, "demo_bc_contact_weighted", False)):
					threshold = float(getattr(self.cfg, "demo_bc_contact_threshold_n", 0.5))
					contact_weight = float(getattr(self.cfg, "demo_bc_contact_weight", 8.0))
					weights = weights + (force >= threshold).float() * contact_weight
				if (
					bool(getattr(self.cfg, "demo_bc_force_band_weighted", False))
					or bool(getattr(self.cfg, "demo_bc_low_force_acquisition_weighted", False))
				):
					target_idx = int(getattr(self.cfg, "demo_bc_force_target_obs_idx", getattr(self.cfg, "force_plan_target_obs_idx", -1)))
					if 0 <= target_idx < self._demo_obs.shape[-1]:
						target_force = self._demo_obs[:, target_idx].abs()
					else:
						target_force = torch.full_like(force, float(getattr(self.cfg, "demo_bc_force_target_n", 7.5)))
				if bool(getattr(self.cfg, "demo_bc_low_force_acquisition_weighted", False)):
					low_target = target_force <= float(getattr(self.cfg, "demo_bc_low_force_acquisition_target_threshold_n", 5.5))
					low_force = force < float(getattr(self.cfg, "demo_bc_low_force_acquisition_max_force_n", 3.0))
					acq_mask = (low_target & low_force).float()
					acq_weight = float(getattr(self.cfg, "demo_bc_low_force_acquisition_weight", 8.0))
					weights = weights + acq_mask * acq_weight
					self._demo_bc_low_force_acquisition_stats = {
						"target_idx": target_idx,
						"fraction": float(acq_mask.mean()),
						"mean_force": float(force[acq_mask.bool()].mean()) if bool(acq_mask.bool().any()) else float("nan"),
						"mean_target": float(target_force[acq_mask.bool()].mean()) if bool(acq_mask.bool().any()) else float("nan"),
					}
				if bool(getattr(self.cfg, "demo_bc_force_band_weighted", False)):
					min_contact = float(getattr(self.cfg, "demo_bc_force_band_min_contact_n", 0.5))
					band_fraction = float(getattr(self.cfg, "demo_bc_force_band_fraction", 0.0))
					width_floor = max(float(getattr(self.cfg, "demo_bc_force_band_width_n", 2.0)), 1e-6)
					if band_fraction > 0.0:
						width = torch.clamp(target_force * band_fraction, min=width_floor)
					else:
						width = torch.full_like(target_force, width_floor)
					band_weight = float(getattr(self.cfg, "demo_bc_force_band_weight", 8.0))
					force_error = (force - target_force).abs()
					band_score = (1.0 - force_error / width).clamp(min=0.0, max=1.0)
					band_score = band_score * (force >= min_contact).float()
					weights = weights + band_score * band_weight
					self._demo_bc_force_band_stats = {
						"target_idx": target_idx,
						"mean_force": float(force.mean()),
						"mean_target": float(target_force.mean()),
						"mean_width": float(width.mean()),
						"band_fraction": float((band_score > 0).float().mean()),
						"over_fraction": float((force > target_force + width).float().mean()),
					}
			progress_idx = int(getattr(self.cfg, "demo_bc_progress_obs_idx", -1))
			if 0 <= progress_idx < self._demo_obs.shape[-1]:
				progress = self._demo_obs[:, progress_idx].clamp(min=0)
				progress_weight = float(getattr(self.cfg, "demo_bc_progress_weight", 0.0))
				weights = weights + progress * progress_weight
			weights = weights.clamp(min=1e-6)
			self._demo_bc_weight = weights / weights.sum()
			eff_contact = float((weights > 1.0).float().mean()) if len(weights) else 0.0
			print(
				"Weighted demo BC enabled: "
				f"force_idx={force_idx}, "
				f"threshold={float(getattr(self.cfg, 'demo_bc_contact_threshold_n', 0.5)):.3f}, "
				f"force_band={bool(getattr(self.cfg, 'demo_bc_force_band_weighted', False))}, "
				f"band_width={float(getattr(self.cfg, 'demo_bc_force_band_width_n', 2.0)):.3f}, "
				f"weighted_fraction={eff_contact:.3f}"
			)
			if hasattr(self, "_demo_bc_force_band_stats"):
				stats = self._demo_bc_force_band_stats
				print(
					"Force-band demo stats: "
					f"target_idx={stats['target_idx']}, "
					f"mean_force={stats['mean_force']:.3f}, "
					f"mean_target={stats['mean_target']:.3f}, "
					f"mean_width={stats['mean_width']:.3f}, "
					f"band_fraction={stats['band_fraction']:.3f}, "
					f"over_fraction={stats['over_fraction']:.3f}"
				)
			if hasattr(self, "_demo_bc_low_force_acquisition_stats"):
				stats = self._demo_bc_low_force_acquisition_stats
				print(
					"Low-force acquisition demo stats: "
					f"target_idx={stats['target_idx']}, "
					f"fraction={stats['fraction']:.3f}, "
					f"mean_force={stats['mean_force']:.3f}, "
					f"mean_target={stats['mean_target']:.3f}"
				)

	def _sample_demo_bc_batch(self):
		"""Sample scripted demo pairs for optional online BC regularization."""
		if (
			self.cfg.demo_bc_online_coef <= 0
			or not hasattr(self, "_demo_tds")
		):
			return None
		self._prepare_demo_bc_data()
		if len(self._demo_action) == 0:
			return None
		batch_size = min(int(self.cfg.demo_bc_online_batch_size), len(self._demo_action))
		if getattr(self, "_demo_bc_weight", None) is not None:
			idx = torch.multinomial(self._demo_bc_weight, batch_size, replacement=True)
		else:
			idx = torch.randint(len(self._demo_action), (batch_size,), device=self.agent.device)
		return self._demo_obs[idx], self._demo_action[idx]

	def _prepare_demo_plan_data(self):
		"""Cache successful demo snippets as MPPI candidate trajectories."""
		if not self.cfg.demo_plan or not hasattr(self, "_demo_tds"):
			return
		horizon = int(self.cfg.horizon)
		obs_rows, action_rows = [], []
		for demo_td in self._demo_tds:
			obs = demo_td["obs"][:-1]
			action = demo_td["action"][1:]
			if len(action) < horizon:
				continue
			valid = ~torch.isnan(action).any(dim=-1)
			max_start = len(action) - horizon + 1
			for start in range(max_start):
				if bool(valid[start:start+horizon].all()):
					obs_rows.append(obs[start])
					action_rows.append(action[start:start+horizon])
		if not action_rows:
			self._demo_plan_library_size = 0
			self.agent.set_demo_plan_library(None, None)
			print("Demo plan library is empty; MPPI demo proposals disabled.")
			return
		demo_obs = torch.stack(obs_rows).to(self.agent.device)
		demo_actions = torch.stack(action_rows).to(self.agent.device)
		self._demo_plan_library_size = len(demo_actions)
		self.agent.set_demo_plan_library(demo_obs, demo_actions)
		print(
			"Demo plan library "
			f"snippets={self._demo_plan_library_size}, "
			f"horizon={horizon}, "
			f"proposal_trajs={self.cfg.demo_plan_num_trajs}"
		)

	def _prepare_demo_trajectory_data(self):
		"""Cache full successful demo trajectories for phase-aligned action blending."""
		if not self.cfg.demo_trajectory_blend or not hasattr(self, "_demo_tds"):
			return
		trajectories = []
		for demo_td in self._demo_tds:
			obs = demo_td["obs"][:-1]
			action = demo_td["action"][1:]
			if len(action) == 0:
				continue
			valid = ~torch.isnan(action).any(dim=-1)
			obs = obs[valid].cpu()
			action = action[valid].cpu()
			if len(action) == 0:
				continue
			progress_idx = 38 if obs.shape[-1] > 38 else 37
			progress = obs[:, progress_idx].cpu()
			force = obs[:, self.cfg.force_obs_idx].cpu()
			contact_candidates = torch.nonzero(force >= float(self.env.unwrapped.min_wipe_force_n)).flatten()
			contact_start = int(contact_candidates[0]) if len(contact_candidates) > 0 else 0
			wipe_candidates = torch.nonzero(progress > 0.02).flatten()
			wipe_start = int(wipe_candidates[0]) if len(wipe_candidates) > 0 else len(action) - 1
			trajectories.append({
				"obs": obs,
				"action": action,
				"progress": progress,
				"force": force,
				"contact_start": contact_start,
				"wipe_start": wipe_start,
			})
		self._demo_trajectories = trajectories
		self._demo_trajectory_library_size = len(trajectories)
		if len(trajectories) == 0:
			print("Demo trajectory library is empty; phase-aligned blend disabled.")
			return
		avg_len = float(np.mean([len(t["action"]) for t in trajectories]))
		print(
			"Demo trajectory library "
			f"trajectories={len(trajectories)}, "
			f"avg_len={avg_len:.1f}"
		)

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False
		self._assert_wipe_band_consistency()
		apply_curriculum_after_prefill = bool(getattr(self.cfg, "wipe_curriculum_after_demo_prefill", False))
		if not apply_curriculum_after_prefill:
			self._apply_wipe_curriculum()
		self._prefill_force_wipe_demos()
		self._prefill_force_press_demos()
		if apply_curriculum_after_prefill:
			self._apply_wipe_curriculum()
		while self._step <= self.cfg.steps:
			self._apply_wipe_curriculum()
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					self._apply_wipe_curriculum()
					eval_metrics = self.eval()
					if bool(getattr(self.cfg, "save_eval_checkpoints", False)):
						min_stage = int(getattr(self.cfg, "save_eval_checkpoints_min_stage", -1))
						stage = -1 if self._wipe_curriculum_stage is None else int(self._wipe_curriculum_stage)
						if stage >= min_stage:
							model_dir = Path(self.cfg.work_dir) / "models"
							model_dir.mkdir(parents=True, exist_ok=True)
							eval_path = model_dir / f"eval_step_{int(self._step)}_stage_{stage}.pt"
							model = (
								self.agent._eval_ema_model
								if bool(getattr(self.cfg, "eval_ema", False))
								and self.agent._eval_ema_model is not None
								else self.agent.model
							)
							torch.save({
								"model": model.state_dict(),
								"step": self._step,
								"curriculum_stage": stage,
								"episode_success": float(eval_metrics.get("episode_success", 0.0)),
							}, eval_path)
							print(f" eval_checkpoint I: {self._step:,} stage: {stage} success: {float(eval_metrics.get('episode_success', 0.0)):.3f} path: {eval_path}")
					if bool(getattr(self.cfg, "save_best_eval", False)):
						min_best_stage = int(getattr(self.cfg, "save_best_eval_min_stage", -1))
						current_stage = -1 if self._wipe_curriculum_stage is None else int(self._wipe_curriculum_stage)
						score = float(eval_metrics.get("episode_success", 0.0))
						# A later curriculum stage supersedes any earlier-stage best,
						# so an easy stage-0 score can never lock out the full path.
						if current_stage > self._best_eval_stage:
							self._best_eval_success = -np.inf
						if current_stage >= min_best_stage and score > self._best_eval_success:
							self._best_eval_success = score
							self._best_eval_stage = current_stage
							model_dir = Path(self.cfg.work_dir) / "models"
							model_dir.mkdir(parents=True, exist_ok=True)
							best_path = model_dir / "best_eval.pt"
							model = (
								self.agent._eval_ema_model
								if bool(getattr(self.cfg, "eval_ema", False))
								and self.agent._eval_ema_model is not None
								else self.agent.model
							)
							torch.save({"model": model.state_dict(), "step": self._step, "episode_success": score}, best_path)
							self._write_csv_row(self._best_eval_path, {
								"step": self._step,
								"episode_success": score,
								"episode_reward": float(eval_metrics.get("episode_reward", np.nan)),
								"final_wipe_progress": float(eval_metrics.get("final_wipe_progress", np.nan)),
								"final_force_error": float(eval_metrics.get("final_force_error", np.nan)),
								"best_path": str(best_path),
							})
							print(f" best_eval       I: {self._step:,} success: {score:.3f} path: {best_path}")
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				self._apply_wipe_curriculum()
				obs = self.env.reset()
				obs = self._maybe_demo_state_reset(obs)
				self._reset_dagger_teacher()
				self._reset_demo_trajectory(obs)
				self._reset_force_press_pi()
				self._reset_hybrid_floor()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			self._maybe_collect_dagger_correction(obs, action)
			action = self._apply_demo_trajectory_blend(obs, action, eval_mode=False)
			action = self._apply_demo_action_blend(obs, action, eval_mode=False)
			action = self._apply_contact_force_control(obs, action)
			action = self._apply_residual_force_scaffold(obs, action)
			action = self._apply_end_contact_recovery(obs, action)
			action = self._apply_hybrid_force_floor(obs, action)
			action = self._apply_force_safe_action(obs, action)
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(
						self.buffer,
						demo_batch=self._sample_demo_bc_batch(),
						corrective_batch=self._sample_dagger_batch(),
					)
				train_metrics.update(_train_metrics)
				update_diag_freq = max(1, int(getattr(self.cfg, "update_diagnostics_freq", 250)))
				if self._step >= self._next_update_diag_step:
					self._write_csv_row(self._update_diag_path, {
						"step": self._step,
						"force_loss": self._scalar(_train_metrics.get("force_loss")),
						"force_regime_loss": self._scalar(_train_metrics.get("force_regime_loss")),
						"force_regime_accuracy": self._scalar(_train_metrics.get("force_regime_accuracy")),
						"force_regime_actor_loss": self._scalar(_train_metrics.get("force_regime_actor_loss")),
						"phase_policy_loss": self._scalar(_train_metrics.get("phase_policy_loss")),
						"phase_policy_accuracy": self._scalar(_train_metrics.get("phase_policy_accuracy")),
						"teacher_anchor_loss": self._scalar(_train_metrics.get("teacher_anchor_loss")),
						"teacher_encoder_anchor_loss": self._scalar(_train_metrics.get("teacher_encoder_anchor_loss")),
						"dagger_bc_loss": self._scalar(_train_metrics.get("dagger_bc_loss")),
						"dagger_samples": self._dagger_size,
						"dagger_states_seen": self._dagger_seen,
						"dagger_label_mse_mean": (
							self._dagger_action_mse_sum / self._dagger_seen
							if self._dagger_seen > 0
							else np.nan
						),
						"dagger_low_seen": int(self._dagger_phase_seen[0]),
						"dagger_in_band_seen": int(self._dagger_phase_seen[1]),
						"dagger_high_seen": int(self._dagger_phase_seen[2]),
						"pi_loss": self._scalar(_train_metrics.get("pi_loss")),
					})
					self._next_update_diag_step = self._step + update_diag_freq

			self._step += 1

		self.logger.finish(self.agent)
