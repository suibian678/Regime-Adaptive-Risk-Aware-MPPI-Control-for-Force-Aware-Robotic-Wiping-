from collections import defaultdict
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

import force_press.press_env  # noqa: F401
import force_press.wipe_env  # noqa: F401
from envs.wrappers.timeout import Timeout


FORCE_PRESS_TASKS = {
    "force-press": dict(
        env="ForcePress-v1",
        control_mode="pd_ee_delta_pos",
        max_episode_steps=80,
    ),
    "force-wipe": dict(
        env="ForceWipe-v1",
        control_mode="pd_ee_delta_pos",
        max_episode_steps=160,
    ),
}


class ForcePressWrapper(gym.Wrapper):
    """Adapt ManiSkill3 Gymnasium API to TD-MPC2's older env API."""

    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        target_values = str(
            getattr(cfg, "wipe_target_force_values", "") or ""
        ).strip()
        self._wipe_target_force_values = [
            float(value)
            for value in target_values.replace(",", "|").split("|")
            if value.strip()
        ]
        self._wipe_target_force_band_fraction = float(
            getattr(cfg, "wipe_target_force_band_fraction", 0.25)
        )
        self._wipe_target_rng = np.random.default_rng(
            int(getattr(cfg, "seed", 0)) + 9107
        )
        self._wipe_target_cycle = []
        self._wipe_target_cycle_index = 0
        self._wipe_action_basis = str(getattr(cfg, "wipe_action_basis", "delta_pos"))
        self._wipe_force_admittance_z_gain = float(
            getattr(cfg, "wipe_force_admittance_z_gain", 0.003)
        )
        self._wipe_force_admittance_z_residual_scale = float(
            getattr(cfg, "wipe_force_admittance_z_residual_scale", 0.020)
        )
        self._wipe_force_admittance_z_clip = float(
            getattr(cfg, "wipe_force_admittance_z_clip", 0.040)
        )
        self._wipe_phase_admittance_target_offset_n = float(
            getattr(cfg, "wipe_phase_admittance_target_offset_n", 3.0)
        )
        self._wipe_phase_admittance_gain_min = float(
            getattr(cfg, "wipe_phase_admittance_gain_min", 0.0015)
        )
        self._wipe_phase_admittance_gain_max = float(
            getattr(cfg, "wipe_phase_admittance_gain_max", 0.0060)
        )
        self._wipe_phase_admittance_z_clip = float(
            getattr(cfg, "wipe_phase_admittance_z_clip", 0.080)
        )
        self._wipe_phase_admittance_acq_residual_scale = float(
            getattr(cfg, "wipe_phase_admittance_acq_residual_scale", 0.010)
        )
        self._wipe_phase_admittance_safe_x = float(
            getattr(cfg, "wipe_phase_admittance_safe_x", 0.004)
        )
        self._wipe_native_admittance_target_offset_n = float(
            getattr(cfg, "wipe_native_admittance_target_offset_n", 3.0)
        )
        self._wipe_native_admittance_kp_min = float(
            getattr(cfg, "wipe_native_admittance_kp_min", 0.0015)
        )
        self._wipe_native_admittance_kp_max = float(
            getattr(cfg, "wipe_native_admittance_kp_max", 0.0060)
        )
        self._wipe_native_admittance_ki = float(
            getattr(cfg, "wipe_native_admittance_ki", 0.00008)
        )
        self._wipe_native_admittance_integral_clip = float(
            getattr(cfg, "wipe_native_admittance_integral_clip", 80.0)
        )
        self._wipe_native_admittance_z_clip = float(
            getattr(cfg, "wipe_native_admittance_z_clip", 0.080)
        )
        self._wipe_native_admittance_low_extra_down = float(
            getattr(cfg, "wipe_native_admittance_low_extra_down", 0.004)
        )
        self._wipe_native_admittance_high_extra_lift = float(
            getattr(cfg, "wipe_native_admittance_high_extra_lift", 0.004)
        )
        self._wipe_native_admittance_pi_without_force = bool(
            getattr(cfg, "wipe_native_admittance_pi_without_force", False)
        )
        self._wipe_native_contact_gate_x = bool(
            getattr(cfg, "wipe_native_contact_gate_x", False)
        )
        self._wipe_native_terminal_force_gate = bool(
            getattr(cfg, "wipe_native_terminal_force_gate", False)
        )
        self._wipe_native_terminal_gate_progress = float(
            getattr(cfg, "wipe_native_terminal_gate_progress", 0.88)
        )
        self._wipe_native_contact_gate_progress_start = float(
            getattr(cfg, "wipe_native_contact_gate_progress_start", 0.02)
        )
        self._wipe_native_force_integral = 0.0
        obs_space = self.env.observation_space
        if (
            hasattr(obs_space, "shape")
            and len(obs_space.shape) == 2
            and obs_space.shape[0] == 1
        ):
            obs_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_space.shape[1:],
                dtype=np.float32,
            )
        self.observation_space = obs_space
        self.action_space = gym.spaces.Box(
            low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
            high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
            dtype=np.float32,
        )

    def _set_next_wipe_target(self):
        if self.cfg.task != "force-wipe" or not self._wipe_target_force_values:
            return
        if self._wipe_target_cycle_index >= len(self._wipe_target_cycle):
            order = self._wipe_target_rng.permutation(
                len(self._wipe_target_force_values)
            )
            self._wipe_target_cycle = [
                self._wipe_target_force_values[int(index)] for index in order
            ]
            self._wipe_target_cycle_index = 0
        target_force = self._wipe_target_cycle[self._wipe_target_cycle_index]
        self._wipe_target_cycle_index += 1
        raw_env = self.env.unwrapped
        raw_env.target_force_n = float(target_force)
        band = self._wipe_target_force_band_fraction * float(target_force)
        raw_env.force_success_band_n = band
        # Optional audit fix: keep go-brake reward band consistent with the
        # runtime per-target success band. Some diagnostic branches disable
        # this to isolate phase-label changes from reward-shaping changes.
        if bool(getattr(self.cfg, "wipe_sync_go_brake_band_to_success", True)):
            if hasattr(raw_env, "go_brake_force_band_n"):
                raw_env.go_brake_force_band_n = band

    def reset(self, seed=None):
        self._set_next_wipe_target()
        self._wipe_native_force_integral = 0.0
        reset_seed = getattr(self.cfg, "seed", None) if seed is None else seed
        obs, _ = self.env.reset(seed=reset_seed)
        return self._to_numpy_obs(obs)

    def _current_wipe_normal_force(self):
        if self.cfg.task != "force-wipe":
            return 0.0
        raw_env = self.env.unwrapped
        if not hasattr(raw_env, "_normal_force"):
            return 0.0
        return self._scalar(raw_env._normal_force())

    def _current_wipe_z_gap(self):
        if self.cfg.task != "force-wipe":
            return 0.0
        raw_env = self.env.unwrapped
        if not hasattr(raw_env, "agent") or not hasattr(raw_env, "wipe_pad"):
            return 0.0
        tcp_z = self._scalar(raw_env.agent.tcp.pose.p[:, 2])
        pad_top_z = self._scalar(raw_env.wipe_pad.pose.p[:, 2]) + float(raw_env.pad_half_size[2])
        return float(tcp_z - pad_top_z)

    @staticmethod
    def _wipe_approach_z_from_gap(z_gap):
        if z_gap > 0.055:
            return -0.065
        if z_gap > 0.025:
            return -0.040
        if z_gap > 0.013:
            return -0.020
        return -0.010

    def _transform_wipe_action(self, action):
        if self.cfg.task != "force-wipe":
            return action
        if self._wipe_action_basis == "force_admittance_z":
            transformed = np.array(action, dtype=np.float32, copy=True)
            raw_env = self.env.unwrapped
            normal_force = self._current_wipe_normal_force()
            target_force = float(getattr(raw_env, "target_force_n", 0.0))
            base_z = self._wipe_force_admittance_z_gain * (normal_force - target_force)
            residual_z = self._wipe_force_admittance_z_residual_scale * transformed[..., 2]
            transformed[..., 2] = np.clip(
                base_z + residual_z,
                -self._wipe_force_admittance_z_clip,
                self._wipe_force_admittance_z_clip,
            )
            return transformed
        if self._wipe_action_basis == "native_admittance_z":
            transformed = np.array(action, dtype=np.float32, copy=True)
            raw_env = self.env.unwrapped
            normal_force = self._current_wipe_normal_force()
            z_gap = self._current_wipe_z_gap()
            target_force = float(getattr(raw_env, "target_force_n", 0.0))
            band = float(getattr(raw_env, "force_success_band_n", 0.25 * target_force))
            min_force = float(getattr(raw_env, "min_wipe_force_n", 3.0))
            max_force = float(getattr(raw_env, "max_safe_force_n", 15.0))
            low_edge = max(min_force, target_force - band)
            high_edge = min(max_force, target_force + band)
            tcp_x = self._scalar(raw_env.agent.tcp.pose.p[:, 0])
            start_x = float(raw_env.path_start_xy[0])
            end_x = float(raw_env.path_end_xy[0])
            tcp_progress = float(np.clip((tcp_x - start_x) / max(end_x - start_x, 1e-6), 0.0, 1.0))
            in_success_band = low_edge <= normal_force <= high_edge
            if (
                self._wipe_native_contact_gate_x
                and tcp_progress >= self._wipe_native_contact_gate_progress_start
                and normal_force < min_force
            ):
                transformed[..., 0] = np.minimum(transformed[..., 0], 0.0)
            if (
                self._wipe_native_terminal_force_gate
                and tcp_progress >= self._wipe_native_terminal_gate_progress
                and not in_success_band
            ):
                transformed[..., 0] = 0.0

            semantic_force = np.clip(transformed[..., 2], -1.0, 1.0)
            gain_norm = np.clip(transformed[..., -1], -1.0, 1.0)
            kp = self._wipe_native_admittance_kp_min + 0.5 * (gain_norm + 1.0) * (
                self._wipe_native_admittance_kp_max - self._wipe_native_admittance_kp_min
            )
            target_setpoint = np.clip(
                target_force + self._wipe_native_admittance_target_offset_n * semantic_force,
                min_force,
                max_force,
            )

            if (
                (not self._wipe_native_admittance_pi_without_force and normal_force < 0.5)
                or z_gap > 0.013
            ):
                z_cmd = self._wipe_approach_z_from_gap(z_gap)
                self._wipe_native_force_integral = 0.0
            else:
                error = normal_force - target_setpoint
                self._wipe_native_force_integral = float(
                    np.clip(
                        self._wipe_native_force_integral + error,
                        -self._wipe_native_admittance_integral_clip,
                        self._wipe_native_admittance_integral_clip,
                    )
                )
                z_cmd = kp * error + self._wipe_native_admittance_ki * self._wipe_native_force_integral
                if normal_force < low_edge:
                    z_cmd -= self._wipe_native_admittance_low_extra_down
                elif normal_force > high_edge:
                    z_cmd += self._wipe_native_admittance_high_extra_lift

            transformed[..., 2] = np.clip(
                z_cmd,
                -self._wipe_native_admittance_z_clip,
                self._wipe_native_admittance_z_clip,
            )
            transformed[..., -1] = -1.0
            return transformed
        if self._wipe_action_basis != "phase_admittance_z":
            return action

        transformed = np.array(action, dtype=np.float32, copy=True)
        raw_env = self.env.unwrapped
        normal_force = self._current_wipe_normal_force()
        z_gap = self._current_wipe_z_gap()
        target_force = float(getattr(raw_env, "target_force_n", 0.0))
        band = float(getattr(raw_env, "force_success_band_n", 0.25 * target_force))
        min_force = float(getattr(raw_env, "min_wipe_force_n", 3.0))
        max_force = float(getattr(raw_env, "max_safe_force_n", 15.0))
        low_edge = max(min_force, target_force - band)
        high_edge = min(max_force, target_force + band)

        semantic_z = np.clip(transformed[..., 2], -1.0, 1.0)
        gain_norm = np.clip(transformed[..., -1], -1.0, 1.0)
        gain = self._wipe_phase_admittance_gain_min + 0.5 * (gain_norm + 1.0) * (
            self._wipe_phase_admittance_gain_max - self._wipe_phase_admittance_gain_min
        )
        target_setpoint = np.clip(
            target_force + self._wipe_phase_admittance_target_offset_n * semantic_z,
            min_force,
            max_force,
        )

        if normal_force < 0.5 or z_gap > 0.013:
            base_z = self._wipe_approach_z_from_gap(z_gap)
            z_cmd = base_z + self._wipe_phase_admittance_acq_residual_scale * semantic_z
        else:
            z_cmd = gain * (normal_force - target_setpoint)
            if normal_force < low_edge:
                z_cmd -= 0.004
            elif normal_force > high_edge:
                z_cmd += 0.004

        transformed[..., 2] = np.clip(
            z_cmd,
            -self._wipe_phase_admittance_z_clip,
            self._wipe_phase_admittance_z_clip,
        )
        if normal_force >= 0.5 and (normal_force < low_edge or normal_force > high_edge):
            transformed[..., 0] = np.minimum(transformed[..., 0], self._wipe_phase_admittance_safe_x)
        transformed[..., -1] = -1.0
        return transformed

    def step(self, action):
        env_action = self._transform_wipe_action(action)
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        done = np.logical_or(terminated, truncated)
        info = defaultdict(float, info)
        info["success"] = self._scalar(info.get("success", False))
        info["terminated"] = self._scalar(terminated)
        info["truncated"] = self._scalar(truncated)
        info["normal_force"] = self._scalar(info.get("normal_force", 0.0))
        info["force_error"] = self._scalar(info.get("force_error", 0.0))
        info["xy_error"] = self._scalar(info.get("xy_error", 0.0))
        info["wipe_progress"] = self._scalar(info.get("wipe_progress", 0.0))
        info["tcp_progress"] = self._scalar(info.get("tcp_progress", 0.0))
        info["instant_wipe_progress"] = self._scalar(info.get("instant_wipe_progress", 0.0))
        info["force_band_progress"] = self._scalar(info.get("force_band_progress", 0.0))
        info["instant_force_band_progress"] = self._scalar(info.get("instant_force_band_progress", 0.0))
        info["force_band_contact"] = self._scalar(info.get("force_band_contact", 0.0))
        info["contact_flag"] = self._scalar(info.get("contact_flag", 0.0))
        info["force_phase_potential"] = self._scalar(info.get("force_phase_potential", 0.0))
        info["potential_shaping_reward"] = self._scalar(info.get("potential_shaping_reward", 0.0))
        info["target_force_n"] = float(
            getattr(self.env.unwrapped, "target_force_n", np.nan)
        )
        info["force_success_band_n"] = float(
            getattr(self.env.unwrapped, "force_success_band_n", np.nan)
        )
        return (
            self._to_numpy_obs(obs),
            self._scalar(reward),
            bool(self._scalar(done)),
            info,
        )

    @staticmethod
    def _scalar(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return float(np.asarray(value).reshape(-1)[0])

    @staticmethod
    def _to_numpy_obs(obs):
        if hasattr(obs, "detach"):
            obs = obs.detach().cpu().numpy().astype(np.float32)
        else:
            obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 2 and obs.shape[0] == 1:
            obs = obs[0]
        return obs


def make_env(cfg):
    if cfg.task not in FORCE_PRESS_TASKS:
        raise ValueError("Unknown task:", cfg.task)
    assert cfg.obs == "state", "ForcePress currently supports state observations only."
    task_cfg = FORCE_PRESS_TASKS[cfg.task]
    env_kwargs = {}
    if cfg.task == "force-wipe":
        env_kwargs = dict(
            path_end_x=float(cfg.wipe_path_end_x),
            success_progress_threshold=float(cfg.wipe_success_progress),
            target_force_n=float(cfg.wipe_target_force_n),
            force_success_band_n=float(cfg.wipe_force_success_band_n),
            min_wipe_force_n=float(cfg.wipe_min_wipe_force_n),
            max_safe_force_n=float(cfg.wipe_max_safe_force_n),
            progress_reward_weight=float(cfg.wipe_progress_reward_weight),
            instant_progress_reward_weight=float(cfg.wipe_instant_progress_reward_weight),
            force_band_regularization_weight=float(cfg.wipe_force_band_regularization_weight),
            force_band_regularization_clip=float(cfg.wipe_force_band_regularization_clip),
            force_band_regularization_start_progress=float(cfg.wipe_force_band_regularization_start_progress),
            force_retention_weight=float(cfg.wipe_force_retention_weight),
            low_force_contact_reward_weight=float(cfg.wipe_low_force_contact_reward_weight),
            low_force_no_contact_penalty_weight=float(cfg.wipe_low_force_no_contact_penalty_weight),
            low_force_threshold_n=float(cfg.wipe_low_force_threshold_n),
            low_force_contact_start_progress=float(cfg.wipe_low_force_contact_start_progress),
            force_band_progress_reward_weight=float(cfg.wipe_force_band_progress_reward_weight),
            instant_force_band_progress_reward_weight=float(cfg.wipe_instant_force_band_progress_reward_weight),
            force_band_progress_reward_start_progress=float(cfg.wipe_force_band_progress_reward_start_progress),
            over_force_penalty_weight=float(cfg.wipe_over_force_penalty_weight),
            over_force_penalty_start_n=float(cfg.wipe_over_force_penalty_start_n),
            over_force_penalty_start_progress=float(cfg.wipe_over_force_penalty_start_progress),
            go_brake_forward_reward_weight=float(cfg.wipe_go_brake_forward_reward_weight),
            go_brake_unsafe_forward_penalty_weight=float(cfg.wipe_go_brake_unsafe_forward_penalty_weight),
            go_brake_lift_reward_weight=float(cfg.wipe_go_brake_lift_reward_weight),
            go_brake_min_progress=float(cfg.wipe_go_brake_min_progress),
            go_brake_max_progress=float(cfg.wipe_go_brake_max_progress),
            go_brake_force_band_n=float(cfg.wipe_go_brake_force_band_n),
            go_brake_min_force_n=float(cfg.wipe_go_brake_min_force_n),
            go_brake_high_force_n=float(cfg.wipe_go_brake_high_force_n),
            go_brake_low_force_n=float(cfg.wipe_go_brake_low_force_n),
            go_brake_target_x=float(cfg.wipe_go_brake_target_x),
            go_brake_safe_x=float(cfg.wipe_go_brake_safe_x),
            go_brake_target_lift_z=float(cfg.wipe_go_brake_target_lift_z),
            go_brake_x_action_idx=int(cfg.wipe_go_brake_x_action_idx),
            go_brake_z_action_idx=int(cfg.wipe_go_brake_z_action_idx),
            path_alignment_reward_weight=float(cfg.wipe_path_alignment_reward_weight),
            path_alignment_start_progress=float(cfg.wipe_path_alignment_start_progress),
            path_alignment_y_gain=float(cfg.wipe_path_alignment_y_gain),
            path_alignment_progress_weight=float(cfg.wipe_path_alignment_progress_weight),
            potential_shaping_weight=float(cfg.wipe_potential_shaping_weight),
            potential_shaping_gamma=float(cfg.wipe_potential_shaping_gamma),
            potential_contact_fraction=float(cfg.wipe_potential_contact_fraction),
            explicit_force_regime_obs=bool(cfg.wipe_explicit_force_regime_obs),
            force_dynamics_obs=bool(cfg.wipe_force_dynamics_obs),
            force_history_obs_steps=int(cfg.wipe_force_history_obs_steps),
            force_obs_filter_alpha=float(cfg.wipe_force_obs_filter_alpha),
        )
    control_freq = int(getattr(cfg, "wipe_control_freq", 20) or 20)
    max_steps = int(task_cfg["max_episode_steps"] * control_freq / 20)
    freq_kwargs = {}
    if control_freq != 20:
        # Lifted-setting support (V432/V433): finer control rate, same
        # wall-clock episode length. register_env pins max_episode_steps,
        # so it must be overridden here explicitly.
        freq_kwargs = dict(
            sim_config=dict(control_freq=control_freq),
            max_episode_steps=max_steps,
        )
    env = gym.make(
        task_cfg["env"],
        obs_mode="state",
        reward_mode="dense",
        control_mode=task_cfg["control_mode"],
        render_mode=None,
        num_envs=1,
        sim_backend="physx_cpu",
        render_backend="none",
        **freq_kwargs,
        **env_kwargs,
    )
    env = ForcePressWrapper(env, cfg)
    env = Timeout(env, max_episode_steps=max_steps)
    return env
