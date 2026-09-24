"""SAPIEN environment for native TD-MPC2 first-pass Cartesian control.

The policy directly owns each 100-Hz task-frame Cartesian displacement.  The
adapter performs only a coordinate transform and standard actuator scaling;
it contains no force controller, safety shield, recovery state machine, or
re-wiping logic.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

import gymnasium as gym
import numpy as np

from forcewipe.direct.sapien_sandbox_binding import _quat_conjugate, _quat_rotate, _quaternion
from forcewipe.direct.tdmpc2_direct_firstpass import (
    DirectFirstPassConfig,
    DirectFirstPassContractError,
    DirectFirstPassState,
    SafeContactDoseTracker,
    direct_first_pass_reward,
    encode_observation,
    task_frame_cartesian_delta,
)


def nominal_direct_scenario(target_force_n: float, scenario_seed: int, scenario_id: int):
    from forcewipe.simulation.scenarios import ScenarioSpec

    spec = ScenarioSpec(
        scenario_id=int(scenario_id),
        role="DEV",
        seed_namespace="v5_dev_native_tdmpc2_direct_firstpass",
        scenario_seed=int(scenario_seed),
        path_kind="line",
        path_length_m=0.16,
        path_lateral_scale_m=0.0,
        surface_kind="flat",
        cylinder_radius_m=0.5,
        tool_kind="wide_soft",
        tool_width_m=0.05,
        tool_footprint_length_m=0.03,
        tool_normal_stiffness_n_m=650.0,
        friction_coefficient=0.5,
        support_stiffness_n_m=2500.0,
        effective_mass_kg=0.35,
        damping_ratio=0.8,
        restitution=0.05,
        residual_family="single_blob",
        residual_seed=int(scenario_seed) + 1_000_000,
        residual_severity=0.7,
        disturbance_kind="none",
        disturbance_scale=0.0,
        sensor_latency_steps=0,
        constraint_kind="none",
        obstacle_center_s=0.5,
        obstacle_half_width_s=0.0,
        target_force_n=float(target_force_n),
        residual_cleanability=1.0,
    )
    spec.validate()
    return spec


def _vector(value: Any, *, length: int = 3, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (1, length):
        array = array[0]
    array = array.reshape(-1)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise DirectFirstPassContractError(f"{name} has an invalid shape/value")
    return array.copy()


def _scalar(value: Any, *, name: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1 or not math.isfinite(float(array[0])):
        raise DirectFirstPassContractError(f"{name} must be one finite scalar")
    return float(array[0])


class V6DirectFirstPassEnv(gym.Env):
    """One first-pass episode with direct, unshielded TD-MPC2 Cartesian actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        target_force_n: float = 8.0,
        scenario_seed: int = 61_000_001,
        scenario_id: int = 6_100_001,
        config: DirectFirstPassConfig | None = None,
    ) -> None:
        super().__init__()
        if float(target_force_n) not in {5.0, 8.0, 12.0}:
            raise DirectFirstPassContractError("target force must be 5, 8, or 12 N")
        self.config = config or DirectFirstPassConfig()
        self.config.validate()
        self.target_force_n = float(target_force_n)
        self.scenario_seed = int(scenario_seed)
        self.scenario_id = int(scenario_id)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.full((16,), -2.0, dtype=np.float32),
            high=np.full((16,), 2.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.max_episode_steps = self.config.maximum_steps
        self._env = None
        self._raw = None
        self._spec = None
        self._path = None
        self._initial_tool = None
        self._path_start = None
        self._dose = None
        self._previous_force = 0.0
        self._previous_action = np.zeros(3, dtype=np.float64)
        self._elapsed_steps = 0
        self._previous_progress = 0.0
        self._last_geometry = None
        self._peak_force_n = 0.0
        self._force_limit_samples = 0

    def _close_physics(self) -> None:
        if self._env is not None:
            self._env.close()
        self._env = None
        self._raw = None

    def close(self) -> None:
        self._close_physics()

    def rand_act(self):
        import torch

        return torch.from_numpy(self.action_space.sample())

    def _geometry(self) -> dict[str, Any]:
        from forcewipe.simulation.scenarios import surface_normal

        tool = _vector(self._raw.v4_tool.pose.p, name="tool position")
        relative = self._path_start + (tool - self._initial_tool)
        projection = self._path.project(relative)
        progress = float(np.clip(projection.progress, 0.0, 1.0))
        point = _vector(projection.point_xyz, name="path projection")
        tangent = _vector(projection.tangent_xyz, name="path tangent")
        outward = _vector(
            surface_normal(self._spec, float(point[0])),
            name="surface normal",
        )
        outward /= np.linalg.norm(outward)
        tangent -= float(np.dot(tangent, outward)) * outward
        tangent /= np.linalg.norm(tangent)
        cross = np.cross(outward, tangent)
        cross /= np.linalg.norm(cross)
        error = relative - point
        velocity = _vector(self._raw.v4_tool.linear_velocity, name="tool velocity")
        return {
            "tool_position_world_m": tool,
            "progress": progress,
            "tangent_world": tangent,
            "outward_normal_world": outward,
            "cross_world": cross,
            "signed_cross_track_error_m": float(np.dot(error, cross)),
            "signed_normal_offset_m": float(np.dot(error, outward)),
            "tangent_velocity_m_s": float(np.dot(velocity, tangent)),
            "cross_track_velocity_m_s": float(np.dot(velocity, cross)),
            "outward_normal_velocity_m_s": float(np.dot(velocity, outward)),
        }

    def _observation(self, force_n: float, force_rate_n_s: float) -> np.ndarray:
        geometry = self._geometry()
        self._last_geometry = geometry
        return encode_observation(
            DirectFirstPassState(
                measured_force_n=force_n,
                target_force_n=self.target_force_n,
                force_rate_n_s=force_rate_n_s,
                progress=geometry["progress"],
                signed_cross_track_error_m=geometry["signed_cross_track_error_m"],
                signed_normal_offset_m=geometry["signed_normal_offset_m"],
                tangent_velocity_m_s=geometry["tangent_velocity_m_s"],
                cross_track_velocity_m_s=geometry["cross_track_velocity_m_s"],
                outward_normal_velocity_m_s=geometry["outward_normal_velocity_m_s"],
                previous_action=tuple(float(value) for value in self._previous_action),
                elapsed_steps=self._elapsed_steps,
                completed_dose_bins=self._dose.completed_bins,
                minimum_bin_dose=self._dose.minimum_bin_dose,
            ),
            self.config,
        )

    def _physical_action(self, normalized_action: np.ndarray) -> np.ndarray:
        geometry = self._last_geometry or self._geometry()
        world_delta = task_frame_cartesian_delta(
            normalized_action,
            tangent_world=geometry["tangent_world"],
            outward_normal_world=geometry["outward_normal_world"],
            config=self.config,
        )
        arm = self._raw.agent.controller.controllers["arm"]
        if tuple(self._raw.agent.controller.action_mapping["arm"]) != (0, 6):
            raise DirectFirstPassContractError("unexpected arm action mapping")
        root_quaternion_world = _quaternion(
            arm.root_link.pose.q,
            name="arm root quaternion",
        )
        root_delta = _quat_rotate(_quat_conjugate(root_quaternion_world), world_delta)
        actuator = root_delta / self.config.position_action_scale_m
        if np.any(np.abs(actuator) > 1.0 + 1e-12):
            raise DirectFirstPassContractError("direct Cartesian delta exceeds actuator box")
        result = np.zeros(7, dtype=np.float32)
        result[:3] = actuator.astype(np.float32)
        result[3:6] = 0.0
        result[6] = -1.0
        return result

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        import forcewipe.simulation.sapien_env  # noqa: F401
        from forcewipe.simulation.scenarios import make_scenario_path

        super().reset(seed=seed)
        self._close_physics()
        if options:
            if "target_force_n" in options:
                target = float(options["target_force_n"])
                if target not in {5.0, 8.0, 12.0}:
                    raise DirectFirstPassContractError("invalid reset target")
                self.target_force_n = target
            if "scenario_seed" in options:
                self.scenario_seed = int(options["scenario_seed"])
            if "scenario_id" in options:
                self.scenario_id = int(options["scenario_id"])
        self._spec = nominal_direct_scenario(
            self.target_force_n,
            self.scenario_seed,
            self.scenario_id,
        )
        self._path = make_scenario_path(self._spec, points=801)
        self._env = gym.make(
            "ForceWipeV4-v1",
            scenario_spec=asdict(self._spec),
            lifecycle_mode=True,
            num_envs=1,
            obs_mode="state_dict",
            reward_mode="dense",
            control_mode="pd_ee_target_delta_pose",
            render_mode=None,
            sim_backend="physx_cpu",
            render_backend="none",
            sim_config=dict(control_freq=100),
            max_episode_steps=self.config.maximum_steps + 10,
        )
        self._env.reset(seed=self.scenario_seed)
        self._raw = self._env.unwrapped
        self._initial_tool = _vector(self._raw.v4_tool.pose.p, name="initial tool")
        self._path_start = _vector(self._path.at(0.0)[0], name="path start")
        self._dose = SafeContactDoseTracker(self.config)
        self._previous_force = 0.0
        self._previous_action = np.zeros(3, dtype=np.float64)
        self._elapsed_steps = 0
        self._previous_progress = 0.0
        self._last_geometry = None
        self._peak_force_n = 0.0
        self._force_limit_samples = 0
        force = _scalar(self._raw._normal_force(), name="initial force")
        observation = self._observation(force, 0.0)
        return observation, {
            "target_force_n": self.target_force_n,
            "scenario_seed": self.scenario_seed,
            "scenario_id": self.scenario_id,
            "direct_tdmpc2_action_authority": True,
            "force_dependent_action_projection": False,
            "rewiping_enabled": False,
        }

    def step(self, action: object):
        if self._env is None or self._raw is None:
            raise DirectFirstPassContractError("reset must be called before step")
        normalized = np.clip(
            _vector(action, name="TD-MPC2 action"),
            -1.0,
            1.0,
        )
        physical_action = self._physical_action(normalized)
        _raw_obs, _raw_reward, env_terminated, env_truncated, _raw_info = self._env.step(
            physical_action
        )
        self._elapsed_steps += 1
        force = _scalar(self._raw._normal_force(), name="native force")
        force_rate = (force - self._previous_force) * self.config.control_rate_hz
        geometry = self._geometry()
        new_bins = self._dose.update(geometry["progress"], force)
        outcome = direct_first_pass_reward(
            previous_progress=self._previous_progress,
            current_progress=geometry["progress"],
            measured_force_n=force,
            target_force_n=self.target_force_n,
            previous_action=self._previous_action,
            action=normalized,
            new_dose_bins=new_bins,
            dose_complete=self._dose.complete,
            config=self.config,
        )
        self._peak_force_n = max(self._peak_force_n, force)
        self._force_limit_samples += int(force > self.config.force_limit_n)
        self._previous_action = normalized.copy()
        self._previous_force = force
        self._previous_progress = geometry["progress"]
        self._last_geometry = geometry
        observation = self._observation(force, force_rate)
        terminated = bool(env_terminated) or outcome.force_limit_violation or outcome.success
        truncated = bool(env_truncated) or self._elapsed_steps >= self.config.maximum_steps
        info = {
            "success": outcome.success,
            "terminated": terminated,
            "force_limit_violation": outcome.force_limit_violation,
            "force_limit_violation_samples": self._force_limit_samples,
            "peak_force_n": self._peak_force_n,
            "normal_force_n": force,
            "target_force_n": self.target_force_n,
            "progress": geometry["progress"],
            "completed_dose_bins": self._dose.completed_bins,
            "minimum_bin_dose": self._dose.minimum_bin_dose,
            "dose_complete": self._dose.complete,
            "physical_action": tuple(float(value) for value in physical_action),
            "tdmpc2_action": tuple(float(value) for value in normalized),
            "direct_tdmpc2_action_authority": True,
            "force_dependent_action_projection": False,
            "rewiping_enabled": False,
        }
        return observation, np.float32(outcome.reward), terminated, truncated, info
