"""MuJoCo task-dynamics port for zero-shot ForceWipe robustness checks.

The learned policy sees the same 16-dimensional causal task observation and
issues the same normalized task-frame Cartesian displacement as in the PhysX
evaluation.  The port reproduces the explicit compliant tool and supported
surface actors, but replaces the Panda/PhysX execution stack with a calibrated
Cartesian target, MuJoCo contacts, and translational spring--damper dynamics.
It is therefore a cross-engine task-dynamics robustness test, not a hardware
surrogate or a full robot-model equivalence claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import math
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from forcewipe.simulation.plant_numerics import equivalent_explicit_damping
from forcewipe.simulation.scenarios import ScenarioSpec, make_scenario_path, surface_normal
from forcewipe.simulation.scene_geometry import (
    V4_TOOL_TCP_CENTER_OFFSET_M,
    make_surface_geometry,
    make_tool_geometry,
)
from forcewipe.direct.tdmpc2_direct_firstpass import (
    DirectFirstPassConfig,
    DirectFirstPassContractError,
    DirectFirstPassState,
    SafeContactDoseTracker,
    direct_first_pass_reward,
    encode_observation,
    task_frame_cartesian_delta,
)


@dataclass(frozen=True)
class MuJoCoContactConfig:
    physics_timestep_s: float = 0.001
    control_timestep_s: float = 0.01
    contact_time_constant_s: float = 0.005
    contact_damping_ratio: float = 1.0
    contact_margin_m: float = 0.0
    contact_impedance_min: float = 0.9
    contact_impedance_max: float = 0.95
    contact_impedance_width_m: float = 0.001
    tool_drive_scale: float = 1.0
    tool_tangential_drive_scale: float | None = None
    support_drive_scale: float = 1.0
    friction_scale: float = 1.0
    solver_iterations: int = 50
    actuation_gain: float = 1.0
    actuation_time_constant_s: float = 0.0
    actuation_delay_steps: int = 0

    def validate(self) -> None:
        values = (
            self.physics_timestep_s,
            self.control_timestep_s,
            self.contact_time_constant_s,
            self.contact_damping_ratio,
            self.contact_margin_m,
            self.contact_impedance_min,
            self.contact_impedance_max,
            self.contact_impedance_width_m,
            self.tool_drive_scale,
            self.support_drive_scale,
            self.friction_scale,
            self.actuation_gain,
            self.actuation_time_constant_s,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("MuJoCo contact configuration must be finite")
        if (
            self.tool_tangential_drive_scale is not None
            and not math.isfinite(float(self.tool_tangential_drive_scale))
        ):
            raise ValueError("MuJoCo contact configuration must be finite")
        if self.physics_timestep_s <= 0 or self.control_timestep_s <= 0:
            raise ValueError("timesteps must be positive")
        ratio = self.control_timestep_s / self.physics_timestep_s
        if not math.isclose(ratio, round(ratio), abs_tol=1e-12):
            raise ValueError("control timestep must contain an integer number of physics steps")
        if self.contact_time_constant_s < 2.0 * self.physics_timestep_s:
            raise ValueError("contact time constant must satisfy MuJoCo's refsafety bound")
        if self.contact_damping_ratio <= 0 or self.contact_margin_m < 0:
            raise ValueError("contact damping must be positive and margin nonnegative")
        if not 0.0 < self.contact_impedance_min < 1.0:
            raise ValueError("contact_impedance_min must lie in (0, 1)")
        if not self.contact_impedance_min <= self.contact_impedance_max < 1.0:
            raise ValueError(
                "contact_impedance_max must lie in [contact_impedance_min, 1)"
            )
        if self.contact_impedance_width_m <= 0.0:
            raise ValueError("contact_impedance_width_m must be positive")
        if self.tool_drive_scale <= 0.0 or self.support_drive_scale <= 0.0:
            raise ValueError("drive implementation scales must be positive")
        if (
            self.tool_tangential_drive_scale is not None
            and self.tool_tangential_drive_scale <= 0.0
        ):
            raise ValueError("tool_tangential_drive_scale must be positive or null")
        if self.friction_scale <= 0.0:
            raise ValueError("friction_scale must be positive")
        if int(self.solver_iterations) < 1:
            raise ValueError("solver_iterations must be positive")
        if not 0.0 < self.actuation_gain <= 1.0:
            raise ValueError("actuation_gain must lie in (0, 1]")
        if self.actuation_time_constant_s < 0.0:
            raise ValueError("actuation_time_constant_s must be nonnegative")
        if int(self.actuation_delay_steps) != self.actuation_delay_steps or self.actuation_delay_steps < 0:
            raise ValueError("actuation_delay_steps must be a nonnegative integer")

    @property
    def substeps(self) -> int:
        return int(round(self.control_timestep_s / self.physics_timestep_s))

    @property
    def actuation_response_fraction(self) -> float:
        """Causal first-order servo response applied once per control sample."""
        if self.actuation_time_constant_s == 0.0:
            return 1.0
        return float(-math.expm1(
            -self.control_timestep_s / self.actuation_time_constant_s
        ))

    @property
    def effective_tangential_drive_scale(self) -> float:
        return float(
            self.tool_drive_scale
            if self.tool_tangential_drive_scale is None
            else self.tool_tangential_drive_scale
        )


def _numbers(values: np.ndarray) -> str:
    array = np.asarray(values).reshape(-1)
    return " ".join(f"{float(value):.17g}" for value in array)


def _mesh_asset(vertices: np.ndarray, faces: np.ndarray) -> str:
    return (
        '<mesh name="surface_mesh" vertex="'
        + html.escape(_numbers(vertices))
        + '" face="'
        + html.escape(" ".join(str(int(value)) for value in np.asarray(faces).reshape(-1)))
        + '"/>'
    )


def _xml(spec: ScenarioSpec, contact: MuJoCoContactConfig) -> str:
    surface = make_surface_geometry(spec)
    tool = make_tool_geometry(spec)
    if surface.geometry_kind == "box":
        asset = ""
        surface_geom = (
            f'<geom name="surface_geom" type="box" size="{_numbers(surface.box_half_sizes_m)}" '
            f'quat="{_numbers(surface.quaternion_wxyz)}" mass="{spec.effective_mass_kg:.17g}"/>'
        )
    else:
        asset = _mesh_asset(surface.mesh_vertices_m, surface.mesh_faces)
        surface_geom = (
            f'<geom name="surface_geom" type="mesh" mesh="surface_mesh" '
            f'quat="{_numbers(surface.quaternion_wxyz)}" mass="{spec.effective_mass_kg:.17g}"/>'
        )
    sliding_friction = spec.friction_coefficient * contact.friction_scale
    friction = f"{sliding_friction:.17g} {sliding_friction:.17g} 0.001"
    return f"""
<mujoco model="forcewipe_crosssim">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{contact.physics_timestep_s:.17g}" gravity="0 0 -9.81"
          integrator="implicitfast" solver="Newton" iterations="{int(contact.solver_iterations)}"/>
  <size nconmax="256" njmax="2000"/>
  <default>
    <joint damping="0" armature="0"/>
    <geom friction="{friction}" condim="4"
          margin="{contact.contact_margin_m:.17g}" gap="0"
          solref="{contact.contact_time_constant_s:.17g} {contact.contact_damping_ratio:.17g}"
          solimp="{contact.contact_impedance_min:.17g} {contact.contact_impedance_max:.17g} {contact.contact_impedance_width_m:.17g} 0.5 2"/>
  </default>
  <asset>{asset}</asset>
  <worldbody>
    <body name="surface" pos="{_numbers(surface.position_xyz_m)}">
      <joint name="surface_x" type="slide" axis="1 0 0"/>
      <joint name="surface_y" type="slide" axis="0 1 0"/>
      <joint name="surface_z" type="slide" axis="0 0 1"/>
      {surface_geom}
    </body>
    <body name="tool" pos="0 0 {0.1698 - V4_TOOL_TCP_CENTER_OFFSET_M:.17g}">
      <joint name="tool_x" type="slide" axis="1 0 0"/>
      <joint name="tool_y" type="slide" axis="0 1 0"/>
      <joint name="tool_z" type="slide" axis="0 0 1"/>
      <geom name="tool_geom" type="box" size="{_numbers(tool.half_sizes_xyz_m)}"
            mass="0.08"/>
    </body>
  </worldbody>
</mujoco>
"""


class MuJoCoDirectFirstPassEnv(gym.Env):
    """Direct ForceWipe environment with MuJoCo contact/task dynamics."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        scenario_spec: ScenarioSpec,
        config: DirectFirstPassConfig | None = None,
        contact_config: MuJoCoContactConfig | None = None,
    ) -> None:
        super().__init__()
        scenario_spec.validate()
        self.spec = scenario_spec
        self.config = config or DirectFirstPassConfig()
        self.config.validate()
        self.contact_config = contact_config or MuJoCoContactConfig()
        self.contact_config.validate()
        if self.spec.disturbance_kind != "none" or self.spec.disturbance_scale != 0.0:
            raise ValueError(
                "the initial cross-engine study is restricted to disturbance-free "
                "factor-separated blocks; disturbance equivalence is not calibrated"
            )
        if not math.isclose(self.contact_config.control_timestep_s,
                            1.0 / self.config.control_rate_hz, abs_tol=1e-12):
            raise ValueError("MuJoCo and ForceWipe control timesteps differ")
        self.target_force_n = float(self.spec.target_force_n)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.full((16,), -2.0, dtype=np.float32),
            high=np.full((16,), 2.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.max_episode_steps = self.config.maximum_steps
        self.path = make_scenario_path(self.spec, points=801)
        self.surface_geometry = make_surface_geometry(self.spec)
        self.model = mujoco.MjModel.from_xml_string(_xml(self.spec, self.contact_config))
        self.data = mujoco.MjData(self.model)
        self._tool_joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"tool_{axis}")
            for axis in "xyz"
        ])
        self._surface_joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"surface_{axis}")
            for axis in "xyz"
        ])
        self._tool_dofs = np.array([self.model.jnt_dofadr[index] for index in self._tool_joint_ids])
        self._surface_dofs = np.array([self.model.jnt_dofadr[index] for index in self._surface_joint_ids])
        self._tool_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tool")
        self._surface_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "surface")
        self._tool_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tool_geom")
        self._surface_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "surface_geom")
        effective_tool_k = (
            self.contact_config.tool_drive_scale
            * self.spec.tool_normal_stiffness_n_m
        )
        effective_tangential_k = (
            4.0
            * self.contact_config.effective_tangential_drive_scale
            * self.spec.tool_normal_stiffness_n_m
        )
        self._tool_k = np.array([
            effective_tangential_k,
            effective_tangential_k,
            effective_tool_k,
        ])
        normal_d = equivalent_explicit_damping(
            0.08, 2.0 * math.sqrt(effective_tool_k * 0.08), 0.01
        )
        tangential_d = equivalent_explicit_damping(
            0.08, 2.0 * math.sqrt(effective_tangential_k * 0.08), 0.01
        )
        self._tool_d = np.array([tangential_d, tangential_d, normal_d])
        effective_support_k = (
            self.contact_config.support_drive_scale
            * self.spec.support_stiffness_n_m
        )
        support_d = equivalent_explicit_damping(
            self.spec.effective_mass_kg,
            2.0
            * self.spec.damping_ratio
            * math.sqrt(effective_support_k * self.spec.effective_mass_kg),
            0.01,
        )
        self._support_k = np.full(3, effective_support_k)
        self._support_d = np.full(3, support_d)
        self._surface_rest = np.asarray(self.surface_geometry.position_xyz_m, dtype=np.float64)
        self._initial_tool = np.array([0.0, 0.0, 0.1698 - V4_TOOL_TCP_CENTER_OFFSET_M])
        self._path_start = np.asarray(self.path.at(0.0)[0], dtype=np.float64)
        self._tcp_target = np.array([0.0, 0.0, 0.1698], dtype=np.float64)
        self._commanded_tcp_target = self._tcp_target.copy()
        self._actuation_command_history = [
            self._tcp_target.copy() for _ in range(self.contact_config.actuation_delay_steps)
        ]
        self._dose = SafeContactDoseTracker(self.config)
        self._previous_force = 0.0
        self._previous_action = np.zeros(3, dtype=np.float64)
        self._elapsed_steps = 0
        self._previous_progress = 0.0
        self._peak_force_n = 0.0
        self._force_limit_samples = 0
        self._last_geometry: dict[str, Any] | None = None

    def close(self) -> None:
        return None

    def _tool_position(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self._tool_body], dtype=np.float64).copy()

    def _surface_position(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self._surface_body], dtype=np.float64).copy()

    def _tool_velocity(self) -> np.ndarray:
        return np.asarray(self.data.qvel[self._tool_dofs], dtype=np.float64).copy()

    def _geometry(self) -> dict[str, Any]:
        tool = self._tool_position()
        relative = self._path_start + (tool - self._initial_tool)
        projection = self.path.project(relative)
        progress = float(np.clip(projection.progress, 0.0, 1.0))
        point = np.array(projection.point_xyz, dtype=np.float64, copy=True)
        tangent = np.array(projection.tangent_xyz, dtype=np.float64, copy=True)
        outward = np.array(
            surface_normal(self.spec, float(point[0])), dtype=np.float64, copy=True
        )
        outward /= np.linalg.norm(outward)
        tangent -= float(np.dot(tangent, outward)) * outward
        tangent /= np.linalg.norm(tangent)
        cross = np.cross(outward, tangent)
        cross /= np.linalg.norm(cross)
        error = relative - point
        velocity = self._tool_velocity()
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

    def _normal_force(self) -> float:
        total = 0.0
        values = np.zeros(6, dtype=np.float64)
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            if {int(contact.geom1), int(contact.geom2)} != {self._tool_geom, self._surface_geom}:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, values)
            total += max(float(values[0]), 0.0)
        return float(total)

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

    def _apply_springs(self) -> None:
        tool_displacement = (self._tcp_target - np.array([0.0, 0.0, V4_TOOL_TCP_CENTER_OFFSET_M])) - self._tool_position()
        tool_force = self._tool_k * tool_displacement - self._tool_d * self._tool_velocity()
        surface_displacement = self._surface_rest - self._surface_position()
        surface_velocity = np.asarray(self.data.qvel[self._surface_dofs], dtype=np.float64)
        surface_force = self._support_k * surface_displacement - self._support_d * surface_velocity
        self.data.qfrc_applied[self._tool_dofs] = tool_force
        self.data.qfrc_applied[self._surface_dofs] = surface_force

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options:
            raise DirectFirstPassContractError("cross-simulator scenario is immutable")
        mujoco.mj_resetData(self.model, self.data)
        self._tcp_target = np.array([0.0, 0.0, 0.1698], dtype=np.float64)
        self._commanded_tcp_target = self._tcp_target.copy()
        self._actuation_command_history = [
            self._tcp_target.copy() for _ in range(self.contact_config.actuation_delay_steps)
        ]
        self._dose = SafeContactDoseTracker(self.config)
        self._previous_force = 0.0
        self._previous_action = np.zeros(3, dtype=np.float64)
        self._elapsed_steps = 0
        self._previous_progress = 0.0
        self._peak_force_n = 0.0
        self._force_limit_samples = 0
        self._last_geometry = None
        mujoco.mj_forward(self.model, self.data)
        force = self._normal_force()
        observation = self._observation(force, 0.0)
        return observation, {
            "target_force_n": self.target_force_n,
            "scenario_seed": int(self.spec.scenario_seed),
            "scenario_id": int(self.spec.scenario_id),
            "simulator": "MuJoCo",
            "direct_tdmpc2_action_authority": True,
            "force_dependent_action_projection": False,
            "rewiping_enabled": False,
        }

    def step(self, action: object):
        normalized = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if normalized.shape != (3,) or not np.all(np.isfinite(normalized)):
            raise DirectFirstPassContractError("TD-MPC2 action has an invalid shape/value")
        geometry = self._last_geometry or self._geometry()
        world_delta = task_frame_cartesian_delta(
            normalized,
            tangent_world=geometry["tangent_world"],
            outward_normal_world=geometry["outward_normal_world"],
            config=self.config,
        )
        commanded_world_delta = self.contact_config.actuation_gain * world_delta
        self._commanded_tcp_target += commanded_world_delta
        self._actuation_command_history.append(self._commanded_tcp_target.copy())
        delayed_tcp_target = self._actuation_command_history.pop(0)
        previous_tcp_target = self._tcp_target.copy()
        self._tcp_target += self.contact_config.actuation_response_fraction * (
            delayed_tcp_target - self._tcp_target
        )
        issued_world_delta = self._tcp_target - previous_tcp_target
        for _ in range(self.contact_config.substeps):
            self._apply_springs()
            mujoco.mj_step(self.model, self.data)
        self._elapsed_steps += 1
        force = self._normal_force()
        if not math.isfinite(force):
            raise DirectFirstPassContractError("native force must be one finite scalar")
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
        terminated = bool(outcome.force_limit_violation or outcome.success)
        truncated = self._elapsed_steps >= self.config.maximum_steps
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
            "tcp_target_world_m": tuple(float(value) for value in self._tcp_target),
            "tool_position_world_m": tuple(float(value) for value in self._tool_position()),
            "surface_position_world_m": tuple(float(value) for value in self._surface_position()),
            "tdmpc2_action": tuple(float(value) for value in normalized),
            "requested_world_delta_m": tuple(float(value) for value in world_delta),
            "commanded_world_delta_m": tuple(float(value) for value in commanded_world_delta),
            "issued_world_delta_m": tuple(float(value) for value in issued_world_delta),
            "calibrated_actuation_gain": float(self.contact_config.actuation_gain),
            "calibrated_actuation_time_constant_s": float(
                self.contact_config.actuation_time_constant_s
            ),
            "calibrated_actuation_delay_steps": int(
                self.contact_config.actuation_delay_steps
            ),
            "actuation_response_fraction": float(
                self.contact_config.actuation_response_fraction
            ),
            "direct_tdmpc2_action_authority": True,
            "force_dependent_action_projection": False,
            "rewiping_enabled": False,
            "simulator": "MuJoCo",
        }
        return observation, np.float32(outcome.reward), terminated, truncated, info
