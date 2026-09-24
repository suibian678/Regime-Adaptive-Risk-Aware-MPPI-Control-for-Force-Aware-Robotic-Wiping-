"""Dedicated SAPIEN scene for physically mapped ForceWipe V4 factors."""

from __future__ import annotations

from dataclasses import asdict
import math
import os
import tempfile
from typing import Any

import numpy as np
import sapien
import torch

from force_press.wipe_env import ForceWipeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from forcewipe.simulation.plant_numerics import audit_vertical_support, equivalent_explicit_damping
from forcewipe.simulation.disturbances import physical_disturbance_command
from forcewipe.simulation.scene_geometry import (
    SceneGeometryError,
    V4_SURFACE_HALF_X_M,
    V4_SURFACE_HALF_Y_M,
    V4_SURFACE_THICKNESS_M,
    V4_TOOL_TCP_CENTER_OFFSET_M,
    V4_TOOL_THICKNESS_M,
    make_surface_geometry,
    make_obstacle_geometry,
    make_tool_geometry,
    surface_world_height,
    wavefront_obj_text,
)
from forcewipe.simulation.scenarios import ScenarioPath, ScenarioSpec, make_scenario_path, surface_normal


def _set_actor_material(actor, *, friction: float, restitution: float) -> None:
    for entity in actor._objs:
        for component in entity.components:
            for shape in getattr(component, "collision_shapes", ()):
                material = shape.physical_material
                material.set_static_friction(float(friction))
                material.set_dynamic_friction(float(friction))
                material.set_restitution(float(restitution))


def _set_angular_drive(drive, *, stiffness: float, damping: float) -> None:
    for item in drive._objs:
        item.set_drive_property_slerp(
            float(stiffness),
            float(damping),
            3.4028234663852886e38,
            "force",
        )


@register_env("ForceWipeV4-v1", max_episode_steps=1200)
class ForceWipeV4Env(ForceWipeEnv):
    """First dedicated V4 physics scene with explicit tool and support actors.

    This implementation intentionally supports only the factor mappings that
    are already physical. Unsupported surfaces, disturbances, and constraints
    raise before the environment is built; no metadata-only fallback exists.
    """

    pad_half_size = (
        V4_SURFACE_HALF_X_M,
        V4_SURFACE_HALF_Y_M,
        V4_SURFACE_THICKNESS_M,
    )

    def __init__(
        self,
        *args,
        scenario_spec: ScenarioSpec | dict[str, Any],
        tool_mass_kg: float = 0.08,
        tool_tangential_stiffness_multiplier: float = 4.0,
        rotational_drive_stiffness: float = 250.0,
        rotational_drive_damping: float = 25.0,
        path_success_tolerance_m: float = 0.015,
        lifecycle_mode: bool = False,
        **kwargs,
    ):
        spec = (
            scenario_spec
            if isinstance(scenario_spec, ScenarioSpec)
            else ScenarioSpec(**dict(scenario_spec))
        )
        spec.validate()
        if spec.surface_kind not in {"flat", "incline_5", "incline_10", "cylinder_low"}:
            raise SceneGeometryError("unsupported V4 surface kind")
        if spec.disturbance_kind not in {
            "none",
            "lateral_impulse",
            "normal_impulse",
            "reference_noise",
            "sensor_noise_latency",
            "dynamic_height",
            "moving_obstacle",
        }:
            raise SceneGeometryError("unsupported V4 disturbance kind")
        tool_mass = float(tool_mass_kg)
        if not math.isfinite(tool_mass) or tool_mass <= 0.0:
            raise SceneGeometryError("tool_mass_kg must be finite and positive")
        if (
            not math.isfinite(float(path_success_tolerance_m))
            or float(path_success_tolerance_m) <= 0.0
        ):
            raise SceneGeometryError("path_success_tolerance_m must be positive")

        self.v4_scenario = spec
        self.v4_path: ScenarioPath = make_scenario_path(spec, points=801)
        self.v4_surface_geometry = make_surface_geometry(spec)
        self.v4_tool_geometry = make_tool_geometry(spec)
        self.v4_support_audit = audit_vertical_support(
            mass_kg=spec.effective_mass_kg,
            support_stiffness_n_m=spec.support_stiffness_n_m,
            damping_ratio=spec.damping_ratio,
            dt_s=0.01,
        )
        if not self.v4_support_audit.discrete_stable:
            raise SceneGeometryError("scenario fails the 100 Hz support stability gate")
        self.v4_tool_mass_kg = tool_mass
        self.v4_tool_tangential_stiffness_multiplier = float(
            tool_tangential_stiffness_multiplier
        )
        self.v4_rotational_drive_stiffness = float(rotational_drive_stiffness)
        self.v4_rotational_drive_damping = float(rotational_drive_damping)
        self.v4_path_success_tolerance_m = float(path_success_tolerance_m)
        if not isinstance(lifecycle_mode, (bool, np.bool_)):
            raise SceneGeometryError("lifecycle_mode must be a boolean")
        self.v4_lifecycle_mode = bool(lifecycle_mode)

        kwargs.pop("target_force_n", None)
        kwargs.pop("pad_mass_kg", None)
        kwargs.pop("pad_linear_damping", None)
        kwargs.pop("pad_spring_k_z", None)
        kwargs.pop("path_end_x", None)
        super().__init__(
            *args,
            target_force_n=spec.target_force_n,
            pad_mass_kg=spec.effective_mass_kg,
            pad_linear_damping=self.v4_support_audit.explicit_damping_n_s_m,
            pad_spring_k_z=spec.support_stiffness_n_m,
            path_end_x=spec.path_length_m,
            **kwargs,
        )

    def scenario_record(self) -> dict[str, Any]:
        return asdict(self.v4_scenario)

    def set_v4_tool_operational_compliance(
        self,
        *,
        normal_stiffness_n_m: float,
        normal_damping_n_s_m: float,
    ) -> None:
        """Apply one scalar primitive K/D through the frozen anisotropy ratio."""

        stiffness = float(normal_stiffness_n_m)
        damping = float(normal_damping_n_s_m)
        if not math.isfinite(stiffness) or not math.isfinite(damping):
            raise SceneGeometryError("tool compliance must be finite")
        if stiffness <= 0.0 or damping <= 0.0:
            raise SceneGeometryError("tool compliance must be positive")
        multiplier = self.v4_tool_tangential_stiffness_multiplier
        tangential_stiffness = multiplier * stiffness
        tangential_damping = math.sqrt(multiplier) * damping
        self.v4_tool_drive.set_drive_property_x(
            tangential_stiffness, tangential_damping
        )
        self.v4_tool_drive.set_drive_property_y(
            tangential_stiffness, tangential_damping
        )
        self.v4_tool_drive.set_drive_property_z(stiffness, damping)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()
        geometry = self.v4_surface_geometry
        pose = sapien.Pose(
            p=geometry.position_xyz_m.astype(np.float32),
            q=geometry.quaternion_wxyz.astype(np.float32),
        )
        anchor_builder = self.scene.create_actor_builder()
        anchor_builder.set_initial_pose(pose)
        self.v4_surface_anchor = anchor_builder.build_kinematic("v4_surface_anchor")
        surface_color = np.array([112, 145, 136, 255], dtype=np.float32) / 255.0
        if geometry.geometry_kind == "box" and geometry.box_half_sizes_m is not None:
            self.wipe_pad = actors.build_box(
                self.scene,
                half_sizes=geometry.box_half_sizes_m,
                color=surface_color,
                name="v4_surface",
                body_type="dynamic",
                initial_pose=pose,
            )
        elif (
            geometry.geometry_kind == "convex_cylinder_cap"
            and geometry.mesh_vertices_m is not None
            and geometry.mesh_faces is not None
        ):
            descriptor, mesh_path = tempfile.mkstemp(
                prefix="forcewipe_v4_cylinder_", suffix=".obj"
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                    stream.write(
                        wavefront_obj_text(
                            geometry.mesh_vertices_m,
                            geometry.mesh_faces,
                        )
                    )
                builder = self.scene.create_actor_builder()
                builder.add_convex_collision_from_file(mesh_path)
                builder.add_visual_from_file(
                    mesh_path,
                    material=sapien.render.RenderMaterial(base_color=surface_color),
                )
                builder.set_initial_pose(pose)
                self.wipe_pad = builder.build(name="v4_surface")
            finally:
                if os.path.exists(mesh_path):
                    os.unlink(mesh_path)
        else:
            raise SceneGeometryError("surface geometry has no physical actor mapping")
        self.wipe_pad.set_mass(self.v4_scenario.effective_mass_kg)
        self.wipe_pad.set_linear_damping(0.0)
        self.wipe_pad.set_angular_damping(0.0)
        _set_actor_material(
            self.wipe_pad,
            friction=self.v4_scenario.friction_coefficient,
            restitution=self.v4_scenario.restitution,
        )

        tool = self.v4_tool_geometry
        tcp_initial_p = np.array([0.0, 0.0, 0.1698], dtype=np.float32)
        tcp_initial_q = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        tool_center = tcp_initial_p + np.array(
            [0.0, 0.0, -V4_TOOL_TCP_CENTER_OFFSET_M], dtype=np.float32
        )
        self.v4_tool = actors.build_box(
            self.scene,
            half_sizes=tool.half_sizes_xyz_m,
            color=np.array([80, 105, 125, 255], dtype=np.float32) / 255.0,
            name="v4_tool",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=tool_center, q=tcp_initial_q),
        )
        self.v4_tool.set_mass(self.v4_tool_mass_kg)
        self.v4_tool.set_linear_damping(0.0)
        self.v4_tool.set_angular_damping(0.0)
        _set_actor_material(
            self.v4_tool,
            friction=self.v4_scenario.friction_coefficient,
            restitution=self.v4_scenario.restitution,
        )

        self.v4_surface_drive = self.scene.create_drive(
            self.v4_surface_anchor,
            sapien.Pose(),
            self.wipe_pad,
            sapien.Pose(),
        )
        support_k = self.v4_scenario.support_stiffness_n_m
        support_d = self.v4_support_audit.explicit_damping_n_s_m
        for setter in (
            self.v4_surface_drive.set_drive_property_x,
            self.v4_surface_drive.set_drive_property_y,
            self.v4_surface_drive.set_drive_property_z,
        ):
            setter(support_k, support_d)
        _set_angular_drive(
            self.v4_surface_drive,
            stiffness=self.v4_rotational_drive_stiffness,
            damping=self.v4_rotational_drive_damping,
        )

        tool_offset = sapien.Pose(p=[0.0, 0.0, V4_TOOL_TCP_CENTER_OFFSET_M])
        self.v4_tool_drive = self.scene.create_drive(
            self.agent.tcp,
            tool_offset,
            self.v4_tool,
            sapien.Pose(),
        )
        tool_k = self.v4_scenario.tool_normal_stiffness_n_m
        tangential_k = tool_k * self.v4_tool_tangential_stiffness_multiplier
        tool_d = equivalent_explicit_damping(
            self.v4_tool_mass_kg,
            2.0 * math.sqrt(tool_k * self.v4_tool_mass_kg),
            0.01,
        )
        tangential_d = equivalent_explicit_damping(
            self.v4_tool_mass_kg,
            2.0 * math.sqrt(tangential_k * self.v4_tool_mass_kg),
            0.01,
        )
        self.v4_tool_drive.set_drive_property_x(tangential_k, tangential_d)
        self.v4_tool_drive.set_drive_property_y(tangential_k, tangential_d)
        self.v4_tool_drive.set_drive_property_z(tool_k, tool_d)
        _set_angular_drive(
            self.v4_tool_drive,
            stiffness=self.v4_rotational_drive_stiffness,
            damping=self.v4_rotational_drive_damping,
        )

        self.v4_constraint_actor = None
        self.v4_constraint_geometry = None
        if self.v4_scenario.constraint_kind != "none":
            constraint = make_obstacle_geometry(
                self.v4_scenario,
                path=self.v4_path,
            )
            self.v4_constraint_geometry = constraint
            self.v4_constraint_actor = actors.build_box(
                self.scene,
                half_sizes=constraint.half_sizes_xyz_m,
                color=(
                    np.array([165, 62, 62, 210], dtype=np.float32) / 255.0
                    if self.v4_scenario.constraint_kind == "static_obstacle"
                    else np.array([207, 160, 42, 110], dtype=np.float32) / 255.0
                ),
                name="v4_spatial_constraint",
                body_type="static",
                add_collision=self.v4_scenario.constraint_kind == "static_obstacle",
                initial_pose=sapien.Pose(
                    p=constraint.position_xyz_m.astype(np.float32),
                    q=constraint.quaternion_wxyz.astype(np.float32),
                ),
            )

        self.v4_moving_obstacle = None
        self.v4_moving_obstacle_geometry = None
        if self.v4_scenario.disturbance_kind == "moving_obstacle":
            moving = make_obstacle_geometry(
                self.v4_scenario,
                path=self.v4_path,
                moving=True,
            )
            self.v4_moving_obstacle_geometry = moving
            self.v4_moving_obstacle = actors.build_box(
                self.scene,
                half_sizes=moving.half_sizes_xyz_m,
                color=np.array([78, 91, 162, 220], dtype=np.float32) / 255.0,
                name="v4_moving_obstacle",
                body_type="kinematic",
                initial_pose=sapien.Pose(
                    p=moving.position_xyz_m.astype(np.float32),
                    q=moving.quaternion_wxyz.astype(np.float32),
                ),
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            count = len(env_idx)
            surface_state = torch.zeros((count, 13), device=self.device)
            surface_state[:, :3] = torch.as_tensor(
                np.array(self.v4_surface_geometry.position_xyz_m, copy=True),
                dtype=surface_state.dtype,
                device=self.device,
            )
            surface_state[:, 3:7] = torch.as_tensor(
                np.array(self.v4_surface_geometry.quaternion_wxyz, copy=True),
                dtype=surface_state.dtype,
                device=self.device,
            )
            self.wipe_pad.set_state(surface_state)
            self.v4_surface_anchor.set_pose(
                sapien.Pose(
                    p=self.v4_surface_geometry.position_xyz_m.astype(np.float32),
                    q=self.v4_surface_geometry.quaternion_wxyz.astype(np.float32),
                )
            )

            tool_state = torch.zeros((count, 13), device=self.device)
            tool_state[:, :3] = torch.tensor(
                [0.0, 0.0, 0.1698 - V4_TOOL_TCP_CENTER_OFFSET_M],
                device=self.device,
            )
            tool_state[:, 3:7] = torch.tensor(
                [0.0, 1.0, 0.0, 0.0], device=self.device
            )
            self.v4_tool.set_state(tool_state)
            self._best_contact_progress = torch.zeros((count,), device=self.device)
            self._best_force_band_progress = torch.zeros((count,), device=self.device)
            self._force_phase_potential = torch.zeros((count,), device=self.device)
            self._force_filtered = torch.zeros((count,), device=self.device)
            self._force_delta = torch.zeros((count,), device=self.device)
            self._force_history = torch.zeros(
                (count, self.force_history_obs_steps), device=self.device
            )
            self._force_obs_initialized = torch.zeros(
                (count,), dtype=torch.bool, device=self.device
            )
            self._v4_native_step_index = 0
            self._v4_constraint_ever_violated = torch.zeros(
                (count,), dtype=torch.bool, device=self.device
            )
            self._v4_moving_obstacle_ever_contacted = torch.zeros(
                (count,), dtype=torch.bool, device=self.device
            )
            if self.v4_moving_obstacle is not None:
                moving = self.v4_moving_obstacle_geometry
                self.v4_moving_obstacle.set_pose(
                    sapien.Pose(
                        p=moving.position_xyz_m.astype(np.float32),
                        q=moving.quaternion_wxyz.astype(np.float32),
                    )
                )

    def _before_simulation_step(self):
        tool_x = float(self.v4_tool.pose.p[0, 0].detach().cpu())
        normal = np.asarray(surface_normal(self.v4_scenario, tool_x), dtype=np.float64)
        command = physical_disturbance_command(
            self.v4_scenario,
            native_step_index=self._v4_native_step_index,
            native_hz=100.0,
            outward_normal_world=normal,
        )
        if np.any(command.tool_force_world_n != 0.0):
            self.v4_tool.apply_force(command.tool_force_world_n.astype(np.float32))
        if self.v4_scenario.disturbance_kind == "dynamic_height":
            geometry = self.v4_surface_geometry
            position = np.array(geometry.position_xyz_m, copy=True)
            position[2] += command.surface_height_offset_m
            self.v4_surface_anchor.set_pose(
                sapien.Pose(
                    p=position.astype(np.float32),
                    q=geometry.quaternion_wxyz.astype(np.float32),
                )
            )
        if self.v4_moving_obstacle is not None:
            moving = self.v4_moving_obstacle_geometry
            point, tangent = self.v4_path.at(moving.center_progress)
            normal = np.asarray(
                surface_normal(self.v4_scenario, point[0]), dtype=np.float64
            )
            lateral = np.cross(normal, tangent)
            lateral /= np.linalg.norm(lateral)
            phase = 2.0 * math.pi * (
                (self.v4_scenario.scenario_seed % 991) / 991.0
            )
            amplitude = 0.025 + 0.025 * self.v4_scenario.disturbance_scale
            offset = amplitude * math.sin(
                2.0 * math.pi * 0.45 * self._v4_native_step_index / 100.0 + phase
            )
            position = np.array(moving.position_xyz_m, copy=True) + lateral * offset
            self.v4_moving_obstacle.set_pose(
                sapien.Pose(
                    p=position.astype(np.float32),
                    q=moving.quaternion_wxyz.astype(np.float32),
                )
            )
        self._v4_native_step_index += 1

    def _normal_force(self):
        force_vec = self.scene.get_pairwise_contact_forces(self.v4_tool, self.wipe_pad)
        tool_x = self.v4_tool.pose.p[:, 0].detach().cpu().numpy()
        normals = torch.as_tensor(
            surface_normal(self.v4_scenario, tool_x),
            dtype=force_vec.dtype,
            device=force_vec.device,
        )
        return torch.clamp(torch.sum(force_vec * normals, dim=1), min=0.0)

    def _project_tool(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = self.v4_tool.pose.p.detach().cpu().numpy()
        progress = []
        targets = []
        distances = []
        for position in positions:
            relative = position.copy()
            relative[2] -= self.v4_surface_geometry.top_origin_z_m
            projection = self.v4_path.project(relative)
            target = projection.point_xyz.copy()
            target[2] += self.v4_surface_geometry.top_origin_z_m
            progress.append(projection.progress)
            targets.append(target)
            distances.append(float(np.linalg.norm(position - target)))
        return (
            torch.as_tensor(progress, dtype=torch.float32, device=self.device),
            torch.as_tensor(np.asarray(targets), dtype=torch.float32, device=self.device),
            torch.as_tensor(distances, dtype=torch.float32, device=self.device),
        )

    def evaluate(self):
        normal_force = self._normal_force()
        self._update_force_dynamics_obs(normal_force)
        tcp_progress, target_pos, path_error = self._project_tool()
        force_error = torch.abs(normal_force - self.target_force_n)
        contact_flag = normal_force > 0.2
        wipe_contact = (
            (normal_force >= self.min_wipe_force_n)
            & (normal_force <= self.max_safe_force_n)
        )
        force_band_contact = wipe_contact & (force_error <= self.force_success_band_n)
        instant_wipe_progress = torch.where(
            wipe_contact, tcp_progress, torch.zeros_like(tcp_progress)
        )
        self._best_contact_progress = torch.maximum(
            self._best_contact_progress,
            torch.where(wipe_contact, tcp_progress, self._best_contact_progress),
        ).detach()
        self._best_force_band_progress = torch.maximum(
            self._best_force_band_progress,
            torch.where(
                force_band_contact, tcp_progress, self._best_force_band_progress
            ),
        ).detach()
        constraint_violation = torch.zeros_like(contact_flag)
        if self.v4_scenario.constraint_kind == "keep_out_zone":
            lower = max(
                0.0,
                self.v4_scenario.obstacle_center_s
                - self.v4_scenario.obstacle_half_width_s,
            )
            upper = min(
                1.0,
                self.v4_scenario.obstacle_center_s
                + self.v4_scenario.obstacle_half_width_s,
            )
            constraint_violation = contact_flag & (tcp_progress >= lower) & (tcp_progress <= upper)
        elif self.v4_constraint_actor is not None:
            obstacle_force = self.scene.get_pairwise_contact_forces(
                self.v4_tool, self.v4_constraint_actor
            )
            constraint_violation = torch.linalg.norm(obstacle_force, dim=1) > 0.2
        moving_obstacle_contact = torch.zeros_like(contact_flag)
        if self.v4_moving_obstacle is not None:
            moving_force = self.scene.get_pairwise_contact_forces(
                self.v4_tool, self.v4_moving_obstacle
            )
            moving_obstacle_contact = torch.linalg.norm(moving_force, dim=1) > 0.2
        self._v4_constraint_ever_violated = (
            self._v4_constraint_ever_violated | constraint_violation
        ).detach()
        self._v4_moving_obstacle_ever_contacted = (
            self._v4_moving_obstacle_ever_contacted | moving_obstacle_contact
        ).detach()
        task_success = (
            (self._best_contact_progress > self.success_progress_threshold)
            & force_band_contact
            & (path_error <= self.v4_path_success_tolerance_m)
            & ~self._v4_constraint_ever_violated
            & ~self._v4_moving_obstacle_ever_contacted
        )
        episode_success = (
            torch.zeros_like(task_success)
            if self.v4_lifecycle_mode
            else task_success
        )
        return {
            "success": episode_success,
            "task_success": task_success,
            "fail": normal_force > self.max_safe_force_n,
            "normal_force": normal_force,
            "force_error": force_error,
            "xy_error": path_error,
            "y_error": path_error,
            "wipe_progress": self._best_contact_progress,
            "tcp_progress": tcp_progress,
            "instant_wipe_progress": instant_wipe_progress,
            "force_band_progress": self._best_force_band_progress,
            "instant_force_band_progress": torch.where(
                force_band_contact, tcp_progress, torch.zeros_like(tcp_progress)
            ),
            "force_band_contact": force_band_contact,
            "contact_flag": contact_flag,
            "target_pos": target_pos,
            "pad_pos": self.wipe_pad.pose.p,
            "constraint_violation": constraint_violation,
            "moving_obstacle_contact": moving_obstacle_contact,
            "constraint_ever_violated": self._v4_constraint_ever_violated,
            "moving_obstacle_ever_contacted": self._v4_moving_obstacle_ever_contacted,
            "path_within_success_tolerance": (
                path_error <= self.v4_path_success_tolerance_m
            ),
        }
