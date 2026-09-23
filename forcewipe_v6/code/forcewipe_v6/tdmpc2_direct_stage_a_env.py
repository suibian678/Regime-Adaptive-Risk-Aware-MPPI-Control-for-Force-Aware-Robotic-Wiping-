"""Versioned direct-control environment for Stage-A physical scenario replay.

The historical direct environment intentionally binds one nominal physical
scenario.  This subclass keeps the same observation, action, reward, and
termination contracts while accepting one explicit ``ScenarioSpec``.  It is a
development data adapter; it does not add a force controller or action filter.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Mapping

import gymnasium as gym

from .tdmpc2_direct_firstpass import SafeContactDoseTracker
from .tdmpc2_direct_sapien_env import (
    V6DirectFirstPassEnv,
    _scalar,
    _vector,
)


class V6StageADirectFirstPassEnv(V6DirectFirstPassEnv):
    """Direct first-pass environment bound to an explicit physical scenario."""

    def __init__(self, *, scenario_spec: object, config=None) -> None:
        from forcewipe_v4.scenarios import ScenarioSpec

        if isinstance(scenario_spec, ScenarioSpec):
            template = scenario_spec
        elif isinstance(scenario_spec, Mapping):
            template = ScenarioSpec(**dict(scenario_spec))
        else:
            raise TypeError("scenario_spec must be a ScenarioSpec or mapping")
        template.validate()
        self._stage_a_template = template
        super().__init__(
            target_force_n=float(template.target_force_n),
            scenario_seed=int(template.scenario_seed),
            scenario_id=int(template.scenario_id),
            config=config,
        )

    def resolved_scenario(self, *, scenario_seed: int, scenario_id: int):
        """Return the explicit template with only execution identity replaced."""

        spec = replace(
            self._stage_a_template,
            scenario_seed=int(scenario_seed),
            scenario_id=int(scenario_id),
            target_force_n=float(self.target_force_n),
            residual_seed=int(scenario_seed) + 1_000_000,
        )
        spec.validate()
        return spec

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Import for the Gymnasium registration side effect.  It is deliberately
        # local so pure contract tests do not require the SAPIEN/force_press
        # runtime, while every physical reset registers the environment itself.
        import forcewipe_v4.sapien_v4_env  # noqa: F401
        from forcewipe_v4.scenarios import make_scenario_path

        super()._close_physics()
        if options:
            if "target_force_n" in options:
                target = float(options["target_force_n"])
                if target not in {5.0, 8.0, 12.0}:
                    raise ValueError("invalid reset target")
                self.target_force_n = target
            if "scenario_seed" in options:
                self.scenario_seed = int(options["scenario_seed"])
            if "scenario_id" in options:
                self.scenario_id = int(options["scenario_id"])
        if seed is not None:
            self.scenario_seed = int(seed)
        self._spec = self.resolved_scenario(
            scenario_seed=self.scenario_seed,
            scenario_id=self.scenario_id,
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
        self._previous_action.fill(0.0)
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
            "physical_scenario": asdict(self._spec),
            "direct_tdmpc2_action_authority": True,
            "force_dependent_action_projection": False,
            "rewiping_enabled": False,
        }
