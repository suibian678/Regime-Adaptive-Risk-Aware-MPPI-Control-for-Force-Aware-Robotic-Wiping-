import json
from pathlib import Path

from forcewipe_v19.ppo_budget_sensitivity import (
    BASE_TRANSITIONS,
    PROFILES,
    TRAINING_SEEDS,
    config_for_profile,
    configuration_endpoints,
    selection_key,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_design_has_nine_endpoints_and_five_seeds():
    assert len(PROFILES) == 7
    assert len(configuration_endpoints()) == 9
    assert TRAINING_SEEDS == (301, 302, 303, 304, 305)
    assert len(configuration_endpoints()) * len(TRAINING_SEEDS) == 45


def test_each_sensitivity_profile_changes_exactly_one_ppo_value():
    baseline = config_for_profile("default")
    ignored = {"total_environment_transitions"}
    for profile in PROFILES[1:]:
        candidate = config_for_profile(profile.profile_id)
        differences = {
            name
            for name, value in baseline.__dict__.items()
            if name not in ignored and candidate.__dict__[name] != value
        }
        assert differences == {profile.changed_parameter}
        assert candidate.total_environment_transitions == 2 * BASE_TRANSITIONS


def test_default_budget_is_checkpointed_on_one_continuous_four_x_run():
    profile = PROFILES[0]
    assert profile.checkpoint_transitions == (
        BASE_TRANSITIONS,
        2 * BASE_TRANSITIONS,
        4 * BASE_TRANSITIONS,
    )
    assert config_for_profile("default").total_environment_transitions == 4 * BASE_TRANSITIONS


def test_dev_selection_prioritises_compound_then_task_then_bins_then_return():
    def rows(profile, multiplier, *, compound=0, task=0, bins=0, episode_return=0.0):
        output = []
        for index in range(90):
            is_compound = index < compound
            is_task = index < max(compound, task)
            output.append({
                "profile_id": profile,
                "budget_multiplier": multiplier,
                "task_success": is_task,
                "tracking_pass": is_compound,
                "safety_pass": is_compound,
                "authority_pass": is_compound,
                "completed_dose_bins": bins,
                "episode_return": episode_return,
            })
        return output

    all_rows = rows("default", 1, compound=1, task=1, bins=1, episode_return=1)
    all_rows += rows("default", 2, compound=2, task=2, bins=0, episode_return=0)
    assert selection_key(all_rows, profile_id="default", budget_multiplier=2) > selection_key(
        all_rows, profile_id="default", budget_multiplier=1
    )


def test_code_and_frozen_protocol_define_the_same_profiles_and_endpoints():
    protocol = json.loads((
        ROOT / "config" / "PPO_BUDGET_HYPERPARAMETER_STUDY_PROTOCOL_2026-09-20.json"
    ).read_text(encoding="utf-8"))
    assert [row["id"] for row in protocol["hyperparameter_design"]["profiles"]] == [
        profile.profile_id for profile in PROFILES
    ]
    assert [tuple(row) for row in protocol["configuration_budget_endpoints"]] == list(
        configuration_endpoints()
    )
    assert protocol["development_selection"]["total_evaluations"] == 810
