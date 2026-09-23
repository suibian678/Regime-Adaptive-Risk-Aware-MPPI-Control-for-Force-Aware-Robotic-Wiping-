from dataclasses import asdict

from forcewipe_v19.factor_separated_scenarios import (
    BLOCKS,
    REFERENCE,
    TARGETS_N,
    changed_fields,
    factor_separated_scenario,
    scenario_digest,
)


def test_roster_has_one_reference_and_four_two_level_factors():
    assert len(BLOCKS) == 9
    assert [row["factor"] for row in BLOCKS].count("reference") == 1
    for factor in ("surface", "path", "friction", "support_stiffness"):
        assert [row["factor"] for row in BLOCKS].count(factor) == 2


def test_each_shift_changes_only_its_declared_fields():
    reference = asdict(factor_separated_scenario("B0", 8.0))
    ignored_identity = {"scenario_id"}
    for block in BLOCKS[1:]:
        candidate = asdict(factor_separated_scenario(block["block_id"], 8.0))
        observed = {
            key for key in reference
            if key not in ignored_identity and reference[key] != candidate[key]
        }
        assert observed == set(changed_fields(block["block_id"]))


def test_all_block_target_scenarios_are_valid_and_unique():
    specs = [
        factor_separated_scenario(block["block_id"], target)
        for block in BLOCKS for target in TARGETS_N
    ]
    assert len({spec.scenario_id for spec in specs}) == 27
    assert len({scenario_digest(spec) for spec in specs}) == 27
    assert all(spec.residual_seed == REFERENCE["residual_seed"] for spec in specs)
