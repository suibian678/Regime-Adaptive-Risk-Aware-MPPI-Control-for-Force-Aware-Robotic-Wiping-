# Reproduction guide

Run all commands from the repository root. There is one `forcewipe` package.
Earlier version-labelled directories have been consolidated by function.

## Environment

The reference environment used Ubuntu 22.04 and Python 3.10.12. Key dependency
versions are pinned in [`requirements-lock.txt`](../requirements-lock.txt).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128
python -m pip install -r requirements-lock.txt
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/vendor:$PWD/vendor/tdmpc2:$PYTHONPATH"
```

The test configuration sets these import paths automatically for pytest.
The main execution scripts also resolve their source paths from the checkout.

## Recompute the compact evidence

```bash
python audit_public_release.py
python -m pytest -q
```

The first command checks file identities, checkpoint metadata and headline
aggregates from the included evaluation records. The tests cover control
interfaces, scenario definitions, calibration and model configuration without
running a physics experiment.

## Recompute from step traces

Restore the separate archive as described in [Data and models](DATA_AND_MODELS.md),
then run the following in a disposable checkout:

```bash
python scripts/audit_manuscript_experiment_numbers.py
python scripts/audit_mujoco_zero_shot_transfer.py
python scripts/audit_deployment_gap_completed.py
```

These scripts write recomputed audit files. The first checks manuscript numbers
against frozen records. The other two additionally recompute metrics from
step traces. None starts a simulator.

## Code entry points

| Purpose | Entry point |
| --- | --- |
| Force-conditioned training | `scripts/train_force_conditioned_seed.py` |
| Checkpoint calibration | `scripts/calibrate_trained_seed.py` |
| Paired fixed/adaptive MPPI evaluation | `scripts/run_factor_separated_evaluation_once.py` |
| PPO training-budget sensitivity | `scripts/train_ppo_budget_sensitivity.py` |
| Selected PPO evaluation | `scripts/run_ppo_selected_confirmation.py` |
| Deployment-gap evaluation | `scripts/run_deployment_gap_stress.py` |
| MuJoCo zero-shot evaluation | `scripts/run_mujoco_zero_shot_transfer_once.py` |

Evaluation requires the archived checkpoints and the relevant simulator.
Training additionally needs the original demonstration collections, which are
not included in the checkpoint/trace ZIP. Set `FORCEWIPE_TRAINING_DATA_ROOT` to
their parent directory, retaining their archived collection paths.

Archived runners retain consumed run IDs and source-identity checks from the
original experiments. They document those runs, not permission to overwrite
them. New experiments need their own output IDs and protocol identities for
the current source layout. The public manifest identifies the redistributed
code, while frozen result objects retain the identities of the code used to
produce them.

## Source organisation

`src/forcewipe/` contains the main planner and world model. Its subpackages
separate simulation geometry (`simulation`), the direct-action interface
(`direct`), inherited training and comparison components (`learning`), batch
training utilities (`training_data`) and planning primitives (`planning`).

Earlier evidence is under `archive/`, not on the Python import path.
The [development trace](DEVELOPMENT_TRACE.md) explains the main changes and
[portability notes](PORTABILITY_PATCHES.md) describe the source relocation.
