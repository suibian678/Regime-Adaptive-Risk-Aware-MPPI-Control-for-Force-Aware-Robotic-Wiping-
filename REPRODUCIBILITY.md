# Reproducibility guide

## Evidence tiers

The repository deliberately separates three levels of reproduction:

1. **Audit reproduction** uses the compact frozen `RESULT.json` files and
   analysis tables included in Git.
2. **Trace reproduction** additionally requires the archived step-level JSONL
   traces.
3. **Policy reproduction** additionally requires the five checkpoint files in
   `MODEL_MANIFEST.csv` and a compatible ManiSkill/SAPIEN or MuJoCo runtime.

This separation keeps Git history reviewable while retaining cryptographic
links to the full research artifacts.

## Environment

The reference environment used Ubuntu 22.04, Python 3.10.12, and the package
versions in `requirements-lock.txt`. PyTorch with CUDA 12.8 should be installed
from the official PyTorch index before installing the remaining packages.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128
python -m pip install -r requirements-lock.txt
```

Enter the main version directory and set the module search path there. Running
from this directory avoids Python treating the outer `forcewipe_v19/`
directory as a namespace package:

```bash
cd forcewipe_v19
export PYTHONPATH="$PWD/code:$PWD/release/runtime_source:$PWD/release/runtime_source/tdmpc2:$PWD/../forcewipe_v16/code:$PWD/../forcewipe_v15/code:$PWD/../forcewipe_v14/code:$PWD/../forcewipe_v6/code:$PWD/../forcewipe_v4/code:$PYTHONPATH"
```

## Fast audit

```bash
pytest -q tests
python ../audit_public_release.py
```

The compact audit verifies the repository manifest, the five external model
identities, the principal frozen result counts and intervals, and the stored
trace-level audit outcomes. The complete manuscript-number and trace
recomputation scripts remain under `scripts/`; they require restoring the raw
artifact archive using `restore_artifact.py` as described in `DATA_AND_MODELS.md`.
Run these in a disposable checkout: the original audit entry points write
their recomputed audit JSON files to the analysis directories. They do not
start a simulator.

```bash
python scripts/audit_manuscript_experiment_numbers.py
python scripts/audit_v19_mujoco_zero_shot_transfer.py
python scripts/audit_deployment_gap_completed.py
```

The manuscript-number check aggregates frozen evaluation records; the latter
two checks additionally recompute their metrics from step traces. These are
different levels of evidence, not three full independent reruns.

## Full evaluation

The fixed protocols under `config/` define scenario keys, method identities,
budgets, and evaluation rules. Relevant entry points are:

- `forcewipe_v19/scripts/run_v19_factor_separated_evaluation_once.py`
- `forcewipe_v19/scripts/run_ppo_selected_confirmation.py`
- `forcewipe_v19/scripts/run_deployment_gap_stress.py`
- `forcewipe_v19/scripts/run_v19_mujoco_zero_shot_transfer_once.py`

The archived one-shot runner IDs are already consumed; do not delete restored
outputs to rerun them. A new experiment needs a separate run ID and output
directory. The release verification commands above do not perform new runs.

These runs are compute-intensive and require the checkpoint archive. The
published results should be verified against `MANIFEST_SHA256.csv` and
`MODEL_MANIFEST.csv` before execution.

## Numerical interpretation

The cross-engine experiment is a zero-shot robustness test after a
policy-free task-port calibration. It does not establish physics-engine
equivalence or hardware transfer. Force-tail statistics are reported only for
the simulator settings in which they were measured.

