# Data and model archive

The public Git repository contains compact frozen result objects, audit files,
and figure-ready summaries. Two classes of large artifact are intentionally
kept out of Git history:

- five TD-MPC2 checkpoints (approximately 24 MB each), identified in
  `MODEL_MANIFEST.csv`;
- step-level factor-separated, PPO confirmation, deployment-gap, and
  cross-engine traces. Training metrics are included in the public repository.

The software and research artifacts are publicly archived on
[Zenodo](https://doi.org/10.5281/zenodo.22929983), version 1.0.0, published on 24 September 2026.
The archived software ZIP corresponds to repository commit
`026276c048eb04429fb93ce8dcd8a7ecfdd0f965`. Later documentation updates in
GitHub do not change that snapshot. Payload manifests and environment records
identify the archived files and software versions.

## Download and reassemble

Download all eight `ForceWipe_RA_RMPPI_full_artifact_v1.zip.part001` through
`.part008` files, plus `assemble_artifact.py`, `ARTIFACT_PARTS_MANIFEST.json`
and `READ_ME_FIRST.txt`, into one directory. The parts form one archive, not
different versions. With Python 3.8 or later, run in that directory:

```bash
python assemble_artifact.py
```

The script verifies every part and reconstructs
`ForceWipe_RA_RMPPI_full_artifact_v1.zip` (380,931,447 bytes). Its SHA-256 is
`90d1e81d7b7c853a4bdaa8da0d97317a8ecd49743911365e3b3f77b928b95734`.
The separate `forcewipe-ra-rmppi-public-repository.zip` contains the software
snapshot and does not need reassembly.

## Verify and restore

From the repository root, run:

```bash
python audit_public_release.py
python restore_artifact.py /path/to/ForceWipe_RA_RMPPI_full_artifact_v1.zip
python restore_artifact.py /path/to/ForceWipe_RA_RMPPI_full_artifact_v1.zip --restore
```

The first archive command checks the complete ZIP identity and every payload
digest without extracting files. The second restores the runner inputs below.
Neither command imports a simulator or starts an experiment. Different
existing data are never overwritten.

| Archive prefix | Restored prefix under the repository root |
| --- | --- |
| `results/factor_separated_final/` | `results/final/v19_factor_separated_m1_m4_evaluation_20260920_r1/` |
| `results/ppo_selected_confirmation/` | `results/final/ppo_budget_sensitivity_selected_confirmation_20260920_r1/` |
| `results/deployment_gap_stress/` | `results/final/v19_deployment_gap_stress_20260920_r1/` |
| `results/crosssim_zero_shot_transfer/` | `results/crosssim/v19_mujoco_zero_shot_transfer_20260921_r1/` |
| `results/crosssim_calibration/` | `results/calibration/mujoco_crosssim_calibration_r3_v1/` |
| `checkpoints/seed_20x/` | `results/train/v19_force_conditioned_seed20x_bc2_u4000/` |
| `summaries/` | `results/dev/` |

The five training-validation calibration files are copied from the public
training metadata into the corresponding checkpoint directories. The archive
contains the original frozen paper snapshot; the current submission versions
are the separate RAS/OJ-CS packages. Do not overwrite the current code or paper
with those historical snapshots.

The archive includes step traces for the four main final studies. Historical
numerical-sensitivity evidence and PPO development selection are supplied as
audited summaries, not as complete raw numerical/DEV trace archives. The
historical controlled-suite `RESULT.json` records are in the public repository.

The original training-demonstration collections are not included. The archive
supports the documented evaluation and analysis workflow, not a complete
from-scratch reconstruction of the training data.

