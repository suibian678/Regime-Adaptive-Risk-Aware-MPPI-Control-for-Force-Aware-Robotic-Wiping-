# Data and model archive

The public Git repository contains compact frozen result objects, audit files,
and figure-ready summaries. Two classes of large artifact are intentionally
kept out of Git history:

- five TD-MPC2 checkpoints (approximately 24 MB each), identified in
  `MODEL_MANIFEST.csv`;
- step-level factor-separated, PPO confirmation, deployment-gap, and
  cross-engine traces. Training metrics are included in the public repository.

Before journal submission, these files should be deposited in a versioned
research archive such as Zenodo. The release record should include:

1. the archive DOI and version;
2. the repository commit hash;
3. a payload manifest containing path, byte count, and SHA-256 digest;
4. explicit simulator and software versions;
5. a statement that the cross-engine experiment is simulation robustness
   evidence rather than hardware validation.

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

| Archive prefix | Restored prefix under `forcewipe_v19/` |
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

After deposition, replace the repository and DOI placeholders in both journal
submission packages and add the DOI to `CITATION.cff`.

