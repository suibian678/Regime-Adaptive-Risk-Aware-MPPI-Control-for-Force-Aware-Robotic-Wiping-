# ForceWipe RA-RMPPI

This repository accompanies **Regime-Adaptive Risk-Aware MPPI Control for
Force-Aware Robotic Wiping with Temporal-Difference World Models**. It contains
the force-conditioned TD-MPC2 extensions, regime-adaptive risk-aware MPPI
(RA-RMPPI), simulation task definitions, evaluation protocols, compact frozen
results, analysis scripts, and manuscript sources.

## Main evidence

- In 270 paired evaluations, RA-RMPPI increased compound pass from 85/135 to
  100/135 relative to fixed-budget MPPI.
- The paired improvement was 11.11 percentage points with a block- and
  seed-aware 95% interval of [3.70, 18.52] percentage points.
- RA-RMPPI used eight fewer planning samples and reduced median planning time
  by 25.7 ms.
- Five frozen checkpoints transferred without retraining to the MuJoCo task
  port: all 30 flat and inclined evaluations passed, whereas all 15 weakly
  curved evaluations exposed the remaining portability boundary.

These are simulation results. Cross-engine transfer is a robustness test, not
a substitute for hardware validation.

## Repository layout

| Path | Contents |
| --- | --- |
| `forcewipe_v19/code/` | RA-RMPPI, force conditioning, calibration, and task ports |
| `forcewipe_v19/release/runtime_source/` | Frozen TD-MPC2 runtime snapshot |
| `forcewipe_v4/`--`forcewipe_v16/` | Earlier modules required by frozen runners |
| `forcewipe_v19/config/` | Evaluation and cross-engine protocols |
| `forcewipe_v19/scripts/` | Training, evaluation, analysis, and audit entry points |
| `forcewipe_v19/tests/` | Non-physics unit and contract tests |
| `forcewipe_v19/results/` | Compact frozen results and independent audits |
| `forcewipe_v19/paper/` | RAS and OJ-CS packages, tables, and vector figures |
| `MODEL_MANIFEST.csv` | Checkpoint identities for the external model archive |
| `MANIFEST_SHA256.csv` | Byte counts and SHA-256 identities for repository files |

Raw step traces and model checkpoints are kept out of Git history. Their
archive structure and integrity checks are described in `DATA_AND_MODELS.md`.

## Environment

The reported environment used Python 3.10.12, PyTorch 2.11.0 with CUDA 12.8,
ManiSkill 3.0.1, SAPIEN 3.0.3, and MuJoCo 3.1.2. Exact key versions are listed
in `requirements-lock.txt`.

Run the public commands from the `forcewipe_v19` directory. This avoids the
top-level version directory being interpreted as a Python namespace package
and keeps the frozen sibling-version imports unambiguous:

```bash
cd forcewipe_v19
export PYTHONPATH="$PWD/code:$PWD/release/runtime_source:$PWD/release/runtime_source/tdmpc2:$PWD/../forcewipe_v16/code:$PWD/../forcewipe_v15/code:$PWD/../forcewipe_v14/code:$PWD/../forcewipe_v6/code:$PWD/../forcewipe_v4/code:$PYTHONPATH"
pytest -q tests
python ../audit_public_release.py
```

The public tree preserves the original sibling-version layout expected by the
frozen runners. A small path-only portability patch is documented in
`PORTABILITY_PATCHES.md`. The full training and evaluation workflow, including expected inputs and
which commands require the separately archived checkpoints, is documented in
`REPRODUCIBILITY.md`.

## Licence and attribution

Original ForceWipe extensions are released under the MIT licence. The bundled
TD-MPC2 runtime derives from Nicklas Hansen's MIT-licensed TD-MPC2 repository
at commit `8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe`. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`.

## Citation

Citation metadata are provided in `CITATION.cff`. A DOI will be added after
the archival release is deposited.

