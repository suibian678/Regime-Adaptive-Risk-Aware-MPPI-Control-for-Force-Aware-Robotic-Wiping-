# ForceWipe RA-RMPPI

Force-conditioned TD-MPC2 with **regime-adaptive risk-aware model predictive path
integral control (RA-RMPPI)** for direct robotic wiping in simulation.

Companion code for *Regime-Adaptive Risk-Aware MPPI Control for Force-Aware
Robotic Wiping with Temporal-Difference World Models*.

**[Paper](paper/ras/manuscript_ras.pdf)** ·
**[Reproduction guide](docs/REPRODUCIBILITY.md)** ·
**[Development trace](docs/DEVELOPMENT_TRACE.md)** ·
**[Data and models](docs/DATA_AND_MODELS.md)**

## What this repository contains

One current implementation, organised by function. Start with
[`risk_aware_mppi.py`](src/forcewipe/risk_aware_mppi.py) for the planner,
[`risk_conditioning.py`](src/forcewipe/risk_conditioning.py) for regime-dependent
costs and sampling budgets, and
[`force_conditioned_world_model.py`](src/forcewipe/force_conditioned_world_model.py)
for the world-model extension.

| Directory | Contents |
| --- | --- |
| `src/forcewipe/` | Current method, simulation interfaces, training utilities and comparison methods |
| `scripts/` | Training, evaluation, analysis and figure-generation entry points |
| `tests/` | Non-physics unit and interface tests |
| `config/` | Experiment definitions and frozen protocols |
| `results/` | Compact result records, training metrics and analyses |
| `paper/` | The same paper in RAS and OJ-CS submission formats |
| `archive/` | Earlier controlled-study and numerical-audit evidence, not alternative software versions |
| `vendor/tdmpc2/` | The attributed TD-MPC2 runtime snapshot |
| `docs/` | Reproduction instructions and the main development milestones |

Historical identifiers inside result records, protocols and checkpoint names
are retained to keep the evidence traceable. They are not versions to choose
between. The changes that led to this implementation are summarised in the
[development trace](docs/DEVELOPMENT_TRACE.md).

## Main results

- In 270 paired evaluations, compound pass increased from **85/135 to 100/135**
  relative to fixed-budget MPPI using the same checkpoints. The improvement was
  **11.11 percentage points**, with a block- and seed-aware 95% interval of
  **[3.70, 18.52] percentage points**.
- Median planning time decreased by **25.7 ms**.
- Five frozen checkpoints transferred to MuJoCo without retraining. All
  **30 flat and inclined evaluations** passed the compound criterion. The
  **15 weakly curved evaluations** exposed the remaining transfer boundary.

These are simulation results. Cross-engine evaluation tests robustness, not
hardware deployment.

## Quick start

The compact evidence check needs only Python's standard library:

```bash
python audit_public_release.py
```

For the unit tests, use the environment in
[the reproduction guide](docs/REPRODUCIBILITY.md), then run from this repository:

```bash
python -m pytest -q
```

The checkpoint and raw-trace archive is separate from Git. Its identities and
restoration commands are documented in [Data and models](docs/DATA_AND_MODELS.md).

## Licence and citation

ForceWipe extensions use the [MIT licence](LICENSE). The TD-MPC2 runtime is
based on Nicklas Hansen's implementation at commit
`8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe`.
See [third-party notices](docs/THIRD_PARTY_NOTICES.md) and [citation metadata](CITATION.cff).
An archival DOI will be added after the data and model deposit.

