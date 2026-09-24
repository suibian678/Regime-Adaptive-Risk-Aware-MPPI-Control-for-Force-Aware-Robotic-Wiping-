# Development trace

This is the short history of the main problems and changes that led to the
current ForceWipe method. Version labels below identify research milestones,
not separate supported releases. Infrastructure-only retries are grouped with
their parent milestone.

| Research milestone | Main problem or finding | Change and outcome |
| --- | --- | --- |
| Early hierarchical studies, V4–V5.7 | Force tracking, contact recovery and task progress could fail independently. Repeated controller adjustments did not establish reliable high-force wiping. | Added causal logging and explicit motion authority, then separated structural correctness from closed-loop performance. These studies motivated a change of approach, rather than a final controller release. |
| Unified and direct-control development, V6–V15 | Separate force and recovery logic complicated state transitions. A direct learned policy also needed to acquire and maintain contact. | Investigated a unified supervisor, then developed direct Cartesian TD-MPC2 control, behavioural training support and actor-centred planning. The current evaluation policy does not use the earlier classical force/recovery supervisor. |
| Direct TD-MPC2, V16 | Controlled-task feasibility did not imply generalisation to weak curvature. Completion and force regulation were distinct outcomes. | Recovery-aware behavioural cloning and force prediction supported direct wiping. Qualification completed 75/75 tasks and CAL completed 150/150, with strict tracking passing 75/75 and 148/150 respectively. Matched out-of-distribution comparisons exposed a remaining boundary and did not resolve a general MPPI advantage over actor-only control. |
| Numerical and provenance audit, V17 | Force-tail claims required testing against solver/contact choices. Some claimed disturbances were not bound to the learner-visible path. | Audited parameter binding and numerical sensitivity. The study found configuration-sensitive force tails, so the paper limited those statistics to their measured settings and corrected the affected disturbance claims. Untested time discretisation was not treated as converged. |
| Current method, V19 | A fixed planning objective and sampling budget did not adapt to the contact regime. | Added explicit target-force conditioning and regime-adaptive risk-aware MPPI. In the same-checkpoint paired comparison, compound pass increased from 85/135 to 100/135 and median planning time decreased by 25.7 ms. |
| Baseline and deployment checks | A short PPO budget and single-engine evaluation left important comparisons unresolved. | Added PPO budget/hyperparameter evaluation, deployment-gap tests and policy-free MuJoCo port calibration. Five checkpoints transferred without retraining: 30/30 flat/inclined cases passed the compound criterion, while 15 weakly curved cases exposed the transfer boundary. |
| Public repository consolidation | Development-labelled folders made the current implementation difficult to identify. | Consolidated runtime code under `src/forcewipe`, separated scripts, evidence and paper files, and removed unused development code from the current tree. This was a packaging change, not a new experiment or method revision. |

## Evidence in this repository

- [Controlled direct-policy results](../archive/direct_study/results/)
- [Numerical-audit record](../archive/numerical_audit/)
- [Current paired method comparison](../results/factor_separated/RESULT.json)
- [Selected PPO confirmation](../results/ppo_confirmation/RESULT.json)
- [Deployment-gap evaluation](../results/deployment_gap/RESULT.json)
- [MuJoCo zero-shot evaluation](../results/crosssim/RESULT.json)

The early milestones explain the research decisions. The linked frozen
evaluation records support the reported results. Native sampling frequency is
not a claim of real-time planning, and cross-engine transfer is simulation
evidence rather than a hardware experiment.
