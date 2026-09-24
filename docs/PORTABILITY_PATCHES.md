# Public-source layout

The public source is organised as one `forcewipe` package. The initial public
commit retained development-version directories. This cleanup replaces those
import namespaces with functional packages, relocates the runtime to `vendor/`,
and gives current scripts version-independent names.

The shared evaluation metrics and PPO collector were extracted from earlier
scripts without changing their function bodies. Modules outside the current
source/test/runner dependency graph were removed from the working tree. Their
original files remain recoverable from Git commit
`e022b79aa801eec4391c9952aa8b0328a763564a`.

Experiment configurations, result records, checkpoint identities, manuscript
files and the vendored runtime retain their original bytes. Historical format
strings, scenario IDs and run names remain inside these records. The public
manifest is regenerated for the new layout and is separate from original-run
manifests.

Earlier path-only release changes redirected local workspace fallbacks to the
bundled runtime and obsolete sandbox outputs to `/tmp`. The current training
loader uses `FORCEWIPE_TRAINING_DATA_ROOT` for separately supplied collections.
No policy weights, force thresholds, planner parameters, metric definitions or
reported outcomes are changed by the directory cleanup.
