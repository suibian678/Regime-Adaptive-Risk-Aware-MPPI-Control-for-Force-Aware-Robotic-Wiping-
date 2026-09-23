# Public-release portability patches

The experiment runners were developed in a fixed local workspace. The public
repository preserves all algorithms, protocols, parameters, and result
objects, but applies the following path-only substitutions while building the
release tree:

1. hard-coded TD-MPC2 workspace fallbacks are replaced with
   `forcewipe_v19/release/runtime_source` relative to the public repository;
2. the earlier training helper resolves `tdmpc2/config.yaml` from that frozen
   runtime snapshot;
3. obsolete local sandbox output roots in unused legacy modules are redirected
   from a user home directory to `/tmp`.

No policy weights, scenario definitions, planner settings, metric rules, or
reported result objects are changed by these substitutions. The repository
manifest is generated after the substitutions so that the public files have
their own auditable identities.

