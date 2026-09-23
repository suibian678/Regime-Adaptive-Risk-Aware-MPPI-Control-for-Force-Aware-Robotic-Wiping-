# Data and model archive

The public Git repository contains compact frozen result objects, audit files,
and figure-ready summaries. Two classes of large artifact are intentionally
kept out of Git history:

- five TD-MPC2 checkpoints (approximately 24 MB each), identified in
  `MODEL_MANIFEST.csv`;
- step-level training, factor-separated, PPO, deployment-gap, numerical, and
  cross-engine traces.

Before journal submission, these files should be deposited in a versioned
research archive such as Zenodo. The release record should include:

1. the archive DOI and version;
2. the repository commit hash;
3. a payload manifest containing path, byte count, and SHA-256 digest;
4. explicit simulator and software versions;
5. a statement that the cross-engine experiment is simulation robustness
   evidence rather than hardware validation.

After deposition, replace the repository and DOI placeholders in both journal
submission packages and add the DOI to `CITATION.cff`.

