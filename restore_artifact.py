#!/usr/bin/env python3
"""Verify the frozen archive; optionally restore data/model inputs (no simulation)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path, PurePosixPath
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE_SHA256 = "90d1e81d7b7c853a4bdaa8da0d97317a8ecd49743911365e3b3f77b928b95734"
RUNS = {
    "factor_separated_final": "final/v19_factor_separated_m1_m4_evaluation_20260920_r1",
    "ppo_selected_confirmation": "final/ppo_budget_sensitivity_selected_confirmation_20260920_r1",
    "deployment_gap_stress": "final/v19_deployment_gap_stress_20260920_r1",
    "crosssim_zero_shot_transfer": "crosssim/v19_mujoco_zero_shot_transfer_20260921_r1",
    "crosssim_calibration": "calibration/mujoco_crosssim_calibration_r3_v1",
}


def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def destination(name):
    parts = PurePosixPath(name).parts
    if not parts or ".." in parts or name.startswith("/") or "\\" in name:
        raise ValueError(f"Invalid archive path: {name}")
    if len(parts) >= 3 and parts[0] == "results" and parts[1] in RUNS:
        return ROOT / "results" / RUNS[parts[1]] / Path(*parts[2:])
    if len(parts) == 3 and parts[0] == "checkpoints":
        seed = int(parts[1].removeprefix("seed_"))
        if seed not in range(201, 206):
            raise ValueError(f"Unexpected checkpoint seed: {seed}")
        return ROOT / f"results/train/v19_force_conditioned_seed{seed}_bc2_u4000" / parts[2]
    if len(parts) == 2 and parts[0] == "summaries":
        return ROOT / "results/dev" / parts[1]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--restore", action="store_true", help="Also restore runner inputs; existing different files are never overwritten")
    args = parser.parse_args()
    with args.archive.open("rb") as stream:
        if digest(stream) != ARCHIVE_SHA256:
            raise ValueError("Archive SHA-256 differs from the published release")
    with zipfile.ZipFile(args.archive) as z:
        rows = list(csv.DictReader(io.StringIO(z.read("MANIFEST_SHA256.csv").decode("utf-8-sig"))))
        names = z.namelist()
        expected = {r["path"] for r in rows} | {"README.md", "MANIFEST_SHA256.csv"}
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("Archive entries do not match the manifest")
        planned = []
        for row in rows:
            name = row["path"]
            if z.getinfo(name).file_size != int(row["bytes"]):
                raise ValueError(f"Size mismatch: {name}")
            with z.open(name) as stream:
                if digest(stream) != row["sha256"]:
                    raise ValueError(f"Hash mismatch: {name}")
            target = destination(name)
            if target is not None:
                target.resolve().relative_to(ROOT.resolve())
                if args.restore and target.exists():
                    with target.open("rb") as stream:
                        if digest(stream) != row["sha256"]:
                            raise ValueError(f"Refusing to overwrite different data: {target}")
                planned.append((name, target))
        # Public calibration metadata complement the byte-frozen checkpoint ZIP.
        calibrations = []
        for seed in range(201, 206):
            source = ROOT / f"results/training/seed_{seed}/TRAIN_VALIDATION_CALIBRATION.json"
            target = ROOT / f"results/train/v19_force_conditioned_seed{seed}_bc2_u4000/TRAIN_VALIDATION_CALIBRATION.json"
            target.resolve().relative_to(ROOT.resolve())
            if not source.is_file():
                raise FileNotFoundError(source)
            if args.restore and target.exists() and source.read_bytes() != target.read_bytes():
                raise ValueError(f"Refusing to overwrite calibration: {target}")
            calibrations.append((source, target))
        if args.restore:
            for name, target in planned:
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as source, target.open("xb") as out:
                    shutil.copyfileobj(source, out)
            for source, target in calibrations:
                if not target.exists():
                    with source.open("rb") as src, target.open("xb") as out:
                        shutil.copyfileobj(src, out)
        print(f"PASS: archive identity and {len(rows)} payload hashes; "
              f"{len(planned) + len(calibrations)} runner inputs "
              f"{'restored or already identical' if args.restore else 'mapped (verification only)'}")


if __name__ == "__main__":
    main()
