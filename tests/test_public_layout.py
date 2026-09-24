"""Regression checks for the single-package public repository."""
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_no_development_version_directories_or_imports():
    assert not [p for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"forcewipe_v\d+", p.name)]
    for folder in ("src", "scripts", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                assert not any(re.match(r"forcewipe_v\d+(?:\.|$)", name) for name in names), path


def test_all_internal_import_modules_and_symbols_exist():
    failures = []
    for folder in ("src", "scripts", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("forcewipe."):
                    continue
                target = ROOT / "src" / (node.module.replace(".", "/") + ".py")
                if not target.exists():
                    target = target.with_suffix("") / "__init__.py"
                if not target.is_file():
                    failures.append(f"{path.name}: missing {node.module}")
                    continue
                tree = ast.parse(target.read_text())
                bound = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                bound.update(n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
                bound.update(a.asname or a.name for n in ast.walk(tree) if isinstance(n, (ast.ImportFrom, ast.Import)) for a in n.names)
                for alias in node.names:
                    if alias.name not in bound:
                        failures.append(f"{path.name}: missing {node.module}.{alias.name}")
    assert not failures, failures


def test_quickstart_and_trace_links_exist():
    for relative in ("README.md", "docs/REPRODUCIBILITY.md", "docs/DEVELOPMENT_TRACE.md"):
        page = ROOT / relative
        for target in re.findall(r"\]\(([^)]+)\)", page.read_text()):
            if "://" not in target and not target.startswith("#"):
                assert (page.parent / target.split("#")[0]).exists(), (relative, target)


def test_asset_and_training_locations():
    from forcewipe.paths import REPOSITORY_ROOT
    assert REPOSITORY_ROOT == ROOT
    assert (REPOSITORY_ROOT / "vendor/tdmpc2/config.yaml").is_file()


def test_archive_restore_uses_current_root():
    import importlib.util
    spec = importlib.util.spec_from_file_location("restore", ROOT / "restore_artifact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.destination("checkpoints/seed_201/model.pt").is_relative_to(ROOT / "results/train")
    assert module.destination("results/crosssim_zero_shot_transfer/RESULT.json").is_relative_to(ROOT / "results/crosssim")
