"""Run-directory sealing: hashes.json (ENGINEERING.md, invariant 2).

At run completion the runner seals the run directory: hashes.json
records the SHA-256 of every run artifact. Presentation files that the
analysis stage owns and regenerates deterministically (SUMMARY.md,
analysis.json, figures/) are excluded from the artifact seal; when the
analysis writes them, it records their hashes under a separate
"derived" section of the same file. hashes.json itself is never
hashed. After sealing, replay verifies every recorded hash before
trusting the contents, and the analysis refuses to read a run whose
artifact seal does not verify.

Layout of hashes.json:
    {"artifacts": {relpath: sha256, ...},     # written once, by seal
     "derived":   {relpath: sha256, ...}}     # maintained by analyze
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SEAL_FILE = "hashes.json"
DERIVED_NAMES = {"SUMMARY.md", "analysis.json"}
DERIVED_DIRS = {"figures"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _is_derived(rel: str) -> bool:
    parts = rel.split("/")
    return parts[-1] in DERIVED_NAMES or parts[0] in DERIVED_DIRS


def _iter_files(run_dir: Path):
    for path in sorted(Path(run_dir).rglob("*")):
        if path.is_file() and path.name != SEAL_FILE:
            yield path.relative_to(run_dir).as_posix(), path


def seal_run_dir(run_dir) -> dict:
    """Write the artifact seal for a finished run directory."""
    run_dir = Path(run_dir)
    artifacts = {}
    derived = {}
    existing = run_dir / SEAL_FILE
    if existing.exists():
        derived = json.loads(existing.read_text()).get("derived", {})
    for rel, path in _iter_files(run_dir):
        if _is_derived(rel):
            continue
        artifacts[rel] = _sha256(path)
    data = {"artifacts": artifacts, "derived": derived}
    existing.write_text(json.dumps(data, indent=1, sort_keys=True))
    return data


def record_derived(run_dir, relpaths) -> dict:
    """Record (or refresh) hashes for analysis-derived files."""
    run_dir = Path(run_dir)
    seal_path = run_dir / SEAL_FILE
    data = {"artifacts": {}, "derived": {}}
    if seal_path.exists():
        data = json.loads(seal_path.read_text())
    for rel in relpaths:
        path = run_dir / rel
        if path.exists():
            data.setdefault("derived", {})[rel] = _sha256(path)
    seal_path.write_text(json.dumps(data, indent=1, sort_keys=True))
    return data


def verify_run_dir(run_dir) -> list:
    """Mismatch strings for every sealed file that no longer matches;
    empty list means the seal verifies (or no seal exists -- legacy
    run directories predate hashes.json and are reported as such by
    the caller, not here)."""
    run_dir = Path(run_dir)
    seal_path = run_dir / SEAL_FILE
    if not seal_path.exists():
        return []
    data = json.loads(seal_path.read_text())
    diffs = []
    for section in ("artifacts", "derived"):
        for rel, want in sorted(data.get(section, {}).items()):
            path = run_dir / rel
            if not path.exists():
                diffs.append(f"sealed file missing ({section}): {rel}")
            elif _sha256(path) != want:
                diffs.append(f"hash mismatch ({section}): {rel} "
                             f"changed after sealing")
    return diffs
