"""Cached identity of the Python source actually loaded by this process."""

from __future__ import annotations

import hashlib
import os
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path

_EXCLUDED = {".git", "__pycache__", "docs", "tests"}


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _package_version() -> str | None:
    try:
        version = metadata.version("alohamini-platform")
        return version if isinstance(version, str) and version else None
    except (metadata.PackageNotFoundError, OSError, ValueError, UnicodeError):
        return None


def _source_sha256(root: Path) -> str | None:
    def failed(error):
        raise error

    try:
        files = []
        for directory, children, names in os.walk(root, onerror=failed):
            children[:] = [name for name in children if name not in _EXCLUDED]
            # Never report a partial digest if a package contains an external
            # directory link that the traversal would otherwise silently skip.
            if any((Path(directory) / name).is_symlink() for name in children):
                return None
            files.extend(Path(directory) / name for name in names if name.endswith(".py"))
        if not files:
            return None
        digest = hashlib.sha256(b"alohamini-python-source-v1\x00")
        for path in sorted(files, key=lambda value: value.relative_to(root).as_posix()):
            relative = path.relative_to(root).as_posix().encode("utf-8")
            content = path.read_bytes()
            for value in (relative, content):
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
        return digest.hexdigest()
    except (OSError, UnicodeError, ValueError):
        return None


def _git_output(root: Path, *arguments: str) -> str:
    # Inherited Git location overrides must not identify another checkout.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=0.5,
        env=environment,
    )
    return result.stdout.strip()


def _git_revision(root: Path) -> str | None:
    try:
        lines = _git_output(root, "rev-parse", "--show-toplevel", "HEAD").splitlines()
        if len(lines) != 2:
            return None
        repository, revision = Path(lines[0]), lines[1]
        if not repository.is_absolute() or not root.is_relative_to(repository):
            return None
        if len(revision) not in (40, 64) or any(c not in "0123456789abcdef" for c in revision):
            return None
        expected = (root / "__init__.py").relative_to(repository).as_posix()
        # A copied/uninstalled package may sit inside an unrelated parent Git
        # checkout. Its untracked __init__.py must not borrow that repo's HEAD.
        tracked = _git_output(
            root, "ls-files", "--error-unmatch", "--full-name", "--", "__init__.py"
        )
        return revision if tracked == expected else None
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        return None


@lru_cache(maxsize=1)
def software_identity() -> dict:
    """Identify this installed package once; unavailable fields remain None.

    Source identity includes relative Python paths and bytes, independent of
    Git, checkout location and working directory. Callers should copy this
    cached mapping before including it in mutable protocol payloads.
    """
    try:
        root = _package_root()
    except (OSError, RuntimeError):
        root = None
    return {
        "schema_version": 1,
        "package_version": _package_version(),
        "source_sha256": _source_sha256(root) if root is not None else None,
        "git_revision": _git_revision(root) if root is not None else None,
    }
