"""Validate backend dependencies before importing their native libraries."""

import sys
from importlib.metadata import requires, version
from pathlib import Path

from packaging.requirements import Requirement


def check_backend_runtime(adapter):
    from importlib.metadata import PackageNotFoundError

    errors = []
    for text in requires(adapter) or []:
        requirement = Requirement(text)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            actual = version(requirement.name)
        except PackageNotFoundError:
            actual = "missing"
        if actual == "missing" or not requirement.specifier.contains(actual, prereleases=True):
            errors.append(f"{requirement.name}{requirement.specifier} (found {actual})")
    if errors:
        raise ImportError(
            f"{adapter} in {Path(sys.prefix).name} requires: {', '.join(errors)}. "
            "Activate a compatible simulation environment; see docs/install.md."
        )
