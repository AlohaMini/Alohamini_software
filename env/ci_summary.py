"""Render pytest JUnit results in the GitHub Actions job summary."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def summary(path):
    if not path.is_file():
        return "No test report was produced; inspect the failed test step.\n"
    cases = list(ET.parse(path).getroot().iter("testcase"))
    failed = [c for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skipped = sum(c.find("skipped") is not None for c in cases)
    passed = len(cases) - len(failed) - skipped
    lines = [f"CPU compatibility: **{passed} passed, {len(failed)} failed, {skipped} skipped**."]
    if failed:
        lines += ["", "Failures:", ""]
        for case in failed:
            label = f"{case.get('classname', '')}::{case.get('name', '')}".replace("`", "'")
            lines.append(f"- `{label}`")
    lines += ["", "GPU and existing-checkpoint validation are reported separately."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(summary(Path(sys.argv[1])), end="")
