"""Download the small PI0.5 tokenizers explicitly; training stays offline."""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from alohamini.paths import WorkspacePaths

from .fast_tokenizer import FAST_FILES, FAST_REPO, FAST_REVISION


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    try:
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=WorkspacePaths().pretrained / "pi05_tokenizers"
    )
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    sp = root / "paligemma_tokenizer.model"
    download("https://storage.googleapis.com/big_vision/paligemma_tokenizer.model", sp)
    for name in FAST_FILES:
        download(
            f"https://huggingface.co/{FAST_REPO}/resolve/{FAST_REVISION}/{name}",
            root / "fast" / name,
        )
    (root / "sources.json").write_text(
        json.dumps(
            {
                "paligemma": "gs://big_vision/paligemma_tokenizer.model",
                "fast": FAST_REPO,
                "revision": FAST_REVISION,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"PaliGemma tokenizer: {sp}")
    print(f"FAST tokenizer: {root / 'fast'}")


if __name__ == "__main__":
    main()
