"""Explicit Qwen configuration/tokenizer/image-processor asset acquisition."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download Spirit's Qwen assets without backbone weights"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--revision",
        default="ebb281ec70b05090aa6165b016eac8ec08e71b17",
        help="Qwen/Qwen3-VL-4B-Instruct commit SHA",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("--revision must be a full Qwen commit SHA")
    from huggingface_hub import snapshot_download

    root = args.output.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new output directory for pinned Qwen assets")
    snapshot_download(
        "Qwen/Qwen3-VL-4B-Instruct",
        revision=args.revision,
        local_dir=root,
        allow_patterns=["*.json", "*.jinja", "*.txt", "*.tiktoken"],
    )
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(root).parts and p.name != "sources.json"
    }
    provenance = dict(repository="Qwen/Qwen3-VL-4B-Instruct", revision=args.revision, sha256=files)
    (root / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
