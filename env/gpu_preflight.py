"""Refuse dependency validation on missing, duplicate or occupied GPUs."""

import os
import re
import subprocess


def validate_devices(selected, inventory, processes):
    if not re.fullmatch(r"\d+(,\d+)+", selected):
        raise ValueError("Specify at least two physical GPU indices, for example 0,1")
    indices = selected.split(",")
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate GPU indices")
    devices = dict(line.strip().split(", ", 1) for line in inventory.splitlines() if line.strip())
    if any(index not in devices for index in indices):
        raise ValueError("A requested GPU is unavailable")
    busy = {line.strip().split(",", 1)[0] for line in processes.splitlines() if line.strip()}
    if any(devices[index] in busy for index in indices):
        raise RuntimeError("A requested GPU has a running compute process")


def main():
    def query(option):
        return subprocess.check_output(["nvidia-smi", option, "--format=csv,noheader"], text=True)

    validate_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        query("--query-gpu=index,uuid"),
        query("--query-compute-apps=gpu_uuid,pid"),
    )
    print("PASS: requested GPUs exist and have no compute processes")


if __name__ == "__main__":
    main()
