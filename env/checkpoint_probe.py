"""Compare fixed offline checkpoint inputs between dependency environments."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path

import numpy as np


def digest(value):
    import torch

    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            result.update(str((item.dtype, tuple(item.shape))).encode())
            result.update(
                item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        elif isinstance(item, dict):
            for key in sorted(item):
                result.update(key.encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            result.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())

    visit(value)
    return result.hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def capture(cases, *, device):
    import torch

    from alohamini.learning.checkpoint import model_directory
    from alohamini.learning.data import AlohaMiniDataset
    from alohamini.learning.data_fields import prepare_fields
    from alohamini.learning.policy import NativePolicy

    if not cases or len({case["name"] for case in cases}) != len(cases):
        raise ValueError("Provide nonempty cases with unique names")
    results = {}
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for case in cases:
        checkpoint = model_directory(case["checkpoint"])
        model = NativePolicy(
            checkpoint, device=device, offline=True, goal_images=case.get("goal_images")
        )
        manifest = model.manifest
        sampling = getattr(model.algorithm, "evaluation_sample_spec", model.algorithm.sample_spec)
        spec = sampling(manifest["config"], cameras=manifest["cameras"])
        spec["exclude_fixed_state"] = manifest.get("fixed_state_excluded", False)
        if manifest.get("data_fields"):
            spec["context_fields"] = ["observation.state"]
        samples = AlohaMiniDataset(
            case["dataset"],
            episodes=case["episodes"],
            **spec,
            state=manifest["state"],
            cameras=manifest["cameras"],
            image_size=tuple(manifest["image_size"]),
            image_resize_mode=manifest.get("image_resize_mode", "stretch"),
            image_pad_value=manifest.get("image_pad_value", 0.0),
            allow_unannotated=case.get(
                "allow_unannotated", bool(manifest["source_info"].get("training_only"))
            ),
        )
        samples = prepare_fields(
            samples,
            {"data": manifest.get("data_fields", {})},
            model.algorithm,
            options=manifest["config"],
        )
        outputs, hashes, durations, peaks = [], [], [], []
        indices = case.get("indices", [0])
        if not indices:
            raise ValueError("Each case needs at least one sample index")
        for index in indices:
            if type(index) is not int or not 0 <= index < len(samples):
                raise ValueError(f"Sample index out of range: {index}")
            seed = case.get("seed", 42)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            sample = samples[index]
            hashes.append(digest(sample))
            # Warm up once. Reset RNG and policy for the measured prediction.
            model.reset()
            model.predict(sample)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model.reset()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            action = model.predict(sample)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
            if device.startswith("cuda"):
                peaks.append(torch.cuda.max_memory_allocated())
            if not action.isfinite().all():
                raise ValueError(f"Non-finite output: {case['name']}")
            outputs.append(action.tolist())
        results[case["name"]] = {
            "case": case,
            "checkpoint_sha256": file_digest(checkpoint / "model.safetensors"),
            "manifest_sha256": file_digest(checkpoint / "policy.json"),
            "input_sha256": hashes,
            "actions": outputs,
            "median_seconds": float(np.median(durations)),
            "peak_cuda_bytes": max(peaks) if peaks else None,
        }
        del model, samples, action, sample
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return {
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "diffusers",
                "accelerate",
                "tokenizers",
            )
        },
        "device": device,
        "cases": results,
    }


def compare(reference, candidate):
    if not reference["cases"] or reference["cases"].keys() != candidate["cases"].keys():
        raise ValueError("Checkpoint comparison cases differ or are empty")
    results = {}
    for name, old in reference["cases"].items():
        new = candidate["cases"][name]
        for key in ("case", "checkpoint_sha256", "manifest_sha256", "input_sha256"):
            if old[key] != new[key]:
                raise ValueError(f"{name}: {key} changed between environments")
        a, b = np.array(old["actions"]), np.array(new["actions"])
        if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"{name}: invalid action output")
        atol, rtol = old["case"].get("atol", 1e-5), old["case"].get("rtol", 1e-4)
        if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
            raise ValueError("Comparison tolerances must be finite and nonnegative")
        results[name] = {
            "passed": bool(np.allclose(a, b, atol=atol, rtol=rtol)),
            "max_abs_difference": float(np.max(np.abs(a - b))),
            "mean_abs_difference": float(np.mean(np.abs(a - b))),
            "atol": atol,
            "rtol": rtol,
            "baseline_seconds": old["median_seconds"],
            "candidate_seconds": new["median_seconds"],
            "baseline_peak_cuda_bytes": old["peak_cuda_bytes"],
            "candidate_peak_cuda_bytes": new["peak_cuda_bytes"],
        }
    return {
        "passed": all(row["passed"] for row in results.values()),
        "cases": results,
        "baseline_versions": reference["versions"],
        "candidate_versions": candidate["versions"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "compare"))
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "capture":
        if args.cases is None:
            parser.error("capture requires --cases")
        result = capture(json.loads(args.cases.read_text()), device=args.device)
    else:
        if args.reference is None or args.candidate is None:
            parser.error("compare requires --reference and --candidate")
        result = compare(
            json.loads(args.reference.read_text()), json.loads(args.candidate.read_text())
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
