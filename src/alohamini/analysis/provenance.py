"""Checkpoint and data lineage, without importing a model or touching GPU state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def describe_checkpoint(path, *, hash_weights=False):
    from alohamini.learning.checkpoint import read_checkpoint

    path, manifest = read_checkpoint(path)
    training = dict(manifest.get("training", {}))
    train_config = path / "train_config.json"
    if train_config.is_file():
        saved = json.loads(train_config.read_text())
        for key in ("dataset", "train_episodes", "val_episodes"):
            if key not in training and key in saved:
                training[key] = saved[key]
    return {
        "path": str(path),
        "manifest_sha256": digest(path / "policy.json"),
        "weights_sha256": digest(path / "model.safetensors") if hash_weights else None,
        "kind": manifest["kind"],
        "config": manifest["config"],
        "cameras": manifest["cameras"],
        "image_size": manifest["image_size"],
        "image_resize_mode": manifest.get("image_resize_mode", "stretch"),
        "image_pad_value": manifest.get("image_pad_value", 0.0),
        "state": manifest["state"],
        "training": training,
        "dataset_files": manifest.get("table_sha256", {}),
        "action_feature": manifest["source_info"]["features"]["action"],
    }


def display_config(config):
    if not isinstance(config, dict):
        return None
    return {
        key: config[key]
        for key in (
            "chunk_size",
            "n_action_steps",
            "temporal_ensemble_coeff",
            "use_vae",
            "posterior_condition",
            "prior_type",
            "latent_inference_mode",
            "latent_refresh_mode",
        )
        if key in config
    }


def capture_policy(policy):
    path = getattr(policy, "checkpoint_path", None)
    if not isinstance(path, (str, Path)):
        return {"schema_version": 1, "checkpoint": None, "status": "custom_policy"}
    return {
        "schema_version": 1,
        "checkpoint": describe_checkpoint(path, hash_weights=True),
        "status": "recorded",
    }


def write_evaluation(root, policy, captured, *, fps, task, num_episodes):
    """Called once before control starts, after fixed-current/calibration binding."""
    config = getattr(policy, "config", None)
    runtime = (
        asdict(config) if is_dataclass(config) else config if isinstance(config, dict) else None
    )

    def value(name):
        item = getattr(policy, name, None)
        return item if isinstance(item, (str, int, float, list, dict, bool)) else None

    document = {
        **captured,
        "effective_policy_config": runtime,
        "deployment": {
            name: value(name)
            for name in ("fixed_dimensions", "fixed_current", "calibration_mode", "rtc")
        },
        "fps": fps,
        "task": task,
        "num_episodes": num_episodes,
    }
    path = Path(root) / "meta/evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # A newly created evaluation dataset owns this file; never rewrite earlier runs.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def resolve_context(root, *, checkpoint=None, training_root=None, reference_split="train"):
    """An explicit checkpoint relocates/verifies a recorded identity, not replaces it."""
    path = Path(root) / "meta/evaluation.json"
    record = json.loads(path.read_text()) if path.is_file() else {}
    if record and record.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation metadata version")
    stored = record.get("checkpoint")
    description = stored
    source = "recorded" if stored else "unavailable"
    warnings = []
    if checkpoint:
        description = describe_checkpoint(checkpoint, hash_weights=bool(stored))
        if stored:
            for key in ("manifest_sha256", "weights_sha256"):
                if stored.get(key) and stored[key] != description.get(key):
                    raise ValueError(f"所选 checkpoint 与本次评估记录不符：{key}")
        else:
            source = "user_supplied_for_legacy_recording"
            warnings.append("旧评估未保存模型关联；checkpoint 来自本次显式指定，运行时参数仍未知。")
    elif stored:
        candidate = Path(stored["path"])
        if not (candidate / "policy.json").is_file():
            warnings.append(
                "原 checkpoint 路径不可用；示范对比使用已保存的配置，模型探针需 --checkpoint 定位。"
            )
    else:
        warnings.append("旧评估未记录 checkpoint；使用 --checkpoint 指定当时使用的模型以关联示范。")
    training = description.get("training", {}) if description else {}
    original_root = training.get("dataset")
    chosen = (
        Path(training_root or original_root).expanduser().resolve()
        if (training_root or original_root)
        else None
    )
    if training_root and not description:
        raise ValueError("先关联 checkpoint，再重定位它的训练数据集")
    available = chosen is not None and (chosen / "meta/info.json").is_file()
    if training_root and not available:
        raise ValueError(f"--training-root 不是可用数据集：{chosen}")
    fingerprints = (description or {}).get("dataset_files", {})
    if reference_split == "val":
        fingerprints = training.get("validation_table_sha256", fingerprints)
    verified = []
    if available:
        # Check metadata now, selected numeric files when loading the reference. No video scan.
        for relative, expected in fingerprints.items():
            if not relative.startswith("meta/") or not relative.endswith((".json", ".parquet")):
                continue
            file = (chosen / relative).resolve()
            if not file.is_relative_to(chosen):
                raise ValueError("Checkpoint dataset reference escapes its root")
            if not file.is_file() or digest(file) != expected:
                raise ValueError(f"训练数据与 checkpoint 指纹不符：{relative}")
            verified.append(relative)
    elif original_root:
        warnings.append(
            f"checkpoint 对应数据不在本机：{original_root}；"
            "可用 --training-root 指向该数据集的本地副本。"
        )
    return {
        "checkpoint": description,
        "checkpoint_source": source,
        "effective_policy_config": record.get("effective_policy_config"),
        "deployment": record.get("deployment"),
        "training_root_recorded": original_root,
        "training_root": str(chosen) if chosen else None,
        "training_available": available,
        "verified_metadata": verified,
        "fingerprint_scope": "metadata_and_selected_numeric_files; media_not_hashed",
        "reference_split": reference_split,
        "warnings": warnings,
    }


def reference_episode(context, requested=None):
    checkpoint = context["checkpoint"]
    if not checkpoint:
        return None
    training = checkpoint["training"]
    key = "val_episodes" if context["reference_split"] == "val" else "train_episodes"
    episodes = training.get(key)
    if not episodes:
        if requested is not None:
            raise ValueError(f"checkpoint 没有 {key} 划分，不能确认该示范属于所选集合")
        context["warnings"].append(f"checkpoint 没有可用的 {key}；无法自动选择示范。")
        return None
    if requested is not None and requested not in episodes:
        raise ValueError(f"Episode {requested} 不在 checkpoint 的 {key} 中")
    return requested if requested is not None else episodes[0]


def verify_reference(context, data):
    checkpoint = context["checkpoint"]
    fingerprints = checkpoint.get("dataset_files", {})
    if context["reference_split"] == "val":
        fingerprints = checkpoint["training"].get("validation_table_sha256", fingerprints)
    checked = []
    for path in data.signatures:
        relative = str(path.relative_to(data.root))
        if relative in fingerprints:
            if digest(path) != fingerprints[relative]:
                raise ValueError(f"示范数据与 checkpoint 指纹不符：{relative}")
            checked.append(relative)
    if data.info["features"]["action"] != checkpoint["action_feature"]:
        raise ValueError("示范动作字段与 checkpoint 不一致")
    context["verified_reference_files"] = checked
    if not fingerprints:
        context["warnings"].append("checkpoint 没有数据指纹；示范来源身份未验证。")
