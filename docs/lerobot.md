# LeRobot Compatibility

**English** | [简体中文](zh-CN/lerobot.md)

The platform uses LeRobot v3 datasets. ACT, AM-ACT, Diffusion, FastWAM, SmolVLA, and π0.5
share a data reader for training, validation, and offline inference; each algorithm retains its own state, language, and action representation requirements.

For external v3 datasets without platform metadata, add `--dataset.allow_unannotated=true`.
`--state=recorded` uses the original state vector; `--state=none` uses vision only.
Videos, task text, and numeric fields are paired by their original episode. Missing calibration is not filled in automatically.
Hardware inference must satisfy the target robot's action-field, unit, and calibration requirements.

See [Training and deployment](training.md) for dataset usage. To select fields or save a separate dataset,
use the [export command](alohamini.md#export-and-field-selection).

Dataset format and model-weight format are separate: platform-trained checkpoints work with `alohamini evaluate`;
external LeRobot checkpoints load only through the base formats supported by each algorithm. Sharing the v3 dataset format does not make weights interchangeable.
