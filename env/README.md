# Dependency Maintenance

`pyproject.toml` declares project dependencies. `dependencies.toml` selects PC extras, deployment versions, and update groups; scripts generate `.in` and pip `.lock` files. Conda Python, FFmpeg, and base-tool versions remain in `environment.yml`, `host.yml`, and their explicit files.

Run in the `alohamini` Conda environment, first installing the `uv` version specified in `dependencies.toml`:

```bash
python env/dependencies.py check
python env/dependencies.py refresh
python env/dependencies.py refresh --group models --report /tmp/dependency-changes.md
```

`refresh` modifies dependency files only. Existing locked versions are preferred. With a group specified, it checks stable releases at least 7 days old within the same major version and resolves again. Resolution failure restores original files. `models`, `compute`, `host`, and `tools` manage model, compute, hardware, and remaining tool/data dependencies, respectively. PC/Host shared dependencies keep identical versions.

A selected group updates only that group and its transitive dependencies, keeping other groups locked, without automatic downgrades or major upgrades. For coordinated group migrations, explicitly change project declarations or deployment constraints first, then run `refresh` without `--group` and run regression checks.

## GitHub Actions

- `Dependency validation`: runs on relevant PRs, main commits, and manual triggers. It validates manifests, installs CPU equivalents in a new Conda environment, and runs algorithm, checkpoint, resume, precision, and distributed-validation regressions. Separate x86_64/ARM64 runners install the Host lock and check serial libraries, images, and in-process communication. CPU lockfiles are CI artifacts only and do not replace CUDA deployment locks.
- `Dependency update candidates`: runs at 03:17 UTC on the first day of each quarter (January, April, July, October), or manually for a selected group. It updates only that group, creates a draft PR, and triggers CPU/Host checks for the candidate commit. Successful generation does not imply compatibility; check links are in the run summary, and test summaries list failures. PRs record version differences; artifacts are retained for 30 days.
- Dependabot: checks GitHub Actions versions quarterly, combining updates in the same group into one PR.
- `Dependency GPU validation`: manually select a candidate branch, idle GPU IDs, and a checkpoint case file. Requires a Linux x86_64 self-hosted runner labeled `alohamini-gpu`, Conda, a compatible driver, and at least two idle GPUs. Checkpoints, base assets, and datasets in the case file must be readable.

For models exceeding one GPU's memory, set `checkpoint_backend` to `fsdp2` and compare inference on the first two selected GPUs, preserving checkpoint weight precision.

Repository and organization Actions settings must permit PR creation. Candidates remain drafts without automatic merging. Model/compute updates require both CPU and GPU checks before acceptance. GPU workflows use separate temporary environments without updating existing runtime environments; they exit if selected GPUs have compute processes.

## Compare existing checkpoints

Copy `checkpoint-cases.example.json` into an output directory and fill in real checkpoints, datasets, and representative samples. `indices` are sample indices in the dataset formed by selected episodes. Use checkpoint action units and task-appropriate `atol`/`rtol`. Visual-goal policies also require `goal_images`.

Capture separately in original and candidate environments, then compare:

```bash
python env/checkpoint_probe.py capture --cases /path/to/cases.json --device cuda:0 --output /tmp/baseline.json
python env/checkpoint_probe.py capture --cases /path/to/cases.json --device cuda:0 --output /tmp/candidate.json
python env/checkpoint_probe.py compare --reference /tmp/baseline.json --candidate /tmp/candidate.json --output /tmp/comparison.json
```

Comparison requires identical checkpoints, configurations, inputs, and random seeds. It records action differences, inference time, GPU memory, and dependency versions. Outputs outside tolerance fail. Performance records support review; short-sample timing is not an automatic acceptance threshold.

When numeric changes are acceptable after compute-library updates, set `"acceptance": {"metric": "mae", "rtol": 0.02, "atol": [...]}` per case. Specify `atol` in action-dimension order, with absolute error budgets in each dimension's units. Each new MAE must not exceed `old MAE × (1 + rtol) + atol`. Use preselected validation samples and thresholds, exclude trailing padding and historical actions, and retain output differences. This measures offline action error and does not replace closed-loop task evaluation.
