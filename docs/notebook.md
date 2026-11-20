# Notebooks

**English** | [简体中文](zh-CN/notebook.md)

Use notebooks to inspect data and images, plot time series, analyze velocity/current, assemble state/action vectors, inspect single-batch loss, and compare predicted trajectories.
See [Training and deployment](training.md) for production data preparation, training, offline evaluation, and hardware deployment; notebooks are optional.

Complete [PC installation and kernel registration](install.md#3-optional-extensions).

Open the notebook with the VS Code Jupyter extension or an existing JupyterLab installation:
`examples/learning/local_act.ipynb`. Select the **AlohaMini** kernel.
The PC environment includes kernel dependencies, but not the JupyterLab server.

Set the dataset path and episode splits after opening. `RUN_TRAINING` starts training; `ENABLE_ROBOT` enables hardware evaluation. Leave both disabled for data analysis only.
