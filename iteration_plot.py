# %%
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

# %%
working_dir = pathlib.Path("iteration_count_data/TCGA-GBM-1000")
# %%
working_dir = pathlib.Path("iteration_count_data/MCC")
# %%
files = ["1test_iteration_map.csv", "2test_iteration_map.csv", "3test_iteration_map.csv", "4test_iteration_map.csv"]
files = [working_dir / pathlib.Path(i) for i in files]
# %% Iterations BoxPlot
result = pd.DataFrame()
for i, file in enumerate(files):
  if file.is_file():
    df = pd.read_csv(file, names=["row_length", "test_count", "iterations"])
    result[i + 1] = df["iterations"]
ax = result.plot.box(showmeans=True, ylabel = "iterations", xlabel = "level")
ax.get_figure().savefig(working_dir / "iterations.pdf")
# %% Test Count BoxPlot
result = pd.DataFrame()
for i, file in enumerate(files):
  if file.is_file():
    df = pd.read_csv(file, names=["row_length", "test_count", "iterations"])
    result[i + 1] = df["test_count"]
ax = result.plot.box(showmeans=True, ylabel = "test count", xlabel = "level")
ax.get_figure().savefig(working_dir / "test_count.pdf")
# %% Row Length BoxPlot
result = pd.DataFrame()
for i, file in enumerate(files):
  if file.is_file():
    df = pd.read_csv(file, names=["row_length", "test_count", "iterations"])
    result[i + 1] = df["row_length"]
ax = result.plot.box(showmeans=True, ylabel = "row length", xlabel = "level")
ax.get_figure().savefig(working_dir / "row_length.pdf")
# %% Row Length BoxPlot
row_length = pd.DataFrame()
iteration = pd.DataFrame()

for i, file in enumerate(files):
  if file.is_file():
    df = pd.read_csv(file, names=["row_length", "test_count", "iterations"])
    row_length[i + 1] = df["row_length"]
    iteration[i + 1] = df["iterations"]

row_length = row_length.fillna(0)
iteration = iteration.fillna(0)

fig, ax = plt.subplots()

bp = ax.boxplot(row_length, showmeans=True)
ax.set_ylabel("row length")
ax.set_xlabel("level")
ax.set_title("All row lengths of the adjencency rows per level")
plt.legend([bp['medians'][0], bp['means'][0], bp['fliers'][0]], ['median', 'mean', 'outliers'])
fig.savefig(working_dir / "rowl.pdf")

fig, ax = plt.subplots()
bp = ax.boxplot(iteration, showmeans=True)
ax.set_ylabel("iterations")
ax.set_xlabel("level")
ax.set_title("All max iterations of a row per level")
plt.legend([bp['medians'][0], bp['means'][0], bp['fliers'][0]], ['median', 'mean', 'outliers'])
fig.savefig(working_dir / "iter.pdf")
# %%
