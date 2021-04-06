# %%
import pandas as pd
import pathlib

# %%
working_dir = pathlib.Path("iteration_count_data/TCGA-GBM-1000")
# %%
files = ["1test_iteration_map.csv", "2test_iteration_map.csv", "3test_iteration_map.csv", "4test_iteration_map.csv"]
files = [working_dir / pathlib.Path(i) for i in files]
# %% Iterations BoxPlot
result = pd.DataFrame()
for i, file in enumerate(files):
  if file.is_file():
    df = pd.read_csv(file, names=["row_length", "test_count", "iterations"])
    result[i + 1] = df["iterations"]
ax = result.plot.box(showmeans=True, ylabel = "iterations")
# %% Test Count BoxPlot
result = pd.DataFrame()
for i, file in enumerate(files):
  if file.is_file():
    df = pd.read_csv(file, names=["row_length", "test_count", "iterations"])
    result[i + 1] = df["test_count"]
ax = result.plot.box(showmeans=True, ylabel = "test count")
# %% Test Count BoxPlot
result = pd.DataFrame()
for i, file in enumerate(files):
  if file.is_file():
    df = pd.read_csv(file, names=["row_length", "test_count", "iterations"])
    result[i + 1] = df["row_length"]
ax = result.plot.box(showmeans=True, ylabel = "row length")
# %%
