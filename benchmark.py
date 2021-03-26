#%%
import subprocess

import numpy as np
import pandas as pd
import json
import pathlib

working_directory = "benchmarks"
benchmark_json = "benchmarks.json"

# Default program arguments
default_benchmark = {
  "csv_file" : "benchmark.csv",
  "max_level" : 4,
  "input_file" : "/home/Christopher.Hagedorn/genData/TCGA-GBM-100-cor.csv",
  "correlation" : True,
  "alpha" : 0.05,
  "observations" : 3190,
  "verbose" : True,
  "OMPThreads" : 80,
  "GPUs" : [0],
  "GPU_only" : False,
  "CPU_only" : False,
  "print_sepsets" : False,
  "workstealing" : False,
  "num_iterations" : 3
}

def read_benchmarks():
  json_path = pathlib.Path(working_directory, benchmark_json)
  with open(json_path) as f:
    data = json.load(f)
    benchmarks = []
    for bench in data["benchmarks"]:
      benchmark = {}
      for key in default_benchmark.keys():
        if not bench.get(key) is None:
          benchmark[key] = bench.get(key)
        else:
          benchmark[key] = default_benchmark[key]
      benchmarks.append(benchmark)
    return benchmarks

def execute_iterations(benchmark):
  csv_path = pathlib.Path(working_directory, benchmark["csv_file"])
  # Building program arguments
  args = ["/home/Milan.Proell/Masterthesis-Heterogeneous-PC/build/src/heterogpc", "-i", benchmark["input_file"], "-a", str(benchmark["alpha"]), "-o", str(benchmark["observations"]), "-m", str(benchmark["max_level"]), "-t" ,str(benchmark["OMPThreads"]), "--csv-export", str(csv_path)]
  if (benchmark["correlation"]):
    args.append("--corr")
  if (benchmark["verbose"]):
    args.append("-v")
  if (benchmark["GPU_only"]):
    args.append("--gpu-only")
  if (benchmark["CPU_only"]):
    args.append("--cpu-only")
  if (benchmark["print_sepsets"]):
    args.append("-p")
  if (benchmark["workstealing"]):
    args.append("-w")
  for gpu in benchmark["GPUs"]:
    args.append("-g")
    args.append(str(gpu))

  # Writing CSV header
  with open(csv_path, "w") as f:
    f.write("num GPUs,num OMP threads,num edges,")
    for i in range(benchmark["max_level"] + 1):
      f.write(f'L{i} duration,L{i} balance duration,L{i} CPU duration,L{i} GPU duration,')
    f.write("execution duration\n")

  # Run iterations
  print("Start benchmarking with the following arguments:")
  print(" ".join(args))
  print()
  logging_path = pathlib.Path("logs", benchmark["csv_file"].replace(".csv", ".log"))
  with open(logging_path, "w") as f:
    for i in range(benchmark["num_iterations"]):
      print(f'Iteration {i} running...')
      subprocess.run(args, stdout=f)
      print(f'Iteration {i} finished\n')


def execute_missing_benchmarks(benchmarks):
  for benchmark in benchmarks:
    csv_path = pathlib.Path(working_directory, benchmark["csv_file"])
    if not csv_path.is_file():
      execute_iterations(benchmark)

def plot_results(file, max_level):
  csv_path = pathlib.Path(working_directory, file)
  print(csv_path)
  results = pd.read_csv(csv_path)

  edges = results.iloc[:,2]
  if (not (edges[0] == edges).all()):
    print("Edges are not equal in every execution. Something went wrong")
    exit(-1)
  print(f'GPUs: {results.iloc[0,0]} OMP Threads: {results.iloc[0,1]}')

  cols_per_level = 4
  levels = max_level + 1
  durations = results.iloc[:,3:(levels * cols_per_level) + 4]
  mean_durations = durations.mean()

  print(f'Mean execution duration: {mean_durations[-1]}')
  plot_frame = pd.DataFrame(index=np.arange(0, levels), columns=["execution", "balancing", "cpu", "gpu", ])

  for i in range(levels):
    dur = mean_durations[i * cols_per_level:(i + 1) * cols_per_level]
    plot_frame.iloc[i] = dur.to_list()
    
  ax = plot_frame.plot(xlabel="Level", ylabel="microseconds", xticks= np.arange(0,levels, 1))
  ax.get_figure().savefig(pathlib.Path("figures", file.replace(".csv", ".svg")))

def plot_benchmark(benchmark):
  plot_results(benchmark["csv_file"], benchmark["max_level"])


# %%
benchmarks = read_benchmarks()
# %%
execute_missing_benchmarks(benchmarks)
# %%
plot_benchmark(benchmarks[0])
# %%
plot_benchmark(benchmarks[1])
# %%
plot_benchmark(benchmarks[2])
# %%
plot_benchmark(benchmarks[3])
# %%
