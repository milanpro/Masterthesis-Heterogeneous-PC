#%%
import subprocess

import numpy as np
import pandas as pd

# Program arguments
csv_file = "benchmark.csv"
max_level = 4
input_file = "/home/Christopher.Hagedorn/genData/TCGA-GBM-100-cor.csv"
correlation = True
alpha = 0.05
observations = 3190
verbose = True
OMPThreads = 80
GPUs = [0]
GPU_only = False
CPU_only = False
print_sepsets = False

# Iteration options
num_iterations = 3

def execute_iterations():
  # Building program arguments
  args = ["/home/Milan.Proell/Masterthesis-Heterogeneous-PC/build/src/heterogpc", "-i", input_file, "-a", str(alpha), "-o", str(observations), "-m", str(max_level), "-t" ,str(OMPThreads), "--csv-export", csv_file]
  if (correlation):
    args.append("--corr")
  if (verbose):
    args.append("-v")
  if (GPU_only):
    args.append("--gpu-only")
  if (CPU_only):
    args.append("--cpu-only")
  if (print_sepsets):
    args.append("-p")
  for gpu in GPUs:
    args.append("-g")
    args.append(str(gpu))

  # Writing CSV header
  with open(csv_file, "w") as f:
    f.write("num GPUs,num OMP threads,num edges,")
    for i in range(max_level + 1):
      f.write(f'L{i} duration,L{i} balance duration,L{i} CPU duration,L{i} GPU duration,')
    f.write("execution duration\n")

  # Run iterations
  print("Start benchmarking with the following arguments:")
  print(" ".join(args))
  print()
  with open("benchmark.log", "w") as f:
    for i in range(num_iterations):
      print(f'Iteration {i} running...')
      subprocess.run(args, stdout=f)
      print(f'Iteration {i} finished\n')


def plot_results():
  results = pd.read_csv(csv_file)

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
    
  plot_frame.plot(xlabel="Level", ylabel="microseconds", xticks= np.arange(0,levels, 1))

# %%
if __name__ == "__main__":
  execute_iterations()
  plot_results()
