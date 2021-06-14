#%%
import numpy as np
import pandas as pd
import pathlib
import os
import json
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

delos_benchmark_files = pathlib.Path("benchmarks_delos")
delos_noatomics_benchmark_files = delos_benchmark_files / "benchmarks_11.3_gcc_noatomics"
delos_nomigrations_benchmark_files = delos_benchmark_files / "benchmarks_11.3_gcc_nomigrations"
ac922_benchmark_files = pathlib.Path("benchmarks_ac922")
ac922_nosmt_benchmark_files = ac922_benchmark_files / "smt_disabled"
benchmark_json = "benchmarks.json"

def read_benchmarks(file):
  with open(file) as f:
    data = json.load(f)
    return data["benchmarks"]

#%%
benchmarks_delos = read_benchmarks(delos_benchmark_files / benchmark_json)
benchmarks_delos_noatomics = read_benchmarks(delos_noatomics_benchmark_files / benchmark_json)
benchmarks_delos_nomigrations = read_benchmarks(delos_nomigrations_benchmark_files / benchmark_json)

#%%
benchmarks_ac922 = read_benchmarks(ac922_benchmark_files / benchmark_json)
benchmarks_ac922_nosmt = read_benchmarks(ac922_nosmt_benchmark_files / benchmark_json)
# %%
def get_csv(bench):
  return bench["csv_file"]

def read_csv(dir, bench):
  file_name = dir / get_csv(bench)
  if not file_name.is_file():
    print("File not found " + file_name)
  df =  pd.read_csv(file_name)
  if len(df.index) == 0:
    print("No data in csv " + file_name)
  return df.mean()

def attach_data(dir, benchmarks, system, attribute = None):
  for bench in benchmarks:
    df = read_csv(dir, bench)
    bench["data"] = df
    bench["name"] = bench["csv_file"].removeprefix("benchmark_").removesuffix(".csv")
    bench["system"] = system
    bench["attribute"] = attribute

attach_data(delos_benchmark_files, benchmarks_delos, "delos")
attach_data(delos_noatomics_benchmark_files, benchmarks_delos_noatomics, "delos", "noatomics")
attach_data(delos_nomigrations_benchmark_files, benchmarks_delos_nomigrations, "delos", "nomigrations")

attach_data(ac922_benchmark_files, benchmarks_ac922, "ac922")
attach_data(ac922_nosmt_benchmark_files, benchmarks_ac922_nosmt, "ac922", "nosmt")

benchmarks = []
benchmarks.extend(benchmarks_delos)
benchmarks.extend(benchmarks_delos_noatomics)
benchmarks.extend(benchmarks_delos_nomigrations)

benchmarks.extend(benchmarks_ac922)
benchmarks.extend(benchmarks_ac922_nosmt)
# %%
def search_bench(name, system, attribute = None):
  for bench in benchmarks:
    if bench["name"] == name and bench["system"] == system and bench["attribute"] == attribute:
      return bench
  print("Benchmark not found: " +name+system+str(attribute))

def compare(bench_list):
  for bench in bench_list:
    data = search_bench(bench["name"], bench["system"], bench["attribute"])
    bench["duration"] = data["data"][-1]
  
  y_pos = np.arange(len(bench_list))
  objects = [f'{i["name"]} {i["system"]} {i["attribute"] if i["attribute"] != None else ""}' for i in bench_list]
  performance = [i["duration"] for i in bench_list]
  plt.barh(y_pos, performance, align='center', alpha=0.5)
  plt.yticks(y_pos, objects)
  plt.show()
  return plt

def bench(name, system, attribute = None):
  return {
    "name": name,
    "system": system,
    "attribute": attribute
  }
# %%
pre_balanced = [bench("pre_balanced", "delos"), bench("pre_balanced", "delos", "noatomics"), bench("pre_balanced", "delos", "nomigrations")]

compare(pre_balanced)
# %%
comp = [bench("gpu_only", "delos"), bench("pre_balanced", "delos"), bench("workstealing_78_threads", "delos"), bench("workstealing_1_thread_numa0", "ac922", "nosmt")]

plt = compare(comp)

# %%
comp = [bench("workstealing_1_thread_numa0", "ac922", "nosmt"), bench("workstealing_2_thread_numa0", "ac922", "nosmt"), bench("workstealing_3_thread_numa0", "ac922", "nosmt"), bench("workstealing_4_thread_numa0", "ac922", "nosmt"), bench("workstealing_5_thread_numa0", "ac922", "nosmt"), bench("workstealing_8_threads_numa0", "ac922", "nosmt"), bench("workstealing_numa0", "ac922", "nosmt")]

compare(comp)
# %%
comp = [bench("workstealing_1_thread_numa0", "ac922"), bench("workstealing_2_thread_numa0", "ac922"), bench("workstealing_3_thread_numa0", "ac922"), bench("workstealing_4_thread_numa0", "ac922"), bench("workstealing_5_thread_numa0", "ac922"), bench("workstealing_30_threads_numa0", "ac922"), bench("workstealing_numa0", "ac922")]

compare(comp)

# %%
comp = [bench("workstealing_1_thread_numa0", "ac922"), bench("workstealing_2_thread_numa0", "ac922"), bench("workstealing_3_thread_numa0", "ac922"), bench("workstealing_4_thread_numa0", "ac922"), bench("workstealing_5_thread_numa0", "ac922")]

compare(comp)
# %%
comp = [bench("workstealing_numa0", "ac922"), bench("workstealing_numa0", "ac922", "nosmt")]

compare(comp)
# %%
comp = [bench("workstealing_1_thread", "delos"), bench("workstealing_2_threads", "delos"), bench("workstealing_3_threads", "delos"), bench("workstealing_4_threads", "delos"), bench("workstealing_5_threads", "delos"), bench("workstealing_20_threads", "delos"), bench("workstealing_40_threads", "delos"), bench("workstealing_60_threads", "delos"), bench("workstealing_78_threads", "delos"), bench("workstealing", "delos")]

compare(comp)
# %%
comp = [bench("workstealing_10000", "delos"), bench("gpu_only_10000", "delos"), bench("workstealing_1_thread_10000", "delos"), bench("workstealing_78_threads_10000", "delos"), bench("workstealing_numa0_10000", "delos")]

compare(comp)
# %%
comp = [bench("workstealing_45000", "delos"), bench("gpu_only_45000", "delos"), bench("workstealing_1_threads_45000", "delos"), bench("workstealing_78_threads_45000", "delos")]

compare(comp)
# %%
