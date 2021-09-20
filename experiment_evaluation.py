# Helper script to create diagrams used in the thesis

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
#%%
benchmarks_delos_old = read_benchmarks(delos_benchmark_files / "benchmarks_thesis" /benchmark_json)
#%%
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
#%%
attach_data(delos_benchmark_files, benchmarks_delos, "delos")
#%%
attach_data(delos_benchmark_files / "benchmarks_thesis", benchmarks_delos_old, "delos")
#%%
attach_data(delos_noatomics_benchmark_files, benchmarks_delos_noatomics, "delos", "noatomics")
attach_data(delos_nomigrations_benchmark_files, benchmarks_delos_nomigrations, "delos", "nomigrations")
#%%
attach_data(ac922_benchmark_files, benchmarks_ac922, "ac922")
attach_data(ac922_nosmt_benchmark_files, benchmarks_ac922_nosmt, "ac922", "nosmt")
#%%
benchmarks = []
benchmarks.extend(benchmarks_delos)
benchmarks.extend(benchmarks_delos_old)
#%%
benchmarks.extend(benchmarks_delos_noatomics)
benchmarks.extend(benchmarks_delos_nomigrations)
#%%
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
pre_balanced = [bench("pre_balanced", "delos"), bench("pre_balanced_numa0", "delos"), bench("pre_balanced", "delos", "noatomics"), bench("pre_balanced", "delos", "nomigrations")]

compare(pre_balanced)
# %%
pre_balanced = [bench("gpu_only", "ac922"), bench("pre_balanced", "ac922"), bench("pre_balanced_numa0", "ac922"), bench("workstealing", "ac922"), bench("workstealing_numa0", "ac922")]

compare(pre_balanced)
# %%
comp = [bench("gpu_only", "delos"), bench("pre_balanced", "delos"), bench("workstealing_78_threads", "delos"), bench("workstealing_1_thread_numa0", "ac922", "nosmt")]

plt = compare(comp)

# %%
comp = [bench("gpu_only", "delos"), bench("workstealing", "delos"), bench("workstealing_numa0", "delos"), bench("workstealing", "delos", "noatomics"), bench("workstealing", "delos", "nomigrations")]

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
comp = []
objects = []
for i in range(80):
  comp.append(bench(f'workstealing_{i + 1}_threads', "delos"))
  objects.append(str(i+1))
for i in comp:
  data = search_bench(i["name"], i["system"], i["attribute"])
  i["duration"] = data["data"][-1]

y_pos = np.arange(1, len(comp) + 1)
performance = [i["duration"] / 1000 for i in comp]
plt.bar(y_pos, performance, align='center', alpha=0.5)
axes = plt.gca()
axes.set_ylabel("seconds")
axes.set_xlabel("CPU thread count")
axes.yaxis.grid(True)
axes.set_title("Multithreaded workstealing execution time - Delos")
plt.xticks(np.arange(0, len(comp) + 1, 5.0))
plt.axvline(x=12.5, color="red")
plt.axvline(x=78.5, color="green")
#plt.xlim(1,80)
fig = axes.get_figure()
fig.set_size_inches(10, 5, forward=True)
plt.show()
fig.savefig("./threaded_wsteal.pdf", bbox_inches = 'tight')

plt.close()
# %%
comp = [bench("gpu_only", "delos"), bench("workstealing_1_thread", "delos"), bench("workstealing_78_threads", "delos"), bench("workstealing_45000", "delos"), bench("gpu_only_45000", "delos"), bench("workstealing_1_threads_45000", "delos"), bench("workstealing_78_threads_45000", "delos"), bench("workstealing_10000", "delos"), bench("gpu_only_10000", "delos"), bench("workstealing_1_thread_10000", "delos"), bench("workstealing_78_threads_10000", "delos"), bench("workstealing_numa0_10000", "delos")]

compare(comp)
# %%
# %%
comp = [bench("gpu_only", "delos"), bench("pre_balanced", "delos"), bench("workstealing_78_threads", "delos")]
for i in comp:
  data = search_bench(i["name"], i["system"], i["attribute"])
  i["L0"] = data["data"][3]
  i["L1"] = data["data"][7]
  i["L2"] = data["data"][11]
  i["L3"] = data["data"][15]
  i["L4"] = data["data"][19]
  i["arr"] = [i["L1"], i["L2"], i["L3"], i["L4"]]

labels = ["1", "2", "3", "4"]
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
gpu_data = comp[0]["arr"]
pre_data = comp[1]["arr"]
work_data = comp[2]["arr"]

gpu = ax.bar(x - width, np.divide(gpu_data,gpu_data), width, label='GPU-only')
pre = ax.bar(x, np.divide(gpu_data,pre_data), width, label='Pre-balanced')
work = ax.bar(x + width, np.divide(gpu_data,work_data), width, label='Workstealing')

ax.set_ylabel('speedup factor')
ax.set_title('Speedup factor compared to GPU-only')
ax.set_xlabel('level')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.yaxis.grid(True)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)

# compare(comp)
fig.tight_layout()
fig.savefig("./levelwise.pdf", bbox_inches = 'tight')
plt.show()

# %%
numa_nosmt = [bench("workstealing_1_thread_numa0", "ac922", "nosmt"), bench("workstealing_2_thread_numa0", "ac922", "nosmt"), bench("workstealing_3_thread_numa0", "ac922", "nosmt"), bench("workstealing_4_thread_numa0", "ac922", "nosmt"), bench("workstealing_5_thread_numa0", "ac922", "nosmt")]
numa = [bench("workstealing_1_thread_numa0", "ac922"), bench("workstealing_2_thread_numa0", "ac922"), bench("workstealing_3_thread_numa0", "ac922"), bench("workstealing_4_thread_numa0", "ac922"), bench("workstealing_5_thread_numa0", "ac922")]

for i in numa_nosmt:
  data = search_bench(i["name"], i["system"], i["attribute"])
  i["duration"] = data["data"][-1]
for i in numa:
  data = search_bench(i["name"], i["system"], i["attribute"])
  i["duration"] = data["data"][-1]

labels = ["1", "2", "3", "4", "5"]
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
numa_nosmt = [i["duration"] for i in numa_nosmt]
numa = [i["duration"] for i in numa]

numa_nosmt_bar = ax.bar(x - width/2, numa_nosmt, width, label='NUMA pinned, No SMT')
numa_bar = ax.bar(x + width/2, numa, width, label='NUMA pinned, SMT-4')

ax.set_ylabel('milliseconds')
ax.set_title('Workstealing approach execution time with different thread counts')
ax.set_xlabel('CPU thread count')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.yaxis.grid(True)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)

# compare(comp)
fig.tight_layout()
fig.savefig("./ac922_threadcount.pdf", bbox_inches = 'tight')
plt.show()
# %%
comp = [bench("gpu_only", "ac922"), bench("workstealing_1_thread_numa0", "ac922")]
for i in comp:
  data = search_bench(i["name"], i["system"], i["attribute"])
  i["L0"] = data["data"][3]
  i["L1"] = data["data"][7]
  i["L2"] = data["data"][11]
  i["L3"] = data["data"][15]
  i["L4"] = data["data"][19]
  i["arr"] = [i["L1"], i["L2"], i["L3"], i["L4"]]

labels = ["1", "2", "3", "4"]
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
gpu_data = comp[0]["arr"]
work_data = comp[1]["arr"]

# gpu = ax.bar(x - width/2, np.divide(gpu_data,gpu_data), width, label='GPU-only')
work = ax.bar(x, np.divide(gpu_data,work_data), width, label='Workstealing, NUMA pinned, 1 Thread')

ax.set_ylabel('speedup factor')
ax.set_title('NUMA pinned workstealing speedup factor compared to GPU-only')
ax.set_xlabel('level')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.yaxis.grid(True)
ax.axhline(1.0, color = "grey")
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          # fancybox=True, shadow=True, ncol=5)

# compare(comp)
fig.tight_layout()
fig.savefig("./ac922_levelwise.pdf", bbox_inches = 'tight')
plt.show()
# %%
comp = [bench("gpu_only_10000", "delos"), bench("workstealing_78_threads_10000", "delos")]
for i in comp:
  data = search_bench(i["name"], i["system"], i["attribute"])
  i["L0"] = data["data"][3]
  i["L1"] = data["data"][7]
  i["L2"] = data["data"][11]
  i["L3"] = data["data"][15]
  i["L4"] = data["data"][19]
  i["arr"] = [i["L1"], i["L2"], i["L3"], i["L4"]]

labels = ["1", "2", "3", "4"]
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
gpu_data = comp[0]["arr"]
work_data = comp[1]["arr"]

# gpu = ax.bar(x - width/2, np.divide(gpu_data,gpu_data), width, label='GPU-only')
work = ax.bar(x, np.divide(gpu_data,work_data), width, label='Workstealing')

ax.set_ylabel('speedup factor')
ax.set_title('Workstealing speedup factor compared to GPU-only with 10 000 variables')
ax.set_xlabel('level')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.yaxis.grid(True)
ax.axhline(1.0, color = "grey")
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#           fancybox=True, shadow=True, ncol=5)

# compare(comp)
fig.tight_layout()
fig.savefig("./levelwise_scaled.pdf", bbox_inches = 'tight')
plt.show()
# %%
