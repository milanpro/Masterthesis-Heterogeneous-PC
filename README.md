### Masterthesis-Heterogeneous-PC

https://github.com/ChristopherSchmidt89/EPIC-causality

https://github.com/philipp-bode/lock-free-pc

https://github.com/LIS-Laboratory/cupc

https://github.com/cran/pcalg

https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/atomic.html

#### Docker
Build the base image using:

x86_64: `docker build -f docker/Dockerfile.base -t milanpro/cmake-armadillo-python38-centos8 .`

ppc64le: `docker build -f docker/Dockerfile.base.ppc64le -t milanpro/cmake-armadillo-python38-centos8-ppc64le .`

The executable image is built using:

x86_64: `docker build -f docker/Dockerfile -t milanpro/heterogpc .`

ppc64le: `docker build -f docker/Dockerfile.ppc64le -t milanpro/heterogpc-ppc64le .`

#### Delos NUMA Topology

```
Milan.Proell@delos:~$ nvidia-smi topo -m
	GPU0	GPU1	GPU2	GPU3	CPU Affinity	NUMA Affinity
GPU0	 X 	NV2	NV2	SYS	0-19,40-59	0
GPU1	NV2	 X 	SYS	NV1	0-19,40-59	0
GPU2	NV2	SYS	 X 	NV2	20-39,60-79	1
GPU3	SYS	NV1	NV2	 X 	20-39,60-79	1

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

```
Milan.Proell@delos:~$ numactl -H
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59
node 0 size: 772696 MB
node 0 free: 427008 MB
node 1 cpus: 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
node 1 size: 774135 MB
node 1 free: 422708 MB
node distances:
node   0   1
  0:  10  21
  1:  21  10
```

#### TCGA Datasets
Can be found in: `/home/Christopher.Hagedorn/genData`

`/home/Milan.Proell/Masterthesis-Heterogeneous-PC/build/src/heterogpc --corr -i "/home/Christopher.Hagedorn/genData/TCGA-GBM-100-cor.csv" -o 3190 -v -t 80`
#### Other Paper Datasets
In `/home/Christopher.Hagedorn/PC-Alg-Data`:

```
1.1M	./coolingData.csv
2.7M	./alarm.csv
216K	./sachs.csv
100K	./earthquake.csv
496K	./NCI-60.csv
13M	./DREAM5-Insilico.csv
128M	./munin.csv
708K	./BR51.csv
59M	./link20k.csv
3.0M	./Saureus.csv
33M	./andes.csv
1.5M	./MCC.csv
5.9M	./Scerevisiae.csv
35M	./arth150.csv
```

#### Debugging
`/usr/local/cuda/bin/cuda-gdb /home/Milan.Proell/Masterthesis-Heterogeneous-PC/build/src/heterogpc`

#### Bugs
https://github.com/xianyi/OpenBLAS/wiki/Faq#multi-threaded
Prevent threading segfault : `export OPENBLAS_NUM_THREADS=1`


#### Balancing ideas

```
float row_size = (float)row_length / (float)(variableCount - 1);
float cpu_row_count = cpuExecutor->tasks.size();
if (row_size < 0.3f || (row_size < 0.4f && (variableCount - row) <= ompThreadCount - cpu_row_count))
```