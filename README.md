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

#### Download Benchmarks and Logs

Delos:

`rsync -aP "delos:~/Masterthesis-Heterogeneous-PC/benchmarks/*" ./benchmarks_delos`

`rsync -aP "delos:~/Masterthesis-Heterogeneous-PC/logs/*" ./logs_delos`

AC922:

`rsync -aP "ac92202:/scratch/milan.proell/enroot/enroot/milan-heterogpc-bench/usr/src/project/benchmarks/*" ./benchmarks_ac922`

`rsync -aP "ac92202:/scratch/milan.proell/enroot/enroot/milan-heterogpc-bench/usr/src/project/logs/*" ./logs_ac922`

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

#### AC922 Power9 Findings

https://on-demand.gputechconf.com/gtc/2018/presentation/s8430-everything-you-need-to-know-about-unified-memory.pdf
https://www.olcf.ornl.gov/wp-content/uploads/2018/03/ORNL_workshop_mar2018.pdf

#### AC922 NUMA Topology

```
[milan.proell@ac922-02 ~]$ nvidia-smi topo -m
	GPU0	GPU1	GPU2	GPU3	mlx5_0	mlx5_1	mlx5_2	mlx5_3	mlx5_4	mlx5_5	CPU Affinity	NUMA Affinity
GPU0	 X 	NV3	SYS	SYS	NODE	NODE	SYS	SYS	SYS	SYS	0-63	0
GPU1	NV3	 X 	SYS	SYS	NODE	NODE	SYS	SYS	SYS	SYS	0-63	0
GPU2	SYS	SYS	 X 	NV3	SYS	SYS	NODE	NODE	NODE	NODE	64-127	8
GPU3	SYS	SYS	NV3	 X 	SYS	SYS	NODE	NODE	NODE	NODE	64-127	8
mlx5_0	NODE	NODE	SYS	SYS	 X 	PIX	SYS	SYS	SYS	SYS
mlx5_1	NODE	NODE	SYS	SYS	PIX	 X 	SYS	SYS	SYS	SYS
mlx5_2	SYS	SYS	NODE	NODE	SYS	SYS	 X 	PIX	NODE	NODE
mlx5_3	SYS	SYS	NODE	NODE	SYS	SYS	PIX	 X 	NODE	NODE
mlx5_4	SYS	SYS	NODE	NODE	SYS	SYS	NODE	NODE	 X 	PIX
mlx5_5	SYS	SYS	NODE	NODE	SYS	SYS	NODE	NODE	PIX	 X

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
[milan.proell@ac922-02 ~]$ numactl -H
available: 6 nodes (0,8,252-255)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
node 0 size: 257716 MB
node 0 free: 239932 MB
node 8 cpus: 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
node 8 size: 261728 MB
node 8 free: 244415 MB
node 252 cpus:
node 252 size: 32256 MB
node 252 free: 30807 MB
node 253 cpus:
node 253 size: 32256 MB
node 253 free: 30823 MB
node 254 cpus:
node 254 size: 32256 MB
node 254 free: 30808 MB
node 255 cpus:
node 255 size: 32256 MB
node 255 free: 30823 MB
node distances:
node   0   8  252  253  254  255
  0:  10  40  80  80  80  80
  8:  40  10  80  80  80  80
 252:  80  80  10  80  80  80
 253:  80  80  80  10  80  80
 254:  80  80  80  80  10  80
 255:  80  80  80  80  80  10
 ```
