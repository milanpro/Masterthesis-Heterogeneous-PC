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

#### TCGA Datasets
Can be found in: `/home/Christopher.Hagedorn/genData`

`/home/Milan.Proell/Masterthesis-Heterogeneous-PC/build/src/heterogpc --corr -i "/home/Christopher.Hagedorn/genData/TCGA-GBM-100-cor.csv" -o 3190 -v -t 80`
 
#### Debugging
/usr/local/cuda/bin/cuda-gdb /home/Milan.Proell/Masterthesis-Heterogeneous-PC/build/src/heterogpc

#### Bugs
https://github.com/xianyi/OpenBLAS/wiki/Faq#multi-threaded
Prevent threading segfault : `export OPENBLAS_NUM_THREADS=1`


#### Balancing ideas

```
      float row_size = (float)row_length / (float)(variableCount - 1);
      float cpu_row_count = cpuExecutor->tasks.size();
      if (row_size < 0.3f || (row_size < 0.4f && (variableCount - row) <= ompThreadCount - cpu_row_count))
```