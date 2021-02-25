### Masterthesis-Heterogeneous-PC

https://github.com/ChristopherSchmidt89/EPIC-causality

https://github.com/philipp-bode/lock-free-pc

https://github.com/LIS-Laboratory/cupc

https://github.com/cran/pcalg

#### TCGA Datasets
Can be found in: `/home/Christopher.Hagedorn/genData`

`/home/Milan.Proell/Masterthesis-Heterogeneous-PC/build/src/heterogpc --corr -i "/home/Christopher.Hagedorn/genData/TCGA-GBM-100-cor.csv" -o 3190 -v -t 80`
 
#### Debugging
/usr/local/cuda/bin/cuda-gdb /home/Milan.Proell/Masterthesis-Heterogeneous-PC/build/src/heterogpc

#### Bugs
https://github.com/xianyi/OpenBLAS/wiki/Faq#multi-threaded
Prevent threading segfault : `export OPENBLAS_NUM_THREADS=1`
