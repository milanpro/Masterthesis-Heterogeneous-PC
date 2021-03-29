FROM nvidia/cuda-ppc64le:11.2.2-devel-centos8 AS builder
# FROM nvidia/cuda:11.2.2-devel AS builder
# ARG DEBIAN_FRONTEND=noninteractive
# ENV TZ=Europe/Berlin
RUN yum update 
RUN yum -y install yum-utils openblas lapack
RUN yum -y groupinstall 'Development Tools'
RUN yum -y install wget gcc-gfortran

# MASS Support
# RUN mkdir -p /tmp/MASS
# WORKDIR /tmp/MASS
# RUN wget http://public.dhe.ibm.com/software/server/POWER/Linux/xl-compiler/eval/ppc64le/rhel7/repodata/repomd.xml.key
# RUN rpm --import repomd.xml.key
# RUN wget http://public.dhe.ibm.com/software/server/POWER/Linux/xl-compiler/eval/ppc64le/rhel7/ibm-xl-compiler-eval.repo
# RUN cp ibm-xl-compiler-eval.repo /etc/yum.repos.d/
# RUN yum install libxlmass-devel.8.1.5
# RUN yum clean all

# CMake
RUN mkdir -p /tmp/cmake
WORKDIR /tmp/cmake
RUN wget 'https://cmake.org/files/v3.20/cmake-3.20.0-linux-x86_64.sh'
RUN sh cmake-3.20.0-linux-x86_64.sh -- --skip-license --prefix=/usr/local
# WORKDIR /tmp/cmake/cmake-3.20.0
# RUN ./bootstrap --prefix=/usr/local
# RUN make -j$(nproc)
# RUN make install
ENV PATH "/usr/local/bin:${PATH}"

# OpenBlas
RUN mkdir -p /tmp/openblas
WORKDIR /tmp/openblas
RUN wget 'https://github.com/xianyi/OpenBLAS/releases/download/v0.3.14/OpenBLAS-0.3.14.tar.gz'
RUN tar -xzf OpenBLAS-0.3.14.tar.gz
WORKDIR /tmp/openblas/OpenBLAS-0.3.14
RUN make USE_MASS=1 TARGET=POWER9

# Armadillo
RUN mkdir -p /tmp/armadillo
WORKDIR /tmp/armadillo
RUN wget 'https://sourceforge.net/projects/arma/files/armadillo-10.3.0.tar.xz'
RUN tar -xJf armadillo-10.3.0.tar.xz
WORKDIR /tmp/armadillo/armadillo-10.3.0
RUN cmake --version
RUN ./configure
RUN make
RUN make install

# Executable 
WORKDIR /usr/src/project
ADD . /usr/src/project
RUN cmake .