FROM nvidia/cuda-ppc64le:11.2.2-devel-centos8 AS builder

RUN yum update 
RUN yum -y install yum-utils
RUN yum -y groupinstall 'Development Tools'
RUN yum -y install wget gcc-gfortran dnf-plugins-core
RUN yum config-manager --set-enabled powertools
RUN dnf --enablerepo=powertools -y install openblas-devel lapack-devel openssl-devel boost
RUN yum clean all

# CMake
RUN mkdir -p /tmp/cmake
WORKDIR /tmp/cmake
RUN wget 'https://cmake.org/files/v3.20/cmake-3.20.0.tar.gz'
RUN tar -xzf cmake-3.20.0.tar.gz
WORKDIR /tmp/cmake/cmake-3.20.0
RUN ./bootstrap --prefix=/usr/local
RUN make -j$(nproc)
RUN make install
ENV PATH "/usr/local/bin:${PATH}"
RUN rm -rf /tmp/cmake

# Armadillo
RUN mkdir -p /tmp/armadillo
WORKDIR /tmp/armadillo
RUN wget 'https://sourceforge.net/projects/arma/files/armadillo-10.3.0.tar.xz'
RUN tar -xJf armadillo-10.3.0.tar.xz
WORKDIR /tmp/armadillo/armadillo-10.3.0
RUN ./configure
RUN make
RUN make install
RUN rm -rf /tmp/armadillo

# Executable 
WORKDIR /usr/src/project
ADD . /usr/src/project
RUN cmake .