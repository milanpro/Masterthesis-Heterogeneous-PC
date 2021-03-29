FROM nvidia/cuda-ppc64le:11.2.2-devel-centos8

RUN yum update 
RUN yum -y install yum-utils
RUN yum -y groupinstall 'Development Tools'
RUN yum -y install wget dnf-plugins-core
RUN yum config-manager --set-enabled powertools
RUN dnf --enablerepo=powertools -y install \
    openblas-devel \
    lapack-devel \
    openssl-devel \
    boost-devel \
    python38 \
    python38-devel
RUN yum clean all

# CMake
RUN mkdir -p /tmp/cmake && \
    cd /tmp/cmake && \
    wget 'https://cmake.org/files/v3.20/cmake-3.20.0.tar.gz' && \
    tar -xzf cmake-3.20.0.tar.gz && \
    cd /tmp/cmake/cmake-3.20.0 && \
    ./bootstrap --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && \
    rm -rf /tmp/cmake

# Armadillo
RUN mkdir -p /tmp/armadillo && \
    cd /tmp/armadillo && \
    wget 'https://sourceforge.net/projects/arma/files/armadillo-10.3.0.tar.xz' && \
    tar -xJf armadillo-10.3.0.tar.xz && \
    cd /tmp/armadillo/armadillo-10.3.0 && \
    ./configure && \
    make && \
    make install && \
    cd /tmp && \
    rm -rf /tmp/armadillo

# Python
RUN pip3 install Cython
RUN pip3 install numpy
RUN pip3 install pandas

# Executable 
ADD . /usr/src/project
RUN mkdir /usr/src/project/build
WORKDIR /usr/src/project/build
RUN cmake -DCMAKE_BUILD_TYPE=Release ..
RUN make
