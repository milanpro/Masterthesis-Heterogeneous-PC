FROM milanpro/cmake-armadillo-python38-centos8:latest

# Executable 
RUN mkdir -p /usr/src/project/src
COPY ./CMakeLists.txt /usr/src/project
COPY ./src /usr/src/project/src
RUN mkdir /usr/src/project/build
WORKDIR /usr/src/project/build
RUN cmake -DCMAKE_BUILD_TYPE=Release ..
RUN make
COPY ./benchmark.py /usr/src/project/benchmark.py
RUN mkdir -p /usr/src/project/benchmarks
RUN mkdir -p /usr/src/project/logs
COPY ./benchmarks/benchmarks_docker.json /usr/src/project/benchmarks/benchmarks.json
RUN mkdir -p /usr/src/project/data
COPY ./data/cooling_house.csv /usr/src/project/data/cooling_house.csv
WORKDIR /usr/src/project
CMD [ "python3", "./benchmark.py" ]