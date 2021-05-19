cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

SET(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

set(SOURCE_FILES
    correlation/corHelper.cu
    correlation/corOwn.cu
    correlation/corOwn.cuh
    executor/executor.hpp
    executor/cpuExecutor.hpp
    executor/cpuExecutor.cpp
    executor/gpuExecutor.cuh
    executor/gpuExecutor.cu
    executor/independence/cpuInd.cpp
    executor/independence/cpuInd.hpp
    executor/independence/gpuInd.cu
    executor/independence/gpuInd.cuh
    executor/testing/cpuRowTests.cpp
    executor/testing/cpuRowTests.hpp
    executor/testing/cpuUtil.cpp
    executor/testing/cpuUtil.hpp
    executor/testing/cpuWorkstealingTests.cpp
    executor/testing/cpuWorkstealingTests.cpp
    executor/testing/gpuRowTests.cu
    executor/testing/gpuRowTests.cuh
    executor/testing/gpuWorkstealingTests.cu
    executor/testing/gpuWorkstealingTests.cuh
    independence/skeleton.cpp
    independence/skeleton.hpp
    independence/compact.cu
    independence/compact.cuh
    loadbalance/balancer.cpp
    loadbalance/balancer.hpp
    util/indep_util.hpp
    util/cuda_util.cu
    util/cuda_util.cuh
    util/state.cu
    util/state.cuh
    util/csv_parser.hpp
    util/csv_parser.cpp
    util/sepset_util.hpp
    util/assertions.hpp
    util/assertions.cpp
)

add_executable(heterogpc main.cpp ${SOURCE_FILES})
 
set_property(TARGET heterogpc 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET heterogpc 
             PROPERTY CUDA_ARCHITECTURES 70)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${BOOST_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(heterogpc
    ${ARMADILLO_LIBRARIES}
    ${Boost_LIBRARIES}
    OpenMP::OpenMP_CXX
)

IF (OPENMP_FOUND)
    IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp-simd")
    ELSE()
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    ENDIF()
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}") 
ENDIF()  

IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G") 
ELSE()
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
ENDIF()

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET heterogpc 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
