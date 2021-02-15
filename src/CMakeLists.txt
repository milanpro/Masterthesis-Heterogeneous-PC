cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(SOURCE_FILES
    csv_parser.hpp
    csv_parser.cpp
    correlation/corHelper.cu
    correlation/corOwn.cu
    correlation/corOwn.cuh
    independence/skeleton.cu
    independence/skeleton.cuh
    independence/cpu.cu
    independence/cpu.cuh
    independence/gpu.cu
    independence/gpu.cuh
    independence/compact.cu
    independence/compact.cuh
    util/constants.hpp
    util/indep_util.hpp
    util/concurrentqueue.h
    util/cuda_util.cu
    util/cuda_util.cuh
    util/matrix_print.cuh
    util/mem_info.cuh
    util/mem_info.cu
    util/state.cu
    util/state.cuh
)

add_executable(heterogpc main.cpp ${SOURCE_FILES})
 
set_property(TARGET heterogpc 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET heterogpc 
             PROPERTY CUDA_ARCHITECTURES 70)

find_package(OpenMP REQUIRED)
find_package(GSL REQUIRED)
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

include_directories(${LAPACK_INCLUDE_DIR})
include_directories(${BLAS_INCLUDE_DIR})
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${BOOST_INCLUDE_DIRS})

target_link_libraries(heterogpc
    GSL::gsl
    GSL::gslcblas
    ${ARMADILLO_LIBRARIES}
    ${Boost_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
)

IF (OPENMP_FOUND) 
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}") 
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}") 
ENDIF()  

SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G") 

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET heterogpc 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()