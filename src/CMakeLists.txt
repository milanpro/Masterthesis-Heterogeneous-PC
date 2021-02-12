cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(SOURCE_FILES
    csv_parser.hpp
    csv_parser.cpp
    correlation/corHelper.cu
    correlation/corOwn.cu
    correlation/corOwn.cuh
    independence/mm_indepTests.cu
    independence/mm_indepTests.cuh
    independence/mm_test.cu
    independence/mm_test.cuh
    independence/skeleton.cu
    independence/skeleton.cuh
    independence/cpu.cu
    independence/cpu.cuh
    independence/gpu.cu
    independence/gpu.cuh
    independence/rowIndTest.cu
    independence/rowIndTest.cuh
    util/constants.hpp
    util/indepUtil.hpp
    util/concurrentqueue.hpp
    util/cudaUtil.cuh
    util/GPUData.cuh
    util/GPUData.cu
    util/matrixPrint.cuh
    util/memInfo.cuh
    util/memInfo.cu
    util/State.cu
    util/State.cuh
    loadbalancing/balancer.cpp
    loadbalancing/balancer.hpp
)

add_executable(heterogpc main.cpp ${SOURCE_FILES})
 
set_property(TARGET heterogpc 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET heterogpc 
             PROPERTY CUDA_ARCHITECTURES 70)

find_package(OpenMP REQUIRED)
find_package(GSL REQUIRED)
find_package(Armadillo REQUIRED)
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)

target_link_libraries(heterogpc
    GSL::gsl
    ${ARMADILLO_LIBRARIES}
    ${Boost_LIBRARIES}
    OpenMP::OpenMP_CXX
)

IF (OPENMP_FOUND) 
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}") 
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}") 
ENDIF()  

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET heterogpc 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()