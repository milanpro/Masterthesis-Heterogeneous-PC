cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(SOURCE_FILES
    csv_parser.hpp
    csv_parser.cpp
    State.cu
    State.cuh
    cudaUtil.cuh
)

add_executable(heterogpc main.cpp ${SOURCE_FILES})
 
set_property(TARGET heterogpc 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

find_package(GSL REQUIRED)
find_package(Armadillo REQUIRED)

include_directories(${ARMADILLO_INCLUDE_DIRS})

target_link_libraries(heterogpc
    GSL::gsl
    ${ARMADILLO_LIBRARIES}
)

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET heterogpc 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()