cmake_minimum_required(VERSION 3.5)
project(heterogeneous-pc LANGUAGES CXX CUDA)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
add_compile_definitions(WITH_CUDA_ATOMICS=1)
add_compile_definitions(MIGRATE_EDGES=1)

add_subdirectory(src)
