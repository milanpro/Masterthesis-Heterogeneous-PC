cmake_minimum_required(VERSION 3.5)
project(heterogeneous-pc LANGUAGES CXX CUDA)

add_compile_definitions(WITH_CUDA_ATOMICS=1)

add_subdirectory(src)
