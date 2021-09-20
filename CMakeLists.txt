cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0074 NEW)
project(heterogeneous-pc LANGUAGES CXX CUDA)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
# Set this definition to 0 if no system wide atomics should be used for the workstealing approach
add_compile_definitions(WITH_CUDA_ATOMICS=1)
# Set this definition to 0 if edge deletion should not migrate after CPU execution
add_compile_definitions(MIGRATE_EDGES=0)

add_subdirectory(src)
