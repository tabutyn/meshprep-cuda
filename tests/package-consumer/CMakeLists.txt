# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.25)
project(parallel-mater-package-consumer LANGUAGES CXX CUDA)

find_package(ParallelMater CONFIG REQUIRED)
add_executable(package-consumer main.cu
    header_cloth.cpp
    header_coupling.cpp
    header_fluid.cpp
    header_paint.cpp
    header_rigid_body.cpp
    header_rope.cpp
    header_smoke.cpp
    header_soft_body.cpp)
target_link_libraries(package-consumer PRIVATE ParallelMater::physics)
