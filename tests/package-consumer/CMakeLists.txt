# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.25)
project(parallel-mater-package-consumer LANGUAGES CXX CUDA)

find_package(ParallelMater CONFIG REQUIRED)
add_executable(package-consumer main.cu)
target_link_libraries(package-consumer PRIVATE ParallelMater::gallery)
