# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.25)
project(meshprep-package-consumer LANGUAGES CXX CUDA)

find_package(meshprep-cuda CONFIG REQUIRED)
add_executable(package-consumer main.cu)
target_link_libraries(package-consumer PRIVATE meshprep::meshprep-cuda)
