# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.25)
project(meshprep-cuda VERSION 0.1.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(MESHPREP_BUILD_TESTS "Build GPU correctness tests" ON)
option(MESHPREP_BUILD_EXAMPLES "Build examples" ON)
option(MESHPREP_BUILD_BENCHMARKS "Build benchmarks" ON)
option(MESHPREP_ENABLE_NVTX "Annotate public operations with NVTX ranges" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "--std=c++20")
set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "--std=c++20")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.6)
    message(FATAL_ERROR "meshprep-cuda requires CUDA Toolkit 12.6 or newer")
endif()

add_library(meshprep-cuda src/meshprep.cu)
add_library(meshprep::meshprep-cuda ALIAS meshprep-cuda)
target_include_directories(
    meshprep-cuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(meshprep-cuda PUBLIC cxx_std_20)
target_compile_definitions(
    meshprep-cuda PRIVATE MESHPREP_ENABLE_NVTX=$<BOOL:${MESHPREP_ENABLE_NVTX}>)
target_compile_options(
    meshprep-cuda
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
set_target_properties(
    meshprep-cuda
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION OFF
        EXPORT_NAME meshprep-cuda)

if(MESHPREP_BUILD_EXAMPLES)
    add_executable(meshprep-example examples/basic.cu)
    target_link_libraries(meshprep-example PRIVATE meshprep::meshprep-cuda)
endif()

if(MESHPREP_BUILD_BENCHMARKS)
    add_executable(meshprep-benchmark benchmarks/benchmark.cu)
    target_link_libraries(meshprep-benchmark PRIVATE meshprep::meshprep-cuda)
    target_compile_options(
        meshprep-benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info>)
endif()

if(MESHPREP_BUILD_TESTS)
    enable_testing()
    add_executable(meshprep-tests tests/meshprep_tests.cu)
    target_link_libraries(meshprep-tests PRIVATE meshprep::meshprep-cuda)
    add_test(NAME meshprep-tests COMMAND meshprep-tests)
endif()

install(TARGETS meshprep-cuda EXPORT meshprep-cuda-targets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    EXPORT meshprep-cuda-targets
    FILE meshprep-cuda-targets.cmake
    NAMESPACE meshprep::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshprep-cuda)

configure_package_config_file(
    cmake/meshprep-cuda-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/meshprep-cuda-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshprep-cuda)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/meshprep-cuda-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/meshprep-cuda-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/meshprep-cuda-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshprep-cuda)
