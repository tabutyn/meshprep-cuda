# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.25)
project(ParallelMater VERSION 0.2.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(PARALLEL_MATER_BUILD_TESTS "Build GPU correctness tests" OFF)
option(PARALLEL_MATER_BUILD_EXAMPLES "Build examples" OFF)
option(PARALLEL_MATER_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(PARALLEL_MATER_BUILD_GALLERY "Build the optional eight-context gallery API" OFF)
option(PARALLEL_MATER_BUILD_LAB "Build the native CUDA/OpenGL simulation lab" OFF)
option(PARALLEL_MATER_BUILD_CAPTURE_TOOLS "Build headless capture audit tools" OFF)
option(PARALLEL_MATER_ENABLE_NVTX "Annotate public operations with NVTX ranges" ON)

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
    message(FATAL_ERROR "ParallelMater requires CUDA Toolkit 12.6 or newer")
endif()

if(PARALLEL_MATER_BUILD_GALLERY OR PARALLEL_MATER_BUILD_LAB)
    # Recipe, validation, and campaign layer used only by the optional gallery
    # and native application. It has no rendering or CUDA-runtime dependency.
    add_library(meshprep-game src/game.cpp)
    add_library(meshprep::meshprep-game ALIAS meshprep-game)
    add_library(ParallelMater::game ALIAS meshprep-game)
    target_include_directories(
        meshprep-game
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_compile_features(meshprep-game PUBLIC cxx_std_20)
    target_compile_options(meshprep-game PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(meshprep-game PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME parallel-mater-game
        EXPORT_NAME game)
endif()

add_library(meshprep-cuda src/meshprep.cu)
add_library(meshprep::meshprep-cuda ALIAS meshprep-cuda)
add_library(ParallelMater::geometry ALIAS meshprep-cuda)
target_include_directories(
    meshprep-cuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(meshprep-cuda PUBLIC cxx_std_20)
target_compile_definitions(
    meshprep-cuda PRIVATE MESHPREP_ENABLE_NVTX=$<BOOL:${PARALLEL_MATER_ENABLE_NVTX}>)
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
        OUTPUT_NAME parallel-mater-geometry
        EXPORT_NAME geometry)

# General-purpose fixed-topology soft-body solver. Applications own scene
# policy, contacts, rendering, input, and their CUDA coupling kernels.
add_library(
    meshprep-physics
    src/physics.cpp
    apps/water_lab/soft_body.cu)
add_library(meshprep::physics ALIAS meshprep-physics)
add_library(ParallelMater::physics ALIAS meshprep-physics)
target_link_libraries(meshprep-physics PUBLIC ParallelMater::geometry)
target_include_directories(
    meshprep-physics
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/apps/water_lab)
target_compile_features(meshprep-physics PUBLIC cxx_std_20)
target_compile_options(
    meshprep-physics
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
set_target_properties(
    meshprep-physics
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION OFF
        OUTPUT_NAME parallel-mater-physics
        EXPORT_NAME physics)

if(PARALLEL_MATER_BUILD_GALLERY OR PARALLEL_MATER_BUILD_LAB)
    # Optional headless gallery/example composition. It intentionally excludes
    # GLFW, OpenGL, the ray tracer, HUD, capture tools, and visual reconstruction.
    add_library(
        meshprep-simulation
        src/simulation.cpp
        apps/water_lab/simulation_gallery.cpp
        apps/water_lab/cloth.cpp
        apps/water_lab/geodesic_sphere.cpp
        apps/water_lab/hybrid_kernels.cu)
    add_library(meshprep::meshprep-simulation ALIAS meshprep-simulation)
    add_library(ParallelMater::gallery ALIAS meshprep-simulation)
    target_link_libraries(
        meshprep-simulation PUBLIC ParallelMater::physics ParallelMater::game)
    target_include_directories(
        meshprep-simulation
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/apps/water_lab)
    target_compile_features(meshprep-simulation PUBLIC cxx_std_20)
    target_compile_options(
        meshprep-simulation
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(
        meshprep-simulation
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CUDA_SEPARABLE_COMPILATION OFF
            OUTPUT_NAME parallel-mater-gallery
            EXPORT_NAME gallery)
endif()

if(PARALLEL_MATER_BUILD_EXAMPLES)
    add_executable(parallel-mater-geometry-example examples/basic.cu)
    target_link_libraries(
        parallel-mater-geometry-example PRIVATE ParallelMater::geometry)

    add_executable(parallel-mater-soft-body-example examples/soft_body.cu)
    target_link_libraries(
        parallel-mater-soft-body-example PRIVATE ParallelMater::physics)

    if(TARGET meshprep-simulation)
        add_executable(parallel-mater-gallery-example examples/simulation_contexts.cu)
        target_link_libraries(
            parallel-mater-gallery-example PRIVATE ParallelMater::gallery)
    endif()
endif()

if(PARALLEL_MATER_BUILD_BENCHMARKS)
    add_executable(meshprep-benchmark benchmarks/benchmark.cu)
    target_link_libraries(meshprep-benchmark PRIVATE meshprep::meshprep-cuda)
    target_compile_options(
        meshprep-benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info>)
endif()

if(PARALLEL_MATER_BUILD_BENCHMARKS OR PARALLEL_MATER_BUILD_TESTS)
    add_executable(
        meshprep-contact-experiment
        apps/water_lab/contact_experiment.cpp
        apps/water_lab/geodesic_sphere.cpp
        apps/water_lab/soft_body.cu
        apps/water_lab/hybrid_kernels.cu)
    target_link_libraries(meshprep-contact-experiment PRIVATE meshprep::meshprep-cuda)
    target_compile_options(
        meshprep-contact-experiment PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(
        meshprep-contact-experiment PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

if(PARALLEL_MATER_BUILD_CAPTURE_TOOLS)
    add_executable(
        meshprep-soft-body-capture-audit
        apps/water_lab/soft_body_capture_audit.cpp
        apps/water_lab/geodesic_sphere.cpp
        apps/water_lab/soft_body.cu
        apps/water_lab/hybrid_kernels.cu)
    target_link_libraries(meshprep-soft-body-capture-audit PRIVATE meshprep::meshprep-cuda)
    target_compile_definitions(meshprep-soft-body-capture-audit PRIVATE
        MESHPREP_SOFT_BODY_ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
    target_compile_options(
        meshprep-soft-body-capture-audit PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(
        meshprep-soft-body-capture-audit PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

if(PARALLEL_MATER_BUILD_TESTS)
    enable_testing()
    if(PARALLEL_MATER_BUILD_CAPTURE_TOOLS)
        add_test(NAME meshprep-skin-contact-reference
            COMMAND meshprep-soft-body-capture-audit --skin-contact-self-test)
    endif()
    if(TARGET meshprep-simulation)
        add_executable(meshprep-simulation-api-tests tests/simulation_api_tests.cpp)
        target_link_libraries(meshprep-simulation-api-tests PRIVATE meshprep::meshprep-simulation)
        target_compile_options(meshprep-simulation-api-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
        add_test(NAME meshprep-simulation-api-tests COMMAND meshprep-simulation-api-tests)

        add_executable(meshprep-simulation-runtime-tests tests/simulation_runtime_tests.cu)
        target_link_libraries(
            meshprep-simulation-runtime-tests PRIVATE meshprep::meshprep-simulation)
        target_compile_definitions(meshprep-simulation-runtime-tests PRIVATE
            MESHPREP_SIMULATION_TEST_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
        target_compile_options(meshprep-simulation-runtime-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info>)
        set_target_properties(meshprep-simulation-runtime-tests
            PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
        add_test(NAME meshprep-simulation-runtime-tests
            COMMAND meshprep-simulation-runtime-tests)
        set_tests_properties(meshprep-simulation-runtime-tests PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(meshprep-tests tests/meshprep_tests.cu)
    target_link_libraries(meshprep-tests PRIVATE meshprep::meshprep-cuda)
    add_test(NAME meshprep-tests COMMAND meshprep-tests)
    set_tests_properties(meshprep-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(meshprep-physics-tests tests/physics_tests.cu)
    target_link_libraries(meshprep-physics-tests PRIVATE meshprep::physics)
    target_compile_definitions(meshprep-physics-tests PRIVATE
        MESHPREP_PHYSICS_TEST_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
    target_compile_options(meshprep-physics-tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info>)
    set_target_properties(meshprep-physics-tests
        PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME meshprep-physics-tests COMMAND meshprep-physics-tests)
    set_tests_properties(meshprep-physics-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(meshprep-triangle-contact-tests tests/triangle_contact_tests.cu)
    target_compile_options(meshprep-triangle-contact-tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info>)
    add_test(NAME meshprep-triangle-contact-host
        COMMAND meshprep-triangle-contact-tests --host-only)
    add_test(NAME meshprep-triangle-contact-gpu COMMAND meshprep-triangle-contact-tests)
    set_tests_properties(meshprep-triangle-contact-gpu PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(
        meshprep-hybrid-tests
        tests/hybrid_tests.cu
        apps/water_lab/geodesic_sphere.cpp
        apps/water_lab/fluid_surface.cu
        apps/water_lab/fluid_visuals.cu
        apps/water_lab/soft_body.cu
        apps/water_lab/water_kernels.cu
        apps/water_lab/hybrid_kernels.cu)
    target_link_libraries(meshprep-hybrid-tests PRIVATE meshprep::meshprep-cuda)
    target_compile_options(
        meshprep-hybrid-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    target_compile_definitions(meshprep-hybrid-tests PRIVATE
        MESHPREP_SOFT_BODY_TEST_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
    set_target_properties(meshprep-hybrid-tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME meshprep-hybrid-tests COMMAND meshprep-hybrid-tests)
    set_tests_properties(meshprep-hybrid-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(
        meshprep-course-tests
        tests/course_tests.cu
        apps/water_lab/geodesic_sphere.cpp
        apps/water_lab/soft_body.cu
        apps/water_lab/hybrid_kernels.cu)
    target_link_libraries(meshprep-course-tests PRIVATE meshprep::meshprep-cuda)
    target_compile_options(
        meshprep-course-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(meshprep-course-tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME meshprep-course-host COMMAND meshprep-course-tests --host-only)
    add_test(NAME meshprep-course-gpu COMMAND meshprep-course-tests)
    set_tests_properties(meshprep-course-gpu PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(meshprep-fluid-visual-tests
        tests/fluid_visual_tests.cu apps/water_lab/fluid_surface.cu apps/water_lab/fluid_visuals.cu)
    target_link_libraries(meshprep-fluid-visual-tests PRIVATE meshprep::meshprep-cuda)
    target_compile_options(meshprep-fluid-visual-tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(meshprep-fluid-visual-tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME meshprep-fluid-visual-tests COMMAND meshprep-fluid-visual-tests)
    set_tests_properties(meshprep-fluid-visual-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(meshprep-soft-body-tests
        tests/soft_body_tests.cu apps/water_lab/soft_body.cu)
    target_link_libraries(meshprep-soft-body-tests PRIVATE meshprep::meshprep-cuda)
    target_compile_definitions(meshprep-soft-body-tests PRIVATE
        MESHPREP_SOFT_BODY_TEST_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
    target_compile_options(meshprep-soft-body-tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(meshprep-soft-body-tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME meshprep-soft-body-tests COMMAND meshprep-soft-body-tests)
    set_tests_properties(meshprep-soft-body-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(meshprep-cloth-tests
        tests/cloth_tests.cu apps/water_lab/cloth.cpp
        apps/water_lab/geodesic_sphere.cpp apps/water_lab/soft_body.cu)
    target_link_libraries(meshprep-cloth-tests PRIVATE meshprep::meshprep-cuda)
    target_compile_options(meshprep-cloth-tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(meshprep-cloth-tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME meshprep-cloth-tests COMMAND meshprep-cloth-tests)
    set_tests_properties(meshprep-cloth-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(meshprep-simulation-gallery-tests
        tests/simulation_gallery_tests.cu
        apps/water_lab/simulation_gallery.cpp
        apps/water_lab/cloth.cpp
        apps/water_lab/geodesic_sphere.cpp
        apps/water_lab/soft_body.cu
        apps/water_lab/hybrid_kernels.cu)
    target_link_libraries(meshprep-simulation-gallery-tests PRIVATE meshprep::meshprep-cuda)
    target_compile_definitions(meshprep-simulation-gallery-tests PRIVATE
        MESHPREP_SOFT_BODY_TEST_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
    target_compile_options(meshprep-simulation-gallery-tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(meshprep-simulation-gallery-tests
        PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME meshprep-simulation-gallery-tests
        COMMAND meshprep-simulation-gallery-tests)
    set_tests_properties(meshprep-simulation-gallery-tests PROPERTIES SKIP_RETURN_CODE 77)

    if(TARGET meshprep-contact-experiment)
        add_test(
            NAME meshprep-contact-stability
            COMMAND meshprep-contact-experiment
                --runs 1
                --iterations 1
                --material default
                --output /tmp/meshprep-contact-experiment/ctest)
        set_tests_properties(meshprep-contact-stability PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

if(PARALLEL_MATER_BUILD_LAB)
    find_package(OpenGL QUIET)
    find_package(glfw3 3.3 QUIET)
    if(OpenGL_FOUND AND TARGET glfw)
        add_library(
            parallel-mater-gallery-runtime
            apps/water_lab/fluid_surface.cu
            apps/water_lab/fluid_visuals.cu
            apps/water_lab/water_kernels.cu)
        target_link_libraries(
            parallel-mater-gallery-runtime PUBLIC ParallelMater::gallery)
        target_include_directories(parallel-mater-gallery-runtime
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/apps/water_lab)
        target_compile_definitions(parallel-mater-gallery-runtime PRIVATE
            MESHPREP_SOFT_BODY_ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
        target_compile_options(
            parallel-mater-gallery-runtime
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
                $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
        set_target_properties(
            parallel-mater-gallery-runtime PROPERTIES
                POSITION_INDEPENDENT_CODE ON
                CUDA_SEPARABLE_COMPILATION OFF)

        add_executable(parallel-mater-lab apps/water_lab/main.cpp)
        target_link_libraries(
            parallel-mater-lab PRIVATE parallel-mater-gallery-runtime
            glfw OpenGL::GL)
        target_compile_definitions(parallel-mater-lab PRIVATE
            MESHPREP_SOFT_BODY_ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
        target_compile_options(parallel-mater-lab PRIVATE
            -Wall -Wextra -Wpedantic)
    else()
        message(STATUS "parallel-mater-lab disabled: OpenGL and glfw3 3.3 are required")
    endif()
endif()

set(meshprep_install_targets meshprep-cuda meshprep-physics)
if(TARGET meshprep-game)
    list(APPEND meshprep_install_targets meshprep-game)
endif()
if(TARGET meshprep-simulation)
    list(APPEND meshprep_install_targets meshprep-simulation)
endif()
install(TARGETS ${meshprep_install_targets} EXPORT ParallelMaterTargets)
install(FILES
    include/meshprep/meshprep.hpp
    include/meshprep/physics.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/meshprep)
install(FILES
    include/parallel_mater/geometry.hpp
    include/parallel_mater/physics.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/parallel_mater)
if(TARGET meshprep-game)
    install(FILES include/meshprep/game.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/meshprep)
    install(FILES include/parallel_mater/game.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/parallel_mater)
endif()
if(TARGET meshprep-simulation)
    install(FILES include/meshprep/simulation.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/meshprep)
    install(FILES include/parallel_mater/gallery.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/parallel_mater)
endif()
install(
    FILES LICENSE THIRD_PARTY_NOTICES.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/ParallelMater)
install(
    EXPORT ParallelMaterTargets
    FILE ParallelMaterTargets.cmake
    NAMESPACE ParallelMater::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ParallelMater)

configure_package_config_file(
    cmake/ParallelMaterConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ParallelMater)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ParallelMater)
