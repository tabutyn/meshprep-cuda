# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.25)
project(ParallelMater VERSION 0.4.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(PARALLEL_MATER_BUILD_TESTS "Build GPU correctness tests" OFF)
option(PARALLEL_MATER_BUILD_EXAMPLES "Build examples" OFF)
option(PARALLEL_MATER_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(PARALLEL_MATER_BUILD_GALLERY "Build the example recipe gallery" OFF)
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
    # Recipe names and gallery composition are example policy, not installed
    # physics API.  They remain reusable by the native lab and example tests.
    add_library(parallel-mater-example-recipes examples/gallery/recipes.cpp)
    target_include_directories(
        parallel-mater-example-recipes
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/gallery
            ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(parallel-mater-example-recipes PUBLIC cxx_std_20)
    target_compile_options(
        parallel-mater-example-recipes PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(parallel-mater-example-recipes PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
endif()

add_library(parallel-mater-geometry src/geometry.cu)
add_library(ParallelMater::geometry ALIAS parallel-mater-geometry)
target_include_directories(
    parallel-mater-geometry
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(parallel-mater-geometry PUBLIC cxx_std_20)
target_compile_options(
    parallel-mater-geometry
    PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>
        $<$<COMPILE_LANGUAGE:CXX>:-I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>)
target_compile_definitions(
    parallel-mater-geometry PRIVATE PARALLEL_MATER_ENABLE_NVTX=$<BOOL:${PARALLEL_MATER_ENABLE_NVTX}>)
target_compile_options(
    parallel-mater-geometry
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
set_target_properties(
    parallel-mater-geometry
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION OFF
        OUTPUT_NAME parallel-mater-geometry
        EXPORT_NAME geometry)

# General-purpose fixed-topology soft-body solver. Applications own scene
# policy, contacts, rendering, input, and their CUDA coupling kernels.
add_library(
    parallel-mater-physics
    src/frame.cpp
    src/coupling.cu
    src/fluid.cu
    src/physics.cpp
    src/rigid_body.cpp
    src/smoke.cu
    src/internal/asset_builders.cpp
    src/internal/fixed_topology.cu)
add_library(ParallelMater::physics ALIAS parallel-mater-physics)
target_link_libraries(parallel-mater-physics PUBLIC ParallelMater::geometry)
target_include_directories(
    parallel-mater-physics
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(parallel-mater-physics PUBLIC cxx_std_20)
target_compile_options(
    parallel-mater-physics PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>)
target_compile_options(
    parallel-mater-physics
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
set_target_properties(
    parallel-mater-physics
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION OFF
        OUTPUT_NAME parallel-mater-physics
        EXPORT_NAME physics)

if(PARALLEL_MATER_BUILD_GALLERY OR PARALLEL_MATER_BUILD_LAB)
    # Optional headless gallery/example composition. It intentionally excludes
    # GLFW, OpenGL, the ray tracer, HUD, capture tools, and visual reconstruction.
    add_library(
        parallel-mater-example-gallery
        examples/gallery/gallery.cpp)
    target_link_libraries(
        parallel-mater-example-gallery
        PUBLIC ParallelMater::physics parallel-mater-example-recipes)
    target_include_directories(
        parallel-mater-example-gallery
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/gallery
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples/gallery)
    target_compile_definitions(parallel-mater-example-gallery PRIVATE
        PARALLEL_MATER_GALLERY_ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
    target_compile_features(parallel-mater-example-gallery PUBLIC cxx_std_20)
    target_compile_options(
        parallel-mater-example-gallery
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(
        parallel-mater-example-gallery
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CUDA_SEPARABLE_COMPILATION OFF)
endif()

if(PARALLEL_MATER_BUILD_EXAMPLES)
    add_executable(parallel-mater-geometry-example examples/basic.cu)
    target_link_libraries(
        parallel-mater-geometry-example PRIVATE ParallelMater::geometry)

    add_executable(parallel-mater-soft-body-example examples/soft_body.cu)
    target_link_libraries(
        parallel-mater-soft-body-example PRIVATE ParallelMater::physics)

    add_executable(parallel-mater-smoke-example examples/smoke.cu)
    target_link_libraries(
        parallel-mater-smoke-example PRIVATE ParallelMater::physics)

    add_executable(parallel-mater-solvers-example examples/solvers.cu)
    target_link_libraries(
        parallel-mater-solvers-example PRIVATE ParallelMater::physics)

    if(TARGET parallel-mater-example-gallery)
        add_executable(parallel-mater-gallery-example examples/simulation_contexts.cu)
        target_link_libraries(
            parallel-mater-gallery-example PRIVATE parallel-mater-example-gallery)
    endif()
endif()

if(PARALLEL_MATER_BUILD_BENCHMARKS)
    add_executable(parallel-mater-benchmark benchmarks/benchmark.cu)
    target_link_libraries(parallel-mater-benchmark PRIVATE ParallelMater::geometry)
    target_compile_options(
        parallel-mater-benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info>)
endif()

if(PARALLEL_MATER_BUILD_BENCHMARKS OR PARALLEL_MATER_BUILD_TESTS)
    add_executable(
        parallel-mater-contact-experiment
        apps/water_lab/contact_experiment.cpp
        apps/water_lab/geodesic_sphere.cpp
        apps/water_lab/soft_body.cu
        apps/water_lab/hybrid_kernels.cu)
    target_link_libraries(parallel-mater-contact-experiment PRIVATE ParallelMater::geometry)
    target_compile_options(
        parallel-mater-contact-experiment PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(
        parallel-mater-contact-experiment PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

if(PARALLEL_MATER_BUILD_CAPTURE_TOOLS)
    add_executable(
        parallel-mater-soft-body-capture-audit
        apps/water_lab/soft_body_capture_audit.cpp
        apps/water_lab/geodesic_sphere.cpp
        apps/water_lab/soft_body.cu
        apps/water_lab/hybrid_kernels.cu)
    target_link_libraries(parallel-mater-soft-body-capture-audit PRIVATE ParallelMater::geometry)
    target_compile_definitions(parallel-mater-soft-body-capture-audit PRIVATE
        PARALLEL_MATER_SOFT_BODY_ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
    target_compile_options(
        parallel-mater-soft-body-capture-audit PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(
        parallel-mater-soft-body-capture-audit PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

if(PARALLEL_MATER_BUILD_TESTS)
    enable_testing()
    if(PARALLEL_MATER_BUILD_CAPTURE_TOOLS)
        add_test(NAME parallel-mater-skin-contact-reference
            COMMAND parallel-mater-soft-body-capture-audit --skin-contact-self-test)
    endif()
    if(TARGET parallel-mater-example-gallery)
        add_executable(parallel-mater-simulation-api-tests tests/simulation_api_tests.cpp)
        target_link_libraries(
            parallel-mater-simulation-api-tests PRIVATE parallel-mater-example-gallery)
        target_compile_options(parallel-mater-simulation-api-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
        add_test(NAME parallel-mater-simulation-api-tests COMMAND parallel-mater-simulation-api-tests)

        add_executable(parallel-mater-gallery-campaign-tests
            tests/campaign_tests.cpp apps/water_lab/campaign.cpp)
        target_link_libraries(
            parallel-mater-gallery-campaign-tests PRIVATE parallel-mater-example-recipes)
        target_compile_options(parallel-mater-gallery-campaign-tests PRIVATE
            -Wall -Wextra -Wpedantic)
        add_test(NAME parallel-mater-gallery-campaign-tests
            COMMAND parallel-mater-gallery-campaign-tests)

        add_executable(parallel-mater-simulation-runtime-tests tests/simulation_runtime_tests.cu)
        target_link_libraries(
            parallel-mater-simulation-runtime-tests PRIVATE parallel-mater-example-gallery)
        target_compile_definitions(parallel-mater-simulation-runtime-tests PRIVATE
            PARALLEL_MATER_SIMULATION_TEST_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
        target_compile_options(parallel-mater-simulation-runtime-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info>)
        set_target_properties(parallel-mater-simulation-runtime-tests
            PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
        add_test(NAME parallel-mater-simulation-runtime-tests
            COMMAND parallel-mater-simulation-runtime-tests)
        set_tests_properties(parallel-mater-simulation-runtime-tests PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(parallel-mater-geometry-tests tests/geometry_tests.cu)
    target_link_libraries(parallel-mater-geometry-tests PRIVATE ParallelMater::geometry)
    add_test(NAME parallel-mater-geometry-tests COMMAND parallel-mater-geometry-tests)
    set_tests_properties(parallel-mater-geometry-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-physics-tests tests/physics_tests.cu)
    target_link_libraries(parallel-mater-physics-tests PRIVATE ParallelMater::physics)
    target_compile_definitions(parallel-mater-physics-tests PRIVATE
        PARALLEL_MATER_PHYSICS_TEST_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
    target_compile_options(parallel-mater-physics-tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info>)
    set_target_properties(parallel-mater-physics-tests
        PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME parallel-mater-physics-tests COMMAND parallel-mater-physics-tests)
    set_tests_properties(parallel-mater-physics-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-smoke-tests tests/smoke_tests.cu)
    target_link_libraries(parallel-mater-smoke-tests PRIVATE ParallelMater::physics)
    target_compile_options(parallel-mater-smoke-tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info>)
    set_target_properties(parallel-mater-smoke-tests
        PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME parallel-mater-smoke-tests COMMAND parallel-mater-smoke-tests)
    set_tests_properties(parallel-mater-smoke-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-triangle-contact-tests tests/triangle_contact_tests.cu)
    target_compile_options(parallel-mater-triangle-contact-tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info>)
    add_test(NAME parallel-mater-triangle-contact-host
        COMMAND parallel-mater-triangle-contact-tests --host-only)
    add_test(NAME parallel-mater-triangle-contact-gpu COMMAND parallel-mater-triangle-contact-tests)
    set_tests_properties(parallel-mater-triangle-contact-gpu PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(
        parallel-mater-hybrid-tests
        tests/hybrid_tests.cu
        apps/water_lab/geodesic_sphere.cpp
        apps/water_lab/fluid_surface.cu
        apps/water_lab/fluid_visuals.cu
        apps/water_lab/soft_body.cu
        apps/water_lab/water_kernels.cu
        apps/water_lab/hybrid_kernels.cu)
    target_link_libraries(parallel-mater-hybrid-tests PRIVATE ParallelMater::geometry)
    target_compile_options(
        parallel-mater-hybrid-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    target_compile_definitions(parallel-mater-hybrid-tests PRIVATE
        PARALLEL_MATER_SOFT_BODY_TEST_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
    set_target_properties(parallel-mater-hybrid-tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME parallel-mater-hybrid-tests COMMAND parallel-mater-hybrid-tests)
    set_tests_properties(parallel-mater-hybrid-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(
        parallel-mater-course-tests
        tests/course_tests.cu
        apps/water_lab/geodesic_sphere.cpp
        apps/water_lab/soft_body.cu
        apps/water_lab/hybrid_kernels.cu)
    target_link_libraries(parallel-mater-course-tests PRIVATE ParallelMater::geometry)
    target_compile_options(
        parallel-mater-course-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(parallel-mater-course-tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME parallel-mater-course-host COMMAND parallel-mater-course-tests --host-only)
    add_test(NAME parallel-mater-course-gpu COMMAND parallel-mater-course-tests)
    set_tests_properties(parallel-mater-course-gpu PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-fluid-visual-tests
        tests/fluid_visual_tests.cu apps/water_lab/fluid_surface.cu apps/water_lab/fluid_visuals.cu)
    target_link_libraries(parallel-mater-fluid-visual-tests PRIVATE ParallelMater::geometry)
    target_compile_options(parallel-mater-fluid-visual-tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(parallel-mater-fluid-visual-tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME parallel-mater-fluid-visual-tests COMMAND parallel-mater-fluid-visual-tests)
    set_tests_properties(parallel-mater-fluid-visual-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-soft-body-tests
        tests/soft_body_tests.cu apps/water_lab/soft_body.cu)
    target_link_libraries(parallel-mater-soft-body-tests PRIVATE ParallelMater::geometry)
    target_compile_definitions(parallel-mater-soft-body-tests PRIVATE
        PARALLEL_MATER_SOFT_BODY_TEST_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
    target_compile_options(parallel-mater-soft-body-tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(parallel-mater-soft-body-tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME parallel-mater-soft-body-tests COMMAND parallel-mater-soft-body-tests)
    set_tests_properties(parallel-mater-soft-body-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-cloth-tests
        tests/cloth_tests.cu apps/water_lab/cloth.cpp
        apps/water_lab/geodesic_sphere.cpp apps/water_lab/soft_body.cu)
    target_link_libraries(parallel-mater-cloth-tests PRIVATE ParallelMater::geometry)
    target_compile_options(parallel-mater-cloth-tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(parallel-mater-cloth-tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME parallel-mater-cloth-tests COMMAND parallel-mater-cloth-tests)
    set_tests_properties(parallel-mater-cloth-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-simulation-gallery-tests
        tests/simulation_gallery_tests.cu
        apps/water_lab/simulation_gallery.cpp
        apps/water_lab/cloth.cpp
        apps/water_lab/geodesic_sphere.cpp
        apps/water_lab/soft_body.cu
        apps/water_lab/hybrid_kernels.cu)
    target_link_libraries(parallel-mater-simulation-gallery-tests PRIVATE ParallelMater::geometry)
    target_include_directories(parallel-mater-simulation-gallery-tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/gallery)
    target_compile_definitions(parallel-mater-simulation-gallery-tests PRIVATE
        PARALLEL_MATER_SOFT_BODY_TEST_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
    target_compile_options(parallel-mater-simulation-gallery-tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
    set_target_properties(parallel-mater-simulation-gallery-tests
        PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_test(NAME parallel-mater-simulation-gallery-tests
        COMMAND parallel-mater-simulation-gallery-tests)
    set_tests_properties(parallel-mater-simulation-gallery-tests PROPERTIES SKIP_RETURN_CODE 77)

    if(TARGET parallel-mater-contact-experiment)
        add_test(
            NAME parallel-mater-contact-stability
            COMMAND parallel-mater-contact-experiment
                --runs 1
                --iterations 1
                --material default
                --output /tmp/parallel-mater-contact-experiment/ctest)
        set_tests_properties(parallel-mater-contact-stability PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

if(PARALLEL_MATER_BUILD_LAB)
    find_package(OpenGL QUIET)
    find_package(glfw3 3.3 QUIET)
    if(OpenGL_FOUND AND TARGET glfw)
        add_library(
            parallel-mater-native-simulations
            apps/water_lab/cloth.cpp
            apps/water_lab/geodesic_sphere.cpp
            apps/water_lab/simulation_gallery.cpp
            apps/water_lab/soft_body.cu
            apps/water_lab/hybrid_kernels.cu)
        target_link_libraries(parallel-mater-native-simulations PUBLIC
            ParallelMater::geometry parallel-mater-example-recipes)
        target_include_directories(parallel-mater-native-simulations PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/apps/water_lab
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/gallery)
        target_compile_options(parallel-mater-native-simulations PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
        set_target_properties(parallel-mater-native-simulations PROPERTIES
            POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION OFF)

        add_library(
            parallel-mater-gallery-runtime
            apps/water_lab/fluid_surface.cu
            apps/water_lab/fluid_visuals.cu
            apps/water_lab/water_kernels.cu)
        target_link_libraries(
            parallel-mater-gallery-runtime PUBLIC parallel-mater-example-gallery
            parallel-mater-native-simulations)
        target_include_directories(parallel-mater-gallery-runtime
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/apps/water_lab)
        target_compile_definitions(parallel-mater-gallery-runtime PRIVATE
            PARALLEL_MATER_SOFT_BODY_ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
        target_compile_options(
            parallel-mater-gallery-runtime
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
                $<$<COMPILE_LANGUAGE:CUDA>:-O3;--generate-line-info;--expt-relaxed-constexpr>)
        set_target_properties(
            parallel-mater-gallery-runtime PROPERTIES
                POSITION_INDEPENDENT_CODE ON
                CUDA_SEPARABLE_COMPILATION OFF)

        add_executable(parallel-mater-lab
            apps/water_lab/main.cpp apps/water_lab/campaign.cpp)
        target_link_libraries(
            parallel-mater-lab PRIVATE parallel-mater-gallery-runtime
            glfw OpenGL::GL)
        target_compile_definitions(parallel-mater-lab PRIVATE
            PARALLEL_MATER_SOFT_BODY_ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/softbody/checker_cylinder.msb")
        target_compile_options(parallel-mater-lab PRIVATE
            -Wall -Wextra -Wpedantic)
    else()
        message(STATUS "parallel-mater-lab disabled: OpenGL and glfw3 3.3 are required")
    endif()
endif()

set(parallel_mater_install_targets parallel-mater-geometry parallel-mater-physics)
install(TARGETS ${parallel_mater_install_targets} EXPORT ParallelMaterTargets)
install(FILES
    include/parallel_mater/cloth.hpp
    include/parallel_mater/coupling.hpp
    include/parallel_mater/fluid.hpp
    include/parallel_mater/frame.hpp
    include/parallel_mater/geometry.hpp
    include/parallel_mater/physics.hpp
    include/parallel_mater/rigid_body.hpp
    include/parallel_mater/rope.hpp
    include/parallel_mater/smoke.hpp
    include/parallel_mater/soft_body.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/parallel_mater)
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
