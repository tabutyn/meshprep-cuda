// SPDX-License-Identifier: MIT
#include "water_lab.hpp"

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::uint32_t width{960};
    std::uint32_t height{720};
    std::uint32_t substeps{8};
    std::uint32_t profile_frames{};
};

struct Interaction {
    bool clicked{};
    bool reset{};
    bool paused{};
    double cursor_x{};
    double cursor_y{};
    std::uint32_t substeps{8};
};

void cuda_check(cudaError_t error, const char* operation)
{
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    std::exit(2);
}

bool parse_u32(std::string_view text, std::uint32_t& value)
{
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool parse_options(int argc, char** argv, Options& options)
{
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--help") return false;
        if (index + 1 >= argc) return false;
        const std::string_view value = argv[++index];
        if (argument == "--width") {
            if (!parse_u32(value, options.width) || options.width < 64) return false;
        } else if (argument == "--height") {
            if (!parse_u32(value, options.height) || options.height < 64) return false;
        } else if (argument == "--substeps") {
            if (!parse_u32(value, options.substeps) || options.substeps == 0 || options.substeps > 64) {
                return false;
            }
        } else if (argument == "--profile") {
            if (!parse_u32(value, options.profile_frames) || options.profile_frames == 0) return false;
        } else {
            return false;
        }
    }
    return true;
}

void usage(const char* executable)
{
    std::fprintf(
        stderr,
        "usage: %s [--width N] [--height N] [--substeps N] [--profile FRAMES]\n",
        executable);
}

float3 subtract(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float3 normalize(float3 value)
{
    const float inverse = 1.0F /
        std::max(std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z), 1.0e-20F);
    return make_float3(value.x * inverse, value.y * inverse, value.z * inverse);
}

float percentile(std::vector<float> values, float fraction)
{
    std::sort(values.begin(), values.end());
    const float position = fraction * static_cast<float>(values.size() - 1U);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1U, values.size() - 1U);
    const float blend = position - static_cast<float>(lower);
    return values[lower] * (1.0F - blend) + values[upper] * blend;
}

struct BuildTimer {
    BuildTimer()
    {
        cuda_check(cudaEventCreate(&begin), "create hierarchy timer");
        cuda_check(cudaEventCreate(&end), "create hierarchy timer");
    }
    ~BuildTimer()
    {
        cudaEventDestroy(end);
        cudaEventDestroy(begin);
    }

    float build(
        meshprep::DeviceMeshView mesh,
        meshprep::Workspace& workspace,
        meshprep::Hierarchy& hierarchy)
    {
        cuda_check(cudaEventRecord(begin), "record hierarchy begin");
        const meshprep::Status status =
            meshprep::build_hierarchy(mesh, {}, workspace, hierarchy);
        if (!status) {
            std::fprintf(stderr, "hierarchy failed: %s\n", status.message);
            std::exit(2);
        }
        cuda_check(cudaEventRecord(end), "record hierarchy end");
        cuda_check(cudaEventSynchronize(end), "synchronize hierarchy timer");
        float milliseconds = 0.0F;
        cuda_check(cudaEventElapsedTime(&milliseconds, begin, end), "measure hierarchy");
        return milliseconds;
    }

    cudaEvent_t begin{};
    cudaEvent_t end{};
};

void print_profile(
    const char* stage,
    const std::vector<float>& values,
    std::uint32_t triangles = 0)
{
    const float median = percentile(values, 0.5F);
    std::printf(
        "PROFILE,%s,samples=%zu,median_ms=%.4f,p5_ms=%.4f,p95_ms=%.4f",
        stage,
        values.size(),
        median,
        percentile(values, 0.05F),
        percentile(values, 0.95F));
    if (triangles > 0) {
        std::printf(
            ",mtriangles_per_second=%.3f",
            static_cast<float>(triangles) / (median * 1000.0F));
    }
    std::putchar('\n');
}

int run_profile(const Options& options, waterlab::HostSurfaceMesh host_mesh)
{
    waterlab::WaterSurface surface(host_mesh);
    meshprep::Workspace workspace;
    meshprep::Hierarchy hierarchy;
    waterlab::RayTracer raytracer;
    waterlab::Camera camera;
    waterlab::PhysicsOptions physics;
    BuildTimer build_timer;

    const std::uint32_t warmups = 10;
    const std::uint32_t total_frames = warmups + options.profile_frames;
    std::vector<float> physics_times;
    std::vector<float> hierarchy_times;
    std::vector<float> raytrace_times;
    std::vector<float> wall_times;
    physics_times.reserve(options.profile_frames);
    hierarchy_times.reserve(options.profile_frames);
    raytrace_times.reserve(options.profile_frames);
    wall_times.reserve(options.profile_frames);

    for (std::uint32_t frame = 0; frame < total_frames; ++frame) {
        if (frame == 2) {
            surface.apply_impulse(
                make_float3(0.0F, 0.0F, 1.0F), make_float3(0.22F, 0.04F, -1.0F), 0.24F, 2.8F);
        }
        const auto wall_begin = std::chrono::steady_clock::now();
        const float physics_ms = surface.step(1.0F / 60.0F, options.substeps, physics);
        const float hierarchy_ms = build_timer.build(surface.mesh_view(), workspace, hierarchy);
        const float raytrace_ms = raytracer.render(
            surface.mesh_view(),
            hierarchy,
            camera,
            options.width,
            options.height,
            static_cast<float>(frame) / 60.0F);
        const float wall_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - wall_begin).count();
        if (frame >= warmups) {
            physics_times.push_back(physics_ms);
            hierarchy_times.push_back(hierarchy_ms);
            raytrace_times.push_back(raytrace_ms);
            wall_times.push_back(wall_ms);
        }
    }

    std::printf(
        "GEOMETRY,vertices=%u,triangles=%u,edges=%u,substeps=%u,resolution=%ux%u,nodes=%u\n",
        surface.vertex_count(),
        surface.triangle_count(),
        surface.edge_count(),
        options.substeps,
        options.width,
        options.height,
        hierarchy.statistics().node_count);
    print_profile("physics", physics_times);
    print_profile("hierarchy", hierarchy_times, surface.triangle_count());
    print_profile("raytrace", raytrace_times);
    print_profile("frame_wall", wall_times);
    const float median_frame = percentile(wall_times, 0.5F);
    std::printf(
        "FRAME_BUDGET,target_ms=16.6667,median_ms=%.4f,headroom_ms=%.4f,status=%s\n",
        median_frame,
        16.6667F - median_frame,
        median_frame <= 16.6667F ? "PASS" : "OVER");
    return 0;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS) return;
    auto* interaction = static_cast<Interaction*>(glfwGetWindowUserPointer(window));
    glfwGetCursorPos(window, &interaction->cursor_x, &interaction->cursor_y);
    interaction->clicked = true;
}

void key_callback(GLFWwindow* window, int key, int, int action, int)
{
    if (action != GLFW_PRESS) return;
    auto* interaction = static_cast<Interaction*>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, GLFW_TRUE);
    else if (key == GLFW_KEY_SPACE) interaction->paused = !interaction->paused;
    else if (key == GLFW_KEY_R) interaction->reset = true;
    else if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) {
        interaction->substeps = std::min(32U, interaction->substeps + 1U);
    } else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) {
        interaction->substeps = std::max(1U, interaction->substeps - 1U);
    }
}

int run_interactive(const Options& options, waterlab::HostSurfaceMesh host_mesh)
{
    if (glfwInit() != GLFW_TRUE) throw std::runtime_error("GLFW initialization failed");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    GLFWwindow* window = glfwCreateWindow(
        static_cast<int>(options.width),
        static_cast<int>(options.height),
        "meshprep-cuda water lab",
        nullptr,
        nullptr);
    if (window == nullptr) {
        glfwTerminate();
        throw std::runtime_error("window creation failed");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    Interaction interaction{.substeps = options.substeps};
    glfwSetWindowUserPointer(window, &interaction);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetKeyCallback(window, key_callback);

    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glEnable(GL_TEXTURE_2D);

    waterlab::WaterSurface surface(host_mesh);
    meshprep::Workspace workspace;
    meshprep::Hierarchy hierarchy;
    waterlab::RayTracer raytracer;
    waterlab::Camera camera;
    waterlab::PhysicsOptions physics;
    BuildTimer build_timer;
    float physics_ms = 0.0F;
    float hierarchy_ms = build_timer.build(surface.mesh_view(), workspace, hierarchy);
    float raytrace_ms = 0.0F;
    std::uint64_t frame = 0;
    auto title_update = std::chrono::steady_clock::now();
    const auto application_start = title_update;
    int texture_width = 0;
    int texture_height = 0;

    std::printf(
        "Water lab ready: %u vertices, %u triangles, %u graph edges.\n"
        "Controls: left click impulse, Space pause, R reset, +/- physics substeps, Esc quit.\n",
        surface.vertex_count(),
        surface.triangle_count(),
        surface.edge_count());

    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        glfwPollEvents();
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if (width <= 0 || height <= 0) continue;
        width = std::max(64, width);
        height = std::max(64, height);
        if (interaction.reset) {
            surface.reset();
            interaction.reset = false;
        }
        if (interaction.clicked) {
            int window_width = 0;
            int window_height = 0;
            glfwGetWindowSize(window, &window_width, &window_height);
            const auto pixel_x = static_cast<std::uint32_t>(std::clamp(
                interaction.cursor_x * width / std::max(window_width, 1), 0.0, static_cast<double>(width - 1)));
            const auto pixel_y = static_cast<std::uint32_t>(std::clamp(
                interaction.cursor_y * height / std::max(window_height, 1), 0.0, static_cast<double>(height - 1)));
            const waterlab::PickResult pick = raytracer.pick(
                surface.mesh_view(), hierarchy, camera, pixel_x, pixel_y, width, height);
            if (pick.hit) {
                surface.apply_impulse(
                    pick.position, normalize(subtract(pick.position, camera.eye)), 0.24F, 3.2F);
            }
            interaction.clicked = false;
        }
        if (!interaction.paused) {
            physics_ms = surface.step(1.0F / 60.0F, interaction.substeps, physics);
        }
        hierarchy_ms = build_timer.build(surface.mesh_view(), workspace, hierarchy);
        const float elapsed = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - application_start).count();
        raytrace_ms = raytracer.render(
            surface.mesh_view(), hierarchy, camera, width, height, elapsed);

        glViewport(0, 0, width, height);
        glBindTexture(GL_TEXTURE_2D, texture);
        if (texture_width != width || texture_height != height) {
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA8,
                width,
                height,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                raytracer.pixels());
            texture_width = width;
            texture_height = height;
        } else {
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                width,
                height,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                raytracer.pixels());
        }
        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0F, 1.0F); glVertex2f(-1.0F, -1.0F);
        glTexCoord2f(1.0F, 1.0F); glVertex2f(1.0F, -1.0F);
        glTexCoord2f(1.0F, 0.0F); glVertex2f(1.0F, 1.0F);
        glTexCoord2f(0.0F, 0.0F); glVertex2f(-1.0F, 1.0F);
        glEnd();
        glfwSwapBuffers(window);
        ++frame;

        const auto now = std::chrono::steady_clock::now();
        if (now - title_update > std::chrono::milliseconds(350)) {
            char title[256];
            std::snprintf(
                title,
                sizeof(title),
                "Water Lab | physics %.2f ms (%u steps) | hierarchy %.2f ms | raytrace %.2f ms | click to splash",
                physics_ms,
                interaction.substeps,
                hierarchy_ms,
                raytrace_ms);
            glfwSetWindowTitle(window, title);
            title_update = now;
        }
    }

    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage(argv[0]);
        return argc > 1 && std::string_view(argv[1]) == "--help" ? 0 : 1;
    }
    try {
        waterlab::HostSurfaceMesh mesh = waterlab::make_geodesic_sphere(45, 1.0F);
        if (options.profile_frames > 0) return run_profile(options, std::move(mesh));
        return run_interactive(options, std::move(mesh));
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "water lab failed: %s\n", exception.what());
        return 1;
    }
}
