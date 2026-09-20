// SPDX-License-Identifier: MIT
#include <parallel_mater/cloth.hpp>
#include <parallel_mater/fluid.hpp>
#include <parallel_mater/rigid_body.hpp>
#include <parallel_mater/rope.hpp>

#include <cstdio>
#include <vector>

int main()
{
    using namespace parallel_mater;
    using namespace parallel_mater::physics;

    std::vector<FluidParticle> initial_particles;
    for (std::uint32_t z = 0U; z < 4U; ++z) {
        for (std::uint32_t y = 0U; y < 4U; ++y) {
            for (std::uint32_t x = 0U; x < 4U; ++x) {
                initial_particles.push_back({make_float3(
                    0.05F * static_cast<float>(x),
                    1.0F + 0.05F * static_cast<float>(y),
                    0.05F * static_cast<float>(z)), {}});
            }
        }
    }

    Fluid fluid;
    Cloth cloth;
    Rope rope;
    RigidBody sphere;
    Status status = fluid.initialize(initial_particles);
    if (status) status = cloth.initialize();
    if (status) status = rope.initialize();
    if (status) status = sphere.initialize();

    const FrameOptions frame{1.0F / 60.0F, 4U,
        make_float3(0.0F, -9.81F, 0.0F)};
    if (status) status = fluid.advance(frame);
    if (status) status = cloth.advance(frame);
    if (status) status = rope.advance(frame);
    if (status) status = sphere.advance(frame);

    if (!status) {
        std::fprintf(stderr, "%s: %s\n",
            status_code_name(status.code), status.message);
        return 1;
    }
    std::printf("fluid=%u cloth=%u rope=%u sphere_y=%.3f\n",
        fluid.statistics().particle_count, cloth.nodes().node_count,
        rope.nodes().node_count, sphere.state().position.y);
    return 0;
}
