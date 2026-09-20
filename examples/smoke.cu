// SPDX-License-Identifier: MIT
#include <parallel_mater/smoke.hpp>

#include <cstdio>

int main()
{
    parallel_mater::physics::SmokeOptions options;
    options.particle_count=8'192U;
    options.capacity=8'192U;
    options.emitter_center=make_float3(-1.5F,0.0F,0.0F);
    options.initial_velocity=make_float3(2.5F,0.15F,0.0F);

    parallel_mater::physics::Smoke smoke;
    parallel_mater::Status status=smoke.initialize(options);
    if (!status) {
        std::fprintf(stderr,"smoke initialization failed: %s\n",status.message);
        return 1;
    }

    parallel_mater::physics::SmokeSphereCollider obstacle{
        make_float3(0.0F,0.2F,0.0F),{},0.35F,0.2F};
    parallel_mater::physics::SmokeTimings timings;
    for (std::uint32_t frame=0U;frame<240U && status;++frame)
        status=smoke.step({{},&obstacle,1U},timings);
    if (!status) {
        std::fprintf(stderr,"smoke step failed: %s\n",status.message);
        return 1;
    }

    // `particles()` returns borrowed device pointers suitable for CUDA,
    // graphics interop, or an application renderer. Reacquire after reset.
    const auto particles=smoke.particles();
    const auto statistics=smoke.statistics();
    std::printf("particles=%u frames=%llu advection_ms=%.3f respawns=%llu\n",
        particles.count,static_cast<unsigned long long>(statistics.frame_index),
        timings.integrate_ms,
        static_cast<unsigned long long>(statistics.respawn_count));
    return 0;
}
