// SPDX-License-Identifier: MIT
#include "../apps/water_lab/campaign.hpp"

#include <cstdio>

namespace {

int failures{};

void expect(bool condition,const char* message)
{
    if (condition) return;
    std::fprintf(stderr,"FAIL: %s\n",message);
    ++failures;
}

} // namespace

int main()
{
    using parallel_mater::examples::SimulationRecipe;

    waterlab::ObjectiveMetrics metrics;
    metrics.painted_fraction=0.42F;
    const auto paint=waterlab::evaluate_objective(SimulationRecipe::water,metrics);
    expect(paint.normalized==0.42F && !paint.completed,
        "paint objective must expose continuous progress");

    metrics={};
    metrics.rope_turns=3.0F;
    expect(waterlab::evaluate_objective(SimulationRecipe::rope,metrics).completed,
        "rope objective must complete at three turns");

    metrics={};
    metrics.treasure_caught=true;
    metrics.treasure_lift_progress=1.0F;
    expect(waterlab::evaluate_objective(
        SimulationRecipe::water_rope,metrics).completed,
        "fishing objective must complete after lifting the caught treasure");

    waterlab::GalleryProgression progression(SimulationRecipe::cloth);
    metrics={};
    metrics.broken_connections=1U;
    waterlab::ObjectiveProgress progress;
    for (int frame=0;frame<90;++frame) progress=progression.update(metrics);
    expect(progress.advanced && progression.current()==SimulationRecipe::soft_body,
        "gallery progression must advance after the completion hold");

    if (failures!=0) return 1;
    std::puts("gallery campaign tests passed");
    return 0;
}
