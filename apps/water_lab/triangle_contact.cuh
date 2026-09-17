// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime.h>

namespace waterlab::detail {

__host__ __device__ inline float3 triangle_subtract(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float triangle_dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Shared by the simulation and its host/device geometry regression tests.
__host__ __device__ inline float3 closest_triangle_barycentric(
    float3 point, float3 a, float3 b, float3 c)
{
    const float3 ab = triangle_subtract(b, a);
    const float3 ac = triangle_subtract(c, a);
    const float3 ap = triangle_subtract(point, a);
    const float d1 = triangle_dot(ab, ap);
    const float d2 = triangle_dot(ac, ap);
    if (d1 <= 0.0F && d2 <= 0.0F) return make_float3(1.0F, 0.0F, 0.0F);

    const float3 bp = triangle_subtract(point, b);
    const float d3 = triangle_dot(ab, bp);
    const float d4 = triangle_dot(ac, bp);
    if (d3 >= 0.0F && d4 <= d3) return make_float3(0.0F, 1.0F, 0.0F);

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0F && d1 >= 0.0F && d3 <= 0.0F) {
        const float v = d1 / (d1 - d3);
        return make_float3(1.0F - v, v, 0.0F);
    }

    const float3 cp = triangle_subtract(point, c);
    const float d5 = triangle_dot(ab, cp);
    const float d6 = triangle_dot(ac, cp);
    if (d6 >= 0.0F && d5 <= d6) return make_float3(0.0F, 0.0F, 1.0F);

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0F && d2 >= 0.0F && d6 <= 0.0F) {
        const float w = d2 / (d2 - d6);
        return make_float3(1.0F - w, 0.0F, w);
    }

    const float va = d3 * d6 - d5 * d4;
    // BC requires nonnegative projections from both endpoints. Reversing the
    // C-end inequality extrapolates interior weights beyond the triangle.
    if (va <= 0.0F && d4 >= d3 && d5 >= d6) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return make_float3(0.0F, 1.0F - w, w);
    }

    const float denominator = va + vb + vc;
    if (!(fabsf(denominator) > 1.0e-20F)) {
        const float da = triangle_dot(ap, ap);
        const float db = triangle_dot(bp, bp);
        const float dc = triangle_dot(cp, cp);
        if (da <= db && da <= dc) return make_float3(1.0F, 0.0F, 0.0F);
        if (db <= dc) return make_float3(0.0F, 1.0F, 0.0F);
        return make_float3(0.0F, 0.0F, 1.0F);
    }
    const float inverse = 1.0F / denominator;
    const float v = vb * inverse;
    const float w = vc * inverse;
    return make_float3(1.0F - v - w, v, w);
}

} // namespace waterlab::detail
