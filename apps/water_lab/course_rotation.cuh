// SPDX-License-Identifier: MIT
#pragma once

#include <vector_functions.h>
#include <vector_types.h>

#include <cfloat>
#include <cmath>

namespace waterlab {

// Returns the proper rotation which best maps rest vectors to current vectors
// for A = sum(current * rest^T). Quaternion storage is {x, y, z, w}.
__host__ __device__ inline float4 best_fit_rotation(
    float3 row0, float3 row1, float3 row2)
{
    const float scale = fabsf(row0.x) + fabsf(row0.y) + fabsf(row0.z) +
        fabsf(row1.x) + fabsf(row1.y) + fabsf(row1.z) +
        fabsf(row2.x) + fabsf(row2.y) + fabsf(row2.z);
    if (!(scale > 1.0e-20F) || !(scale <= FLT_MAX)) return make_float4(0, 0, 0, 1);

    // Davenport matrix in [w, x, y, z] order. The antisymmetric terms have
    // this sign because the supplied covariance is current * rest^T.
    float a[4][4]{};
    a[0][0] = row0.x + row1.y + row2.z;
    a[0][1] = a[1][0] = row2.y - row1.z;
    a[0][2] = a[2][0] = row0.z - row2.x;
    a[0][3] = a[3][0] = row1.x - row0.y;
    a[1][1] = row0.x - row1.y - row2.z;
    a[1][2] = a[2][1] = row0.y + row1.x;
    a[1][3] = a[3][1] = row0.z + row2.x;
    a[2][2] = -row0.x + row1.y - row2.z;
    a[2][3] = a[3][2] = row1.z + row2.y;
    a[3][3] = -row0.x - row1.y + row2.z;

    float eigenvectors[4][4]{};
    for (int i = 0; i < 4; ++i) eigenvectors[i][i] = 1.0F;
    constexpr int pairs[6][2]{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    for (int sweep = 0; sweep < 10; ++sweep) {
        for (int pair = 0; pair < 6; ++pair) {
            const int p = pairs[pair][0];
            const int q = pairs[pair][1];
            const float apq = a[p][q];
            if (fabsf(apq) <= 1.0e-7F * scale) continue;
            const float theta = (a[q][q] - a[p][p]) / (2.0F * apq);
            const float t = copysignf(1.0F, theta) /
                (fabsf(theta) + sqrtf(1.0F + theta * theta));
            const float cosine = 1.0F / sqrtf(1.0F + t * t);
            const float sine = t * cosine;
            const float tau = sine / (1.0F + cosine);
            a[p][p] -= t * apq;
            a[q][q] += t * apq;
            a[p][q] = a[q][p] = 0.0F;
            for (int k = 0; k < 4; ++k) {
                if (k == p || k == q) continue;
                const float g = a[k][p];
                const float h = a[k][q];
                a[k][p] = a[p][k] = g - sine * (h + tau * g);
                a[k][q] = a[q][k] = h + sine * (g - tau * h);
            }
            for (int k = 0; k < 4; ++k) {
                const float g = eigenvectors[k][p];
                const float h = eigenvectors[k][q];
                eigenvectors[k][p] = g - sine * (h + tau * g);
                eigenvectors[k][q] = h + sine * (g - tau * h);
            }
        }
    }

    int largest = 0;
    for (int i = 1; i < 4; ++i) if (a[i][i] > a[largest][largest]) largest = i;
    float q[4]{eigenvectors[0][largest], eigenvectors[1][largest],
        eigenvectors[2][largest], eigenvectors[3][largest]};
    const float inverse_length = 1.0F / sqrtf(fmaxf(
        q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3], 1.0e-20F));
    for (float& value : q) value *= inverse_length;
    int canonical = 0;
    for (int i = 1; i < 4; ++i) if (fabsf(q[i]) > fabsf(q[canonical])) canonical = i;
    if (q[canonical] < 0.0F) {
        for (float& value : q) value = -value;
    }
    return make_float4(q[1], q[2], q[3], q[0]);
}

__host__ __device__ inline float3 rotate_course(float4 q, float3 r)
{
    const float3 u = make_float3(q.x, q.y, q.z);
    const float3 t = make_float3(2.0F * (u.y*r.z - u.z*r.y),
        2.0F * (u.z*r.x - u.x*r.z), 2.0F * (u.x*r.y - u.y*r.x));
    return make_float3(r.x + q.w*t.x + u.y*t.z - u.z*t.y,
        r.y + q.w*t.y + u.z*t.x - u.x*t.z,
        r.z + q.w*t.z + u.x*t.y - u.y*t.x);
}

} // namespace waterlab
