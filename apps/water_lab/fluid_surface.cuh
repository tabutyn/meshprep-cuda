// SPDX-License-Identifier: MIT
#pragma once

#include "fluid_surface.hpp"
#include <vector_functions.h>
#include <cmath>

namespace waterlab {
namespace detail {

__host__ __device__ inline float surface_lerp(float a, float b, float t)
{
    return a + (b-a)*t;
}

// Returns the exact gradient of the trilinear field, followed by its value.
// This scalar is not a signed-distance bound: ray traversal must bracket crossings.
__host__ __device__ inline float4 surface_value_gradient(FluidSurfaceView view, float3 p)
{
    if (!view.grid || !view.values || !std::isfinite(p.x) || !std::isfinite(p.y) ||
        !std::isfinite(p.z)) return make_float4(0, 0, 0, 1.0e20F);
    const FluidSurfaceGrid g = *view.grid;
    if (g.dimensions.x < 2U || g.dimensions.y < 2U || g.dimensions.z < 2U ||
        !(g.cell_size.x > 0) || !(g.cell_size.y > 0) || !(g.cell_size.z > 0)) {
        return make_float4(0, 0, 0, 1.0e20F);
    }
    const float3 maximum = make_float3(g.minimum.x+g.cell_size.x*(g.dimensions.x-1U),
        g.minimum.y+g.cell_size.y*(g.dimensions.y-1U),
        g.minimum.z+g.cell_size.z*(g.dimensions.z-1U));
    const float3 d = make_float3(p.x-fminf(maximum.x, fmaxf(g.minimum.x, p.x)),
        p.y-fminf(maximum.y, fmaxf(g.minimum.y, p.y)),
        p.z-fminf(maximum.z, fmaxf(g.minimum.z, p.z)));
    const float outside = sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);
    // Cell-boundary ray arithmetic can land a few ulps beyond the grid. Clamp
    // that tiny band to the boundary value instead of inventing an air jump.
    const float boundary_slop = 1.0e-4F*fminf(g.cell_size.x,fminf(g.cell_size.y,g.cell_size.z));
    if (outside > boundary_slop) {
        return make_float4(d.x/outside, d.y/outside, d.z/outside, g.support_radius+outside);
    }
    const float3 q = make_float3((p.x-g.minimum.x)/g.cell_size.x,
        (p.y-g.minimum.y)/g.cell_size.y, (p.z-g.minimum.z)/g.cell_size.z);
    const auto ix = static_cast<unsigned>(fminf(static_cast<float>(g.dimensions.x-2U), fmaxf(0, floorf(q.x))));
    const auto iy = static_cast<unsigned>(fminf(static_cast<float>(g.dimensions.y-2U), fmaxf(0, floorf(q.y))));
    const auto iz = static_cast<unsigned>(fminf(static_cast<float>(g.dimensions.z-2U), fmaxf(0, floorf(q.z))));
    const float x = fminf(1, fmaxf(0, q.x-ix));
    const float y = fminf(1, fmaxf(0, q.y-iy));
    const float z = fminf(1, fmaxf(0, q.z-iz));
    const std::size_t row = g.dimensions.x, layer = row*g.dimensions.y;
    const std::size_t i = iz*layer + iy*row + ix;
    const float a = view.values[i], b = view.values[i+1], c = view.values[i+row], d0 = view.values[i+row+1];
    const float e = view.values[i+layer], f = view.values[i+layer+1];
    const float h = view.values[i+layer+row], k = view.values[i+layer+row+1];
    const float y0 = surface_lerp(surface_lerp(a,b,x), surface_lerp(c,d0,x), y);
    const float y1 = surface_lerp(surface_lerp(e,f,x), surface_lerp(h,k,x), y);
    const float dx = surface_lerp(surface_lerp(b-a,d0-c,y), surface_lerp(f-e,k-h,y),z)/g.cell_size.x;
    const float dy = surface_lerp(surface_lerp(c,d0,x)-surface_lerp(a,b,x),
        surface_lerp(h,k,x)-surface_lerp(e,f,x),z)/g.cell_size.y;
    return make_float4(dx, dy, (y1-y0)/g.cell_size.z, surface_lerp(y0,y1,z));
}

} // namespace detail

__host__ __device__ inline float surface_sample(FluidSurfaceView view, float3 point)
{
    return detail::surface_value_gradient(view, point).w;
}

__host__ __device__ inline float3 surface_gradient(FluidSurfaceView view, float3 point)
{
    const float4 result = detail::surface_value_gradient(view, point);
    return make_float3(result.x, result.y, result.z);
}

// First root on [0,1] of the cubic interpolating samples at 0,1/3,2/3,1.
// Derivative extrema split monotone intervals, including tangent roots and
// two crossings whose segment endpoints have the same sign.
__host__ __device__ inline bool surface_cubic_first_root(
    float y0, float y1, float y2, float y3, float& root)
{
    if (!std::isfinite(y0) || !std::isfinite(y1) || !std::isfinite(y2) || !std::isfinite(y3)) return false;
    const float scale=fmaxf(fmaxf(fabsf(y0),fabsf(y1)),fmaxf(fabsf(y2),fabsf(y3)));
    if (scale==0.0F) { root=0.0F; return true; }
    y0/=scale; y1/=scale; y2/=scale; y3/=scale;
    const float third=y3-3*y2+3*y1-y0, second=y2-2*y1+y0;
    const float a=4.5F*third, b=4.5F*(second-third);
    const float c=3*((y1-y0)-third/6-0.5F*(second-third)), d=y0;
    const float coefficient_scale=fmaxf(1,fmaxf(fabsf(a),fmaxf(fabsf(b),fabsf(c))));
    float cuts[4]{0,1};
    int count=2;
    if (fabsf(a)>1.0e-7F*coefficient_scale) {
        const float discriminant=b*b-3*a*c;
        if (discriminant>=0) {
            const float q=-b-copysignf(sqrtf(discriminant),b);
            const float e0=q/(3*a), e1=q!=0 ? c/q : -b/(3*a);
            if (e0>0 && e0<1) cuts[count++]=e0;
            if (e1>0 && e1<1) cuts[count++]=e1;
        }
    } else if (fabsf(b)>1.0e-7F*coefficient_scale) {
        const float e=-c/(2*b);
        if (e>0 && e<1) cuts[count++]=e;
    }
    for (int i=1; i<count; ++i) {
        const float value=cuts[i];
        int j=i;
        while (j>0 && cuts[j-1]>value) { cuts[j]=cuts[j-1]; --j; }
        cuts[j]=value;
    }
    for (int interval=0; interval+1<count; ++interval) {
        float left=cuts[interval], right=cuts[interval+1];
        float lv=((a*left+b)*left+c)*left+d;
        const float rv=((a*right+b)*right+c)*right+d;
        if (fabsf(lv)<=1.0e-6F) { root=left; return true; }
        if (fabsf(rv)<=1.0e-6F) { root=right; return true; }
        if ((lv<0)==(rv<0)) continue;
        for (int iteration=0; iteration<22; ++iteration) {
            const float middle=0.5F*(left+right);
            const float value=((a*middle+b)*middle+c)*middle+d;
            if ((lv<0)==(value<0)) { left=middle; lv=value; }
            else right=middle;
        }
        root=0.5F*(left+right);
        return true;
    }
    return false;
}

} // namespace waterlab
