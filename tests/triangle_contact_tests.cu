// SPDX-License-Identifier: MIT
#include "../apps/water_lab/triangle_contact.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
struct Query { float3 point, a, b, c; };
struct D3 { double x, y, z; };
D3 wide(float3 p) { return {p.x, p.y, p.z}; }
D3 operator+(D3 a, D3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
D3 operator-(D3 a, D3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
D3 operator*(D3 a, double s) { return {a.x*s, a.y*s, a.z*s}; }
double dot(D3 a, D3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
D3 cross(D3 a, D3 b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
double squared(D3 a) { return dot(a, a); }

// Independent reference: project onto the triangle plane if inside, otherwise
// minimize distance over all three closed line segments. No Voronoi branches.
D3 reference_point(const Query& q)
{
    const D3 p=wide(q.point), a=wide(q.a), b=wide(q.b), c=wide(q.c);
    const D3 n=cross(b-a,c-a);
    const double nn=squared(n);
    if (nn > 1.0e-24) {
        const D3 projected=p-n*(dot(p-a,n)/nn);
        if (dot(cross(b-a,projected-a),n) >= 0.0 &&
            dot(cross(c-b,projected-b),n) >= 0.0 &&
            dot(cross(a-c,projected-c),n) >= 0.0) return projected;
    }
    D3 result=a;
    double distance=squared(p-a);
    const D3 vertices[]{a,b,c};
    for (unsigned i=0; i<3; ++i) {
        const D3 start=vertices[i], edge=vertices[(i+1)%3]-start;
        const double t=squared(edge)>0.0 ? std::clamp(dot(p-start,edge)/squared(edge),0.0,1.0) : 0.0;
        const D3 candidate=start+edge*t;
        if (squared(p-candidate)<distance) { result=candidate; distance=squared(p-candidate); }
    }
    return result;
}

std::vector<Query> fixtures()
{
    // Hex literals retain captured float bits. Inputs precede the force spike:
    // capture-20260912-171242-frame-645610, frame 645514, particle 2158, triangle 1996;
    // frame 645528, particle 1564, triangle 1993. No /tmp files needed at test time.
    const Query captured[]{
        {{0x1.83ba76p-3F,0x1.25d7aep-4F,-0x1.057fdap-1F},
         {0x1.bcda2ap-2F,0x1.c8b6cep-8F,-0x1.495feep-2F},
         {0x1.c07052p-2F,0x1.dede32p-14F,-0x1.595ep-2F},
         {0x1.bff6cep-2F,0x1.777faap-4F,-0x1.4c74aap-2F}},
        {{0x1.5216dp-4F,0x1.c91a8ep-4F,-0x1.e00eaap-2F},
         {0x1.6bd082p-2F,0x1.b3cdbep-5F,-0x1.43050ep-2F},
         {0x1.6bccbep-2F,0x1.72443p-4F,-0x1.477baep-2F},
         {0x1.6a96ecp-2F,0x1.48e83cp-3F,-0x1.420b3cp-2F}}};
    std::vector<Query> queries;
    const auto permute=[&](Query q) {
        const float3 vertices[]{q.a,q.b,q.c};
        std::array<unsigned,3> order{0,1,2};
        do { queries.push_back({q.point,vertices[order[0]],vertices[order[1]],vertices[order[2]]}); }
        while (std::next_permutation(order.begin(),order.end()));
    };
    for (const Query q:captured) permute(q);
    // All seven regions: vertices A/B/C, edges AB/AC/BC, and face interior.
    for (const float3 p:std::array<float3,7>{{{-1,-1,0.2F},{2,-0.2F,0.2F},
             {-0.2F,2,0.2F},{0.3F,-0.2F,0.2F},{-0.2F,0.3F,0.2F},
             {0.75F,0.75F,0.2F},{0.2F,0.3F,0.2F}}})
        permute({p,{0,0,0},{1,0,0},{0,1,0}});
    // Thin captured triangles with many nearby queries catch relabeling and
    // force-extrapolation regressions beyond the two observed particles.
    std::mt19937 random(0xBC2026U);
    const auto offset=[&]() { return 0.6F*(static_cast<float>(random()>>8U)/16777216.0F-0.5F); };
    for (const Query q:captured) {
        for (unsigned i=0;i<128;++i)
            permute({{q.a.x+offset(),q.a.y+offset(),q.a.z+offset()},q.a,q.b,q.c});
    }
    return queries;
}

void verify(const std::vector<Query>& queries, const std::vector<float3>& weights, const char* path)
{
    for (std::size_t i=0;i<queries.size();++i) {
        const auto& q=queries[i]; const float3 w=weights[i];
        const double sum=static_cast<double>(w.x)+w.y+w.z;
        const D3 point=wide(q.a)*w.x+wide(q.b)*w.y+wide(q.c)*w.z;
        const double error=std::sqrt(squared(point-reference_point(q)));
        const bool simplex=std::isfinite(sum) && w.x>=-1e-5F && w.y>=-1e-5F && w.z>=-1e-5F &&
            w.x<=1.00001F && w.y<=1.00001F && w.z<=1.00001F && std::abs(sum-1.0)<1e-5;
        // These same scalar weights distribute every vector reaction. Checking
        // the L1 sum rules out hidden opposite forces even when their sum is 1.
        const double reaction_l1=55.0*(std::abs(w.x)+std::abs(w.y)+std::abs(w.z));
        if (!simplex || !std::isfinite(error) || error>2e-5 || reaction_l1>55.001) {
            std::fprintf(stderr,"%s fixture %zu: weights=(%.9g,%.9g,%.9g), closest error=%.9g, reaction L1=%.9g\n",
                path,i,w.x,w.y,w.z,error,reaction_l1);
            throw std::runtime_error("invalid closest-point or amplified contact reaction");
        }
    }
    std::printf("PASS triangle contact %s: %zu queries, all regions and vertex permutations\n",path,queries.size());
}

__global__ void evaluate(const Query* queries, float3* weights, unsigned count)
{
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count) {
        const Query q=queries[i];
        weights[i]=waterlab::detail::closest_triangle_barycentric(q.point,q.a,q.b,q.c);
    }
}

void check(cudaError_t status)
{
    if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace

int main(int argc,char** argv)
{
    try {
        const auto queries=fixtures();
        std::vector<float3> weights(queries.size());
        for (std::size_t i=0;i<queries.size();++i) {
            const Query q=queries[i];
            weights[i]=waterlab::detail::closest_triangle_barycentric(q.point,q.a,q.b,q.c);
        }
        verify(queries,weights,"host");
        if (argc==2 && std::string_view(argv[1])=="--host-only") return 0;
        int count=0;
        if (cudaGetDeviceCount(&count)!=cudaSuccess || count==0) {
            std::puts("SKIP triangle contact GPU: no CUDA device"); return 77;
        }
        Query* device_queries=nullptr; float3* device_weights=nullptr;
        check(cudaMalloc(&device_queries,queries.size()*sizeof(Query)));
        check(cudaMalloc(&device_weights,weights.size()*sizeof(float3)));
        check(cudaMemcpy(device_queries,queries.data(),queries.size()*sizeof(Query),cudaMemcpyHostToDevice));
        evaluate<<<(queries.size()+127)/128,128>>>(device_queries,device_weights,static_cast<unsigned>(queries.size()));
        check(cudaGetLastError());
        check(cudaMemcpy(weights.data(),device_weights,weights.size()*sizeof(float3),cudaMemcpyDeviceToHost));
        check(cudaFree(device_weights)); check(cudaFree(device_queries));
        verify(queries,weights,"CUDA");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"FAIL triangle contact: %s\n",error.what()); return 1;
    }
}
