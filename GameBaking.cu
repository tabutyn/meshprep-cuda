#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector_types.h>
#include <algorithm>
#include <numeric>
#include <string>
#include <source_location>
#include <execution>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <map>
#include "GameBaking.h"
using std::accumulate;
using std::source_location;
inline source_location current() {return source_location::current();}
using std::string;
using std::sqrt;
using std::max;
using std::min;
using std::vector;
using std::chrono::microseconds;
using std::chrono::duration_cast;
using time_point = std::chrono::high_resolution_clock::time_point;
inline time_point now() { return std::chrono::high_resolution_clock::now(); }

constexpr uint64_t VerticesPerEdge = 2;
constexpr uint64_t VerticesPerTriangle = 3;
constexpr dim3 DefaultBlockDim = dim3(128, 1, 1);
constexpr dim3 DefaultGridDim = dim3(256, 1, 1);

static time_point SyncTime;
static uint64_t FreeBytes = 0;
static uint64_t TotalBytes = 0;

static int Debug = 0;

void PrintMemoryState(const source_location& Where){
    const float FreeGB = static_cast<float>(FreeBytes) / (1024.0 * 1024.0 * 1024.0);
    const float TotalGB = static_cast<float>(TotalBytes) / (1024.0 * 1024.0 * 1024.0);
    if (Debug) printf("CUDA Memory stats at line %u: Used memory: %.2f / %.2f GB\n", Where.line(), TotalGB - FreeGB, TotalGB);}

void PrintBandwidth(uint64_t ProcessedBytes, const source_location& Where){
    float GBytes = ProcessedBytes / (1024.0 * 1024.0 * 1024.0);
    float Seconds = 0.000001f * duration_cast<microseconds>(now() - SyncTime).count();
    SyncTime = now();
    if (Debug) printf("Line %u, Bandwidth: %f GB/s, Time: %fus\n", Where.line(), GBytes / Seconds, Seconds * 1'000'000.0f);}

void PrintRuntime(const source_location& Where){
    uint32_t Microseconds = duration_cast<microseconds>(now() - SyncTime).count();
    SyncTime = now();
    if (Debug) printf("Line %u, Time %uus\n", Where.line(), Microseconds);}

int E(string Message, const source_location& Where = source_location::current()){
    printf("File:  %s\n, Function: %s\n, Line: %u\n", Where.file_name(), Where.function_name(), Where.line());
    printf("CUDA GetLast: %s\n", cudaGetErrorString(cudaGetLastError()));
    if (Message.compare("") != 0){
        printf("Message: %s\n", Message.c_str());}
    return 0;}
int E(cudaError_t Error, const source_location& Where = source_location::current()){
    if (Error == cudaSuccess){
        Error = cudaGetLastError();
        if (Error == cudaSuccess){
            return 1;}}
    printf("CUDA Error: %s\n", cudaGetErrorString(Error));
    return E("", Where);}

template <typename T>
int Zero(T* dData, uint64_t Count, const source_location& Where = source_location::current()) {
    return E(cudaMemset(dData, 0, sizeof(T) * Count), Where);}
template <typename T>
int Set(T* dData, uint64_t Count, uint8_t Value, const source_location& Where = source_location::current()) {
    return E(cudaMemset(dData, Value, sizeof(T) * Count), Where);}
template <typename T>
int Malloc(T** dData, uint64_t Count, const source_location& Where = source_location::current()) {
    if (!E(cudaMemGetInfo(&FreeBytes, &TotalBytes), Where)) return 0;
    if (Debug){
        PrintMemoryState(Where);}
    if (sizeof(T) * Count > FreeBytes) return E("Requested memory exceeded free memory", Where);
    *dData = nullptr;
    if (!E(cudaMalloc(reinterpret_cast<void**>(dData), sizeof(T) * Count), Where)) return 0;
    if (*dData == nullptr) return E("Failed to malloc dData", Where);
    return 1;}
template <typename T>
int ToHost(T* hData, T* dData, uint64_t Count, const source_location& Where = source_location::current()){
    if (!E(cudaMemcpy(hData, dData, sizeof(T) * Count, cudaMemcpyDeviceToHost), Where)) return 0;
    return 1;}
template <typename T>
int ToDevice(T* dData, T* hData, uint64_t Count, const source_location& Where = source_location::current()){
    if (!E(cudaMemcpy(dData, hData, sizeof(T) * Count, cudaMemcpyHostToDevice), Where)) return 0;
    return 1;}
template <typename T>
int DeviceCopy(T* dDest, T* dSrc, uint64_t Count, const source_location& Where = source_location::current()){
    if (!E(cudaMemcpy(dDest, dSrc, sizeof(T) * Count, cudaMemcpyDeviceToDevice), Where)) return 0;
    return 1;}
template <typename T>
int Copy(T* DestinationData, T* SourceData, uint64_t Count, int Direction, const source_location& Where = source_location::current()){
    if (Direction == 0) return ToHost(DestinationData, SourceData, Count);
    else return ToDevice(DestinationData, SourceData, Count);}
template <typename T>
int Free(T* dData, const source_location& Where = source_location::current()){
    if (dData == nullptr) return 1;
    return E(cudaFree(dData), Where);}
int Sync(const uint64_t ProcessedBytes = 0, const source_location& Where = source_location::current()){
    if (!E(cudaDeviceSynchronize(), Where)) return 0;
    if (Debug){
        if (ProcessedBytes == 0) PrintRuntime(Where);
        else PrintBandwidth(ProcessedBytes, Where);}
    return 1;}

__forceinline__ __host__ __device__ void Min(float4& Out, const float4 In){
    Out.x = min(Out.x, In.x);
    Out.y = min(Out.y, In.y);
    Out.z = min(Out.z, In.z);
    Out.w = min(Out.w, In.w);}
__forceinline__ __host__ __device__ void Max(float4& Out, const float4 In){
    Out.x = max(Out.x, In.x);
    Out.y = max(Out.y, In.y);
    Out.z = max(Out.z, In.z);
    Out.w = max(Out.w, In.w);}
__forceinline__ __host__ __device__ float Length(const float3& V){
    return sqrtf(V.x * V.x + V.y * V.y + V.z * V.z);}
__forceinline__ __host__ __device__ void Normalize(float3& V){
    const float Len = Length(V);
    V.x /= Len; V.y /= Len; V.z /= Len;}
__forceinline__ __host__ __device__ void Cross(float3& V0, const float3& V1){
    V0 = make_float3(V0.y * V1.z - V0.z * V1.y, V0.z * V1.x - V0.x * V1.z, V0.x * V1.y - V0.y * V1.x);}
__forceinline__ __host__ __device__ float Dot(const float3& V0, const float3& V1){
    return V0.x * V1.x + V0.y * V1.y + V0.z * V1.z;}
__forceinline__ __host__ __device__ float3 operator*(const float S, const float3& V){
    return make_float3(S * V.x, S * V.y, S * V.z);}
__forceinline__ __host__ __device__ float3& operator+=(float3& V0, const float3& V1){
    V0.x += V1.x; V0.y += V1.y; V0.z += V1.z;
    return V0;}
__forceinline__ __host__ __device__ float3 operator+(float3 V0, const float3 V1){
    return make_float3(V0.x + V1.x, V0.y + V1.y, V0.z + V1.z);}
__forceinline__ __host__ __device__ float4& operator+=(float4& V0, const float4& V1){
    V0.x += V1.x; V0.y += V1.y; V0.z += V1.z; V0.w += V1.w;
    return V0;}
__forceinline__ __host__ __device__ float3& operator-=(float3& V0, const float3& V1){
    V0.x -= V1.x; V0.y -= V1.y; V0.z -= V1.z;
    return V0;}
__forceinline__ __host__ __device__ float4& operator-=(float4& V0, const float4& V1){
    V0.x -= V1.x; V0.y -= V1.y; V0.z -= V1.z; V0.w -= V1.w;
    return V0;}
__forceinline__ __host__ __device__ float3& operator/=(float3& V0, const float& S){
    V0.x /= S; V0.y /= S; V0.z /= S;
    return V0;}
__forceinline__ __host__ __device__ bool operator!=(float3 V0, float3 V1){
    return V0.x != V1.x || V0.y != V1.y || V0.z != V1.z;}
__forceinline__ __host__ __device__ float3 operator/(float3& V0, const float& S){
    return make_float3(V0.x / S, V0.y / S, V0.z / S);}
__forceinline__ __host__ __device__ float3& Transform(float3& Position, const float* Matrix){
    Position = make_float3(Position.x * Matrix[0] + Position.y * Matrix[1] + Position.z * Matrix[2] + Matrix[9],
                           Position.x * Matrix[3] + Position.y * Matrix[4] + Position.z * Matrix[5] + Matrix[10],
                           Position.x * Matrix[6] + Position.y * Matrix[7] + Position.z * Matrix[8] + Matrix[11]);
    return Position;}
__forceinline__ __host__ __device__ float4 make_float4(float3 Vec){
    return make_float4(Vec.x, Vec.y, Vec.z, 0.0f);}

__global__ void IotaKernel(
    unsigned int* Values,
    unsigned int InitValue,
    uint32_t Count){
    const uint32_t Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index < Count) Values[Index] = InitValue + Index;}
uint64_t IotaUsage(uint32_t Count){
    return Count * sizeof(unsigned int);}

__global__ void CalculateTriangleNormalsKernel(
    float3* Positions,
    uint3* Indices,
    float3* Normals,
    uint32_t Count){
    const uint32_t Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= Count) return;
    const uint3 Verts = Indices[Index];
    float3 Edge0 = Positions[Verts.y];
    float3 Edge1 = Positions[Verts.z];
    const float3 Vert0 = Positions[Verts.x];
    Edge0 -= Vert0;
    Edge1 -= Vert0;
    Cross(Edge0, Edge1);
    Normalize(Edge0);
    Normals[Index] = Edge0;}
uint64_t CalculateTriangleNormalsUsage(uint32_t Count){
    return Count * (sizeof(uint3) + 4 * sizeof(float3));}

__global__ void TrianglesToReferencesKernel(
    uint3* Indices,
    unsigned long long* TriangleReferences,
    unsigned int* Offset,
    uint32_t StackCount,
    uint32_t TriangleCount,
    uint32_t VertexCount){
    const uint64_t TriangleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (TriangleIndex >= TriangleCount) return;
    const uint3 VertexIndices = Indices[TriangleIndex];
    for (uint32_t VertexIndex : {VertexIndices.x, VertexIndices.y, VertexIndices.z}){
        const uint32_t OffsetIndex = TriangleIndex % StackCount;
        const unsigned int LocalOffset = atomicAdd(&Offset[OffsetIndex], 1UL);
        const uint64_t Ptr = VertexCount + StackCount * LocalOffset + OffsetIndex;
        const uint64_t Reference = (Ptr << 32ULL) + TriangleIndex;
        unsigned long long PreviousReference = atomicExch(TriangleReferences + VertexIndex, Reference);
        TriangleReferences[Ptr] = PreviousReference;}}
uint64_t TriangleToReferencesUsage(uint32_t Count){
    return Count * (sizeof(uint3) + 6 * sizeof(unsigned long) + 9 * sizeof(unsigned long long));}

__global__ void SharpEdgesToReferencesKernel(
    uint2* Indices,
    unsigned long long* SharpEdgesReferences,
    unsigned int* Offset,
    uint32_t StackCount,
    uint32_t SharpEdgesCount,
    uint32_t VertexCount){
    const uint64_t SharpEdgeIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (SharpEdgeIndex >= SharpEdgesCount) return;
    const uint2 VertexIndices = Indices[SharpEdgeIndex];
    for (uint32_t VertexIndex : {VertexIndices.x, VertexIndices.y}){
        const uint32_t OffsetIndex = SharpEdgeIndex % StackCount;
        const unsigned int LocalOffset = atomicAdd(&Offset[OffsetIndex], 1UL);
        const uint64_t Ptr = VertexCount + StackCount * LocalOffset + OffsetIndex;
        const uint64_t Reference = (Ptr << 32ULL) + SharpEdgeIndex;
        unsigned long long PreviousReference = atomicExch(SharpEdgesReferences + VertexIndex, Reference);
        SharpEdgesReferences[Ptr] = PreviousReference;}}
uint64_t SharpEdgesToReferencesUsage(uint32_t Count){
    return Count * (sizeof(uint2) + 4 * sizeof(unsigned long) + 6 * sizeof(unsigned long long));}

constexpr uint64_t NoReference = ~(static_cast<uint64_t>(0));
constexpr uint32_t MaxReferencesBlockSize = 1024u;
__global__ void MaxReferencesKernel(
    unsigned long long* TriangleReferences,
    bool UseSharpEdges,
    unsigned long long* SharpEdgeReferences,
    uint32_t VertexCount,
    unsigned long long* MaxTriangleReferences,
    unsigned long long* MaxSharpEdgeReferences){
    __shared__ uint32_t ReferenceCount[MaxReferencesBlockSize];
    ReferenceCount[threadIdx.x] = 0;
    __syncthreads();
    const uint32_t VertexIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long Reference = NoReference;
    if (VertexIndex < VertexCount){
        Reference = TriangleReferences[VertexIndex];}
    while(Reference != NoReference){
        ReferenceCount[threadIdx.x]++;
        const uint32_t ReferenceIndex = Reference >> 32;
        Reference = TriangleReferences[ReferenceIndex];}
    for (uint32_t Reduction = MaxReferencesBlockSize / 2; Reduction > 0; Reduction /= 2){
        __syncthreads();
        if(threadIdx.x < Reduction){
            ReferenceCount[threadIdx.x] = max(ReferenceCount[threadIdx.x + Reduction], ReferenceCount[threadIdx.x]);}}
    __syncthreads();
    if (threadIdx.x == 0){
        atomicMax(MaxTriangleReferences, static_cast<unsigned long long>(ReferenceCount[threadIdx.x]));}
    if (UseSharpEdges){
        __syncthreads();
        ReferenceCount[threadIdx.x] = 0;
        __syncthreads();
        Reference = NoReference;
        if (VertexIndex < VertexCount){
            Reference = SharpEdgeReferences[VertexIndex];}
        while(Reference != NoReference){
            ReferenceCount[threadIdx.x]++;
            const uint32_t ReferenceIndex = Reference >> 32;
            Reference = SharpEdgeReferences[ReferenceIndex];}
        for (uint32_t Reduction = MaxReferencesBlockSize / 2; Reduction > 0; Reduction /= 2){
            __syncthreads();
            if(threadIdx.x < Reduction){
                ReferenceCount[threadIdx.x] = max(ReferenceCount[threadIdx.x + Reduction], ReferenceCount[threadIdx.x]);}}
        __syncthreads();
        if (threadIdx.x == 0){
            atomicMax(MaxSharpEdgeReferences, static_cast<unsigned long long>(ReferenceCount[threadIdx.x]));}}}
uint64_t MaxReferencesUsage(uint32_t VertexCount, uint32_t TriangleCount, uint32_t SharpEdgeCount){
    uint64_t Usage = VertexCount * sizeof(unsigned long long) + TriangleCount * 3 * sizeof(unsigned long long);
    if (SharpEdgeCount > 0){
        Usage += VertexCount * sizeof(unsigned long long) + SharpEdgeCount * 2 * sizeof(unsigned long long);}
    return Usage;}

__global__ void VertexNormalsKernel(
    unsigned long long* TriangleReferences,
    unsigned long long* SharpEdgeReferences,
    uint32_t MaxTriangleReferenceCount,
    uint32_t MaxSharpEdgeReferenceCount,
    uint32_t VertexCount,
    float3* TriangleNormals,
    unsigned int* NormalPtr,
    float3* VertexNormals,
    uint2* SharpEdges,
    uint3* PositionIndices,
    uint3* NormalIndices){
    extern __shared__ uint32_t Shared[];
    uint32_t TotalReferenceCount = 5 * MaxTriangleReferenceCount + 2 * MaxSharpEdgeReferenceCount;
    const uint32_t ReferenceOffset = threadIdx.x * TotalReferenceCount; 
    uint32_t* TReferences = &Shared[ReferenceOffset];
    uint32_t* EReferences = &Shared[ReferenceOffset + MaxTriangleReferenceCount];
    uint32_t* GroupIDs = &Shared[ReferenceOffset + MaxTriangleReferenceCount + MaxSharpEdgeReferenceCount];
    uint32_t* NextTriangleStack = &Shared[ReferenceOffset + 2 * MaxTriangleReferenceCount + MaxSharpEdgeReferenceCount];
    uint32_t* TVerts = &Shared[ReferenceOffset + 3 * MaxTriangleReferenceCount + MaxSharpEdgeReferenceCount];
    uint32_t* EVerts = &Shared[ReferenceOffset + 5 * MaxTriangleReferenceCount + MaxSharpEdgeReferenceCount];
    for (int t = 0; t < MaxTriangleReferenceCount; t++){
        TReferences[t] = 0u;
        GroupIDs[t] = UINT32_MAX;
        TVerts[2*t+0] = 0u;
        TVerts[2*t+1] = 0u;}
    for (int e = 0; e < MaxSharpEdgeReferenceCount; e++){
        EReferences[e] = 0u;
        EVerts[e] = 0u;}
    __syncthreads();
    const uint32_t VertexIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long Reference = VertexIndex < VertexCount ? TriangleReferences[VertexIndex] : NoReference;
    uint32_t TReferenceCount = 0u;
    while(Reference != NoReference){
        const uint32_t TriangleIndex = Reference & 0xffffffff;
        TReferences[TReferenceCount] = TriangleIndex;
        TReferenceCount++;
        const uint32_t ReferenceIndex = Reference >> 32;
        Reference = TriangleReferences[ReferenceIndex];}
    Reference = VertexIndex < VertexCount ? SharpEdgeReferences[VertexIndex] : NoReference;
    uint32_t EReferenceCount = 0u;
    while (Reference != NoReference){
        const uint32_t SharpEdgeIndex = Reference & 0xffffffff;
        EReferences[EReferenceCount] = SharpEdgeIndex;
        EReferenceCount++;
        const uint32_t ReferenceIndex = Reference >> 32;
        Reference = SharpEdgeReferences[ReferenceIndex];}
    if (MaxSharpEdgeReferenceCount == 0){
        float3 VertexNormal{0.0f, 0.0f, 0.0f};
        for (int t = 0; t < TReferenceCount; t++){
            const float3 TriangleNormal = TriangleNormals[TReferences[t]];
            VertexNormal += TriangleNormal;}
        Normalize(VertexNormal);
        if (VertexIndex >= VertexCount) return;
        unsigned int NormalIndex = atomicAdd(NormalPtr, 1);
        VertexNormals[NormalIndex] = VertexNormal;
        for (int t = 0; t < TReferenceCount; t++){
            uint3 PIndices = PositionIndices[TReferences[t]];
            if (VertexIndex == PIndices.x){
                NormalIndices[TReferences[t]].x = NormalIndex;}
            if (VertexIndex == PIndices.y){
                NormalIndices[TReferences[t]].y = NormalIndex;}
            if (VertexIndex == PIndices.z){
                NormalIndices[TReferences[t]].z = NormalIndex;}}}
    else{
        for (int t = 0; t < TReferenceCount; t++){
            uint3 PIndices = PositionIndices[TReferences[t]];
            uint32_t j = 0;
            if (PIndices.x != VertexIndex){
                TVerts[2*t+0] = PIndices.x;
                j++;}
            if (PIndices.y != VertexIndex){
                if (j == 0){
                    TVerts[2*t+0] = PIndices.y;}
                if (j == 1){
                    TVerts[2*t+1] = PIndices.y;}
                j++;}
            if (PIndices.z != VertexIndex){
                TVerts[2*t+1] = PIndices.z;}}
        for (int e = 0; e < EReferenceCount; e++){
            uint2 EIndices = SharpEdges[EReferences[e]];
            if (EIndices.x != VertexIndex){
                EVerts[e] = EIndices.x;}
            else {
                EVerts[e] = EIndices.y;}}
        __syncthreads();
        if (VertexIndex >= VertexCount) return;
        uint32_t Grouped = 0;
        uint32_t GroupID = 0;
        uint32_t NextTriangleStackPtr = 0;
        while (Grouped < TReferenceCount){
            uint32_t FirstUngroupedT = 0;
            for(; FirstUngroupedT < TReferenceCount; FirstUngroupedT++){
                if (GroupIDs[FirstUngroupedT] == UINT32_MAX) break;}
            NextTriangleStack[NextTriangleStackPtr++] = FirstUngroupedT;
            while(NextTriangleStackPtr > 0){
                uint32_t NextT = NextTriangleStack[--NextTriangleStackPtr];
                if (GroupIDs[NextT] != UINT32_MAX) continue;
                GroupIDs[NextT] = GroupID;
                Grouped++;
                uint32_t OtherVertA = TVerts[2 * NextT + 0];
                uint32_t OtherVertB = TVerts[2 * NextT + 1];
                bool IsSharpA = false;
                bool IsSharpB = false;
                for (int e = 0; e < EReferenceCount; e++){
                    if (EVerts[e] == OtherVertA){
                        IsSharpA = true;}
                    if (EVerts[e] == OtherVertB){
                        IsSharpB = true;}}
                for (int t = 0; t < TReferenceCount; t++){
                    if(GroupIDs[t] != UINT32_MAX) continue;
                    if(!IsSharpA){
                        if (TVerts[2 * t + 0] == OtherVertA || TVerts[2 * t + 1] == OtherVertA){
                            NextTriangleStack[NextTriangleStackPtr++] = t;}}
                    if (!IsSharpB){
                        if (TVerts[2 * t + 0] == OtherVertB || TVerts[2 * t + 1] == OtherVertB){
                            NextTriangleStack[NextTriangleStackPtr++] = t;}}}}
            GroupID++;}
        unsigned int StartNormalIndex = atomicAdd(NormalPtr, GroupID);
        for (int ID = 0; ID < GroupID; ID++){
            float3 VertexNormal{0.0f, 0.0f, 0.0f};
            for (int t = 0; t < TReferenceCount; t++){
                if (GroupIDs[t] == ID){
                    VertexNormal += TriangleNormals[TReferences[t]];}}
            Normalize(VertexNormal);
            unsigned int NormalIndex = StartNormalIndex + ID;
            VertexNormals[NormalIndex] = VertexNormal;
            for (int t = 0; t < TReferenceCount; t++){
                if (GroupIDs[t] == ID){
                    uint3 PIndices = PositionIndices[TReferences[t]];
                    if (VertexIndex == PIndices.x){
                        NormalIndices[TReferences[t]].x = NormalIndex;}
                    if (VertexIndex == PIndices.y){
                        NormalIndices[TReferences[t]].y = NormalIndex;}
                    if (VertexIndex == PIndices.z){
                        NormalIndices[TReferences[t]].z = NormalIndex;}}}}}}
__global__ void ReferencesToVertexNormalsKernel(
    unsigned long long* TriangleReferences,
    float3* TriangleNormals,
    float3* VertexNormals,
    uint32_t VertexCount){
    const uint32_t Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= VertexCount) return;
    unsigned long long Reference = TriangleReferences[Index];
    float3 VertexNormal{0.0f, 0.0f, 0.0f};
    while(Reference != NoReference){
        const uint32_t TriangleIndex = Reference & 0xffffffff;
        const uint32_t ReferenceIndex = Reference >> 32;
        const float3 TriangleNormal = TriangleNormals[TriangleIndex];
        VertexNormal += TriangleNormal;
        Reference = TriangleReferences[ReferenceIndex];}
    Normalize(VertexNormal);
    VertexNormals[Index] = VertexNormal;}
uint64_t ReferencesToVertexNormalsUsage(uint32_t Count){
    return Count * (6 * sizeof(unsigned long long) + 7 * sizeof(float3));}

__global__ void TrianglePositionsKernel(
    float3* Positions,
    uint3* Indices,
    float3* TrianglePositions,
    uint32_t Count){
    uint32_t Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= Count) return;
    float3 TrianglePosition{0.0f, 0.0f, 0.0f};
    uint3 Verts = Indices[Index];
    for (uint32_t Vertex : {Verts.x, Verts.y, Verts.z}){
        TrianglePosition += Positions[Vertex];}
    TrianglePosition /= 3.0f;
    TrianglePositions[Index] = TrianglePosition;}
uint64_t TrianglePositionsUsage(uint32_t TriangleCount){
    return TriangleCount * (sizeof(uint3) + 4 * sizeof(float3));}

__global__ void ParallelAverageKernel(
    unsigned int* HistogramOffsets,
    unsigned int* Histograms,
    unsigned int* TriangleIndices,
    float3* Positions,
    float3* BlockAverages){
    __shared__ float3 Average[DefaultBlockDim.x];
    Average[threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
    const uint32_t HistogramOffset = HistogramOffsets[blockIdx.y];
    const uint32_t Start = HistogramOffset + blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t Stop = HistogramOffset + Histograms[blockIdx.y];
    const uint32_t Step = blockDim.x * gridDim.x;
    for (uint32_t Index = Start; Index < Stop; Index += Step){
        Average[threadIdx.x] += Positions[TriangleIndices[Index]];}
    for (uint32_t Reduction = DefaultBlockDim.x / 2; Reduction > 0; Reduction /= 2){
        __syncthreads();
        if(threadIdx.x < Reduction){
            Average[threadIdx.x] += Average[threadIdx.x + Reduction];}}
    __syncthreads();
    if (threadIdx.x == 0){
        BlockAverages[blockIdx.y * gridDim.x + blockIdx.x] = Average[0] / Histograms[blockIdx.y];}}
uint64_t ParallelAverageUsage(uint32_t Count, dim3 GridDim, dim3 BlockDim){
    const uint32_t BlockCount = GridDim.y * GridDim.x; 
    const uint32_t ThreadCount = BlockDim.x * BlockCount;
    return ThreadCount * (2 * sizeof(unsigned int)) + Count * (sizeof(unsigned int) + sizeof(float3)) + BlockCount * (sizeof(float3) + sizeof(uint32_t));}

__global__ void FinalAverageKernel(
    float3* BlockAverages,
    float3* FinalAverages){
    extern __shared__ float3 Average[];
    Average[threadIdx.x] = BlockAverages[blockIdx.x * blockDim.x + threadIdx.x];
    uint32_t Reduction = 1;
    while (Reduction < (blockDim.x - 1) / 2 + 1) Reduction *= 2;
    for (; Reduction > 0; Reduction /= 2){
        __syncthreads();
        if(threadIdx.x < Reduction){
            const uint32_t OtherIndex = threadIdx.x + Reduction;
            if (OtherIndex < blockDim.x){
                Average[threadIdx.x] += Average[OtherIndex];}}}
    __syncthreads();
    if (threadIdx.x == 0){
        FinalAverages[blockIdx.x] = Average[0];}}
uint64_t FinalAverageUsage(dim3 GridDim, dim3 BlockDim){
    const uint32_t ThreadCount = GridDim.x * BlockDim.x;
    return ThreadCount * sizeof(float3) + GridDim.x * sizeof(float3);}

__global__ void TriangleToHistogramKernel(
    float3* Averages,
    unsigned int* HistogramOffsets,
    unsigned int* Histograms,
    unsigned int* TriangleIndices,
    float3* TrianglePositions,
    unsigned int* ChildHistograms,
    unsigned int* TriangleHistogramIndices,
    unsigned int* TriangleHistogramPositions,
    uint32_t Count){
    const float3 Average = Averages[blockIdx.y];
    const uint32_t HistogramOffset = HistogramOffsets[blockIdx.y];
    const uint32_t Histogram = Histograms[blockIdx.y];
    const uint32_t StartIndex = HistogramOffset + blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t StopIndex = HistogramOffset + Histogram;
    const uint32_t StepIndex = gridDim.x * blockDim.x;
    for (uint32_t Index = StartIndex; Index < StopIndex; Index += StepIndex){
        const uint32_t TriangleIndex = TriangleIndices[Index];
        const float3 Position = TrianglePositions[TriangleIndex];
        const uint32_t HistogramIndex = 8 * blockIdx.y + 4 * static_cast<uint32_t>(Position.z > Average.z) + 2 * static_cast<uint32_t>(Position.y > Average.y) + static_cast<uint32_t>(Position.x > Average.x);
        TriangleHistogramIndices[Index] = HistogramIndex;
        TriangleHistogramPositions[Index] = atomicAdd(ChildHistograms + HistogramIndex, 1);}}
uint64_t TriangleToHistogramUsage(uint32_t Count, dim3 GridDim, dim3 BlockDim){
    const uint32_t ThreadCount = GridDim.y * GridDim.x * BlockDim.y * BlockDim.x;
    return ThreadCount * (sizeof(float3) + 2 * sizeof(uint32_t)) + Count * (4 * sizeof(unsigned int) + sizeof(float3));}

__global__ void HistogramBranchOrLeafKernel(
    unsigned int* Histograms,
    bool* IsBranches,
    bool* IsLeaves,
    uint32_t RemainingChildren){
    const uint32_t Index = blockDim.x * blockIdx.x + threadIdx.x;
    if (Index >= RemainingChildren) return;
    const uint32_t Histogram = Histograms[Index];
    if (Histogram > 8){
        IsBranches[Index] = true;
        IsLeaves[Index] = false;}
    else if (Histogram > 0){
        IsBranches[Index] = false;
        IsLeaves[Index] = true;}
    else {
        IsBranches[Index] = false;
        IsLeaves[Index] = false;}}
uint64_t HistogramBranchOrLeafUsage(uint32_t RemainingChildren){
    return RemainingChildren * (sizeof(uint32_t) + 2 * sizeof(bool));}

__global__ void HistogramOffsetsKernel(
    unsigned int* HistogramOffsets,
    unsigned int* ChildHistograms,
    unsigned int* ChildHistogramOffsets){
    __shared__ unsigned int LocalHistogramOffsets[ChildCount];
    const unsigned int HistogramOffset = HistogramOffsets[blockIdx.y];
    LocalHistogramOffsets[threadIdx.x] = ChildHistograms[blockIdx.y * ChildCount + threadIdx.x];
    for (uint32_t Reduce = 1; Reduce < ChildCount; Reduce*=2){
        __syncthreads();
        if (Reduce & threadIdx.x){
        // threadIdx.x - Reduce goes to the lower partial
        // ORing with (Reduce - 1) make the index the last element of the partial.
        LocalHistogramOffsets[threadIdx.x] += LocalHistogramOffsets[threadIdx.x - Reduce | Reduce - 1];}}
    __syncthreads();
    ChildHistogramOffsets[blockIdx.y * ChildCount + threadIdx.x] = threadIdx.x == 0 ? HistogramOffset : HistogramOffset + LocalHistogramOffsets[threadIdx.x-1];}
uint64_t HistogramOffsetsUsage(uint64_t RemainingBins){
    return RemainingBins * (24 * sizeof(unsigned int));}

__global__ void UpBranchLeafIndicesKernel(
    bool* IsBranches,
    bool* IsLeaves,
    unsigned int* BranchIndices,
    unsigned int* LeafIndices,
    uint32_t Order,
    uint32_t RemainingChildren){
    __shared__ unsigned int Shared[2048];
    unsigned int* LocalBranchIndices = &Shared[0];
    unsigned int* LocalLeafIndices = &Shared[1024];
    const uint32_t Index = (blockIdx.x * blockDim.x + threadIdx.x + 1 << (Order * 10)) - 1;
    LocalBranchIndices[threadIdx.x] = 0U;
    LocalLeafIndices[threadIdx.x] = 0U;
    if (Index < RemainingChildren){
        if (Order == 0){
            if (IsBranches[Index]){
                LocalBranchIndices[threadIdx.x] = 1U;}
            if (IsLeaves[Index]){
                LocalLeafIndices[threadIdx.x] = 1U;}}
        else {
            LocalBranchIndices[threadIdx.x] = BranchIndices[Index];
            LocalLeafIndices[threadIdx.x] = LeafIndices[Index];}}
    for (uint32_t Reduce = 1; Reduce < 1024; Reduce*=2){
        __syncthreads();
        if (Reduce & threadIdx.x){
            // threadIdx.x - Reduce goes to the lower partial
            // ORing with (Reduce - 1) make the index the last element of the partial.
            LocalBranchIndices[threadIdx.x] += LocalBranchIndices[threadIdx.x - Reduce | Reduce - 1];
            LocalLeafIndices[threadIdx.x] += LocalLeafIndices[threadIdx.x - Reduce | Reduce - 1];}}
    __syncthreads();
    if (Index < RemainingChildren){
        BranchIndices[Index] = LocalBranchIndices[threadIdx.x];
        LeafIndices[Index] = LocalLeafIndices[threadIdx.x];}}
uint64_t UpBranchLeafIndicesUsage(uint32_t Order, uint32_t Count){
    const uint32_t ReadCount = Count >> 10 * Order;
    return ReadCount * (2 * sizeof(bool) + 3 * sizeof(uint32_t)) + ReadCount / 1024 * (3 * sizeof(uint32_t));}

__global__ void DownBranchLeafIndicesKernel(
    unsigned int* BranchIndices,
    unsigned int* LeafIndices,
    uint32_t Order,
    uint32_t RemainingChildren){
    const uint32_t Index = (blockIdx.x * blockDim.x + threadIdx.x << Order * 10);
    if (Index >= RemainingChildren) return;
    if (blockIdx.x == 0) return;
    if (threadIdx.x == 1023) return;
    const uint32_t ApplyIndex = (blockIdx.x * blockDim.x << Order * 10) - 1;
    const uint32_t BranchOffset = BranchIndices[ApplyIndex];
    const uint32_t LeafOffset = LeafIndices[ApplyIndex];
    BranchIndices[Index] += BranchOffset;
    LeafIndices[Index] += LeafOffset;}
uint64_t DownBranchLeafIndicesUsage(uint32_t Order, uint32_t Count){
    const uint32_t ReadCount = Count >> 10 * Order;
    return ReadCount * (6ULL * sizeof(unsigned int));}

__global__ void ReorderTrianglesKernel(
    unsigned int* HistogramOffsets,
    unsigned int* Histograms,
    unsigned int* ChildHistogramOffsets,
    unsigned int* TriangleIndices,
    unsigned int* TriangleHistogramIndices,
    unsigned int* TriangleHistogramPositions,
    unsigned int* ReorderedTriangleIndices,
    uint32_t Count){
    const uint32_t StartIndex = HistogramOffsets[blockIdx.y] + blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t StopIndex = HistogramOffsets[blockIdx.y] + Histograms[blockIdx.y];
    const uint32_t StepIndex = blockDim.x * gridDim.x;
    for (uint32_t Index = StartIndex; Index < StopIndex; Index+=StepIndex){
        const uint32_t HistogramIndex = TriangleHistogramIndices[Index];
        const uint32_t HistogramOffset = ChildHistogramOffsets[HistogramIndex];
        const uint32_t HistogramPosition = TriangleHistogramPositions[Index];
        const uint32_t ReorderedIndex = HistogramOffset + HistogramPosition;
        ReorderedTriangleIndices[ReorderedIndex] = TriangleIndices[Index];}}
uint64_t ReorderTrianglesUsage(uint32_t Count){
    return Count * 5 * sizeof(unsigned int);}

__global__ void MakeNextHistogramsKernel(
    bool* IsBranches,
    unsigned int* BranchIndices,
    unsigned int* ChildHistograms,
    unsigned int* ChildHistogramOffsets,
    unsigned int* NextHistograms,
    unsigned int* NextHistogramOffsets,
    unsigned int RemainingChildren){
    const uint32_t Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= RemainingChildren) return;
    if (!IsBranches[Index]) return;
    const uint32_t BranchIndex = Index == 0 ? 0 : BranchIndices[Index - 1];
    NextHistograms[BranchIndex] = ChildHistograms[Index];
    NextHistogramOffsets[BranchIndex] = ChildHistogramOffsets[Index];}
uint64_t MakeNextHistogramsUsage(uint32_t Count){
    return Count * (sizeof(bool) + 5 * sizeof(unsigned int));}

__global__ void PackAABBKernel(
    unsigned int* ChildHistograms,
    unsigned int* ChildHistogramOffsets,
    bool* IsBranches,
    bool* IsLeaves,
    unsigned int* BranchIndices,
    unsigned int* LeafIndices,
    AABBBranch_t* AABBBranches,
    AABBLeaf_t* AABBLeaves,
    uint32_t RemainingBins,
    uint32_t BranchAABBOffset,
    uint32_t LeafAABBOffset,
    uint32_t BranchOffset,
    uint32_t LeafOffset){
    const uint32_t Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= RemainingBins) return;
    AABBBranch_t Parent;
    Parent.BranchCount = 0;
    Parent.LeafCount = 0;
    Parent.BoxIndex = BranchAABBOffset + Index;
    bool FoundBranch = false;
    bool FoundLeaf = false;
    for (uint32_t ChildIndex = 0; ChildIndex < ChildCount; ChildIndex++){
        const uint32_t TotalIndex = ChildCount * Index + ChildIndex;
        if (IsBranches[TotalIndex]){
            Parent.BranchCount++;
            if (!FoundBranch){
                Parent.BranchOffset = TotalIndex == 0 ? 0 : BranchIndices[TotalIndex - 1];
                Parent.BranchOffset += BranchOffset + RemainingBins;}
            FoundBranch = true;}
        if (IsLeaves[TotalIndex]){
            Parent.LeafCount++;
            if (!FoundLeaf){
                Parent.LeafOffset = TotalIndex == 0 ? 0 : LeafIndices[TotalIndex - 1];
                Parent.LeafOffset += LeafOffset;}
            FoundLeaf = true;
        
            AABBLeaf_t Leaf;
            const uint32_t LeafIndex = TotalIndex == 0 ? 0 : LeafIndices[TotalIndex - 1];
            Leaf.BoxIndex = LeafAABBOffset + LeafIndex;
            Leaf.TriangleCount = ChildHistograms[TotalIndex];
            Leaf.TriangleOffset = TotalIndex == 0 ? 0 : ChildHistogramOffsets[TotalIndex];
            AABBLeaves[LeafOffset + LeafIndex] = Leaf;}}
    AABBBranches[BranchOffset + Index] = Parent;

    for (uint32_t ChildIndex = 0u; ChildIndex < ChildCount; ChildIndex++){
        const uint32_t TotalIndex = ChildCount * Index + ChildIndex;
        if (IsLeaves[TotalIndex]){}}}
uint64_t PackAABBUsage(uint32_t Count){
    return Count * (5 * sizeof(unsigned int));}

__global__ void CalculateTrianglesKernel(
    unsigned int* ReorderedIndices,
    uint3* PositionIndices,
    float3* Positions,
    uint3* NormalIndices,
    float3* VertexNormals,
    Triangle_t* Triangles,
    uint32_t Count){
    const uint32_t Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= Count) return;
    Triangle_t Out;
    const uint32_t ReorderedIndex = ReorderedIndices[Index];
    const uint3 PositionIndex = PositionIndices[ReorderedIndex];
    const uint3 NormalIndex = NormalIndices[ReorderedIndex];
    Out.V0 = make_float4(Positions[PositionIndex.x]);
    Out.Edge0 = make_float4(Positions[PositionIndex.y]);
    Out.Edge0 -= Out.V0;
    Out.Edge1 = make_float4(Positions[PositionIndex.z]);
    Out.Edge1 -= Out.V0;
    Out.Normal0 = make_float4(VertexNormals[NormalIndex.x]);
    Out.Normal1 = make_float4(VertexNormals[NormalIndex.y]);
    Out.Normal2 = make_float4(VertexNormals[NormalIndex.z]);
    Triangles[Index] = Out;}
uint64_t CalculateTrianglesUsage(uint32_t Count){
    return Count * (sizeof(uint32_t) + sizeof(uint3) + 3 * sizeof(float3) + 3 * sizeof(float3) + 3 * sizeof(float3));}

__global__ void LeafMinMaxKernel(
    AABBLeaf_t* AABBLeaves,
    Triangle_t* Triangles,
    uint32_t* LeafOffsets,
    uint32_t* LeafCounts,
    AABB_t* AABBs){
    const uint32_t StartIndex = LeafOffsets[blockIdx.x] + threadIdx.x;
    const uint32_t StopIndex = LeafOffsets[blockIdx.x] + LeafCounts[blockIdx.x];
    for (uint32_t LeafIndex = StartIndex; LeafIndex < StopIndex; LeafIndex += blockDim.x){
        AABB_t Out;
        Out.Max = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        Out.Min = make_float4(INFINITY, INFINITY, INFINITY, INFINITY);
        AABBLeaf_t AABBLeaf = AABBLeaves[LeafIndex];
        for (uint32_t ChildIndex = 0; ChildIndex < AABBLeaf.TriangleCount; ChildIndex++){
            const Triangle_t Triangle = Triangles[AABBLeaf.TriangleOffset + ChildIndex];
            float4 V1 = Triangle.V0;
            V1 += Triangle.Edge0;
            float4 V2 = Triangle.V0;
            V2 += Triangle.Edge1;
            for (float4 Position : {Triangle.V0, V1, V2}){
                Max(Out.Max, Position);
                Min(Out.Min, Position);}}
        AABBs[AABBLeaf.BoxIndex] = Out;}}
uint64_t LeafMinMaxUsage(uint32_t LeafCount, uint32_t TriangleCount){
    return LeafCount * (sizeof(AABBLeaf_t) + sizeof(AABB_t)) + TriangleCount * sizeof(Triangle_t);}

__global__ void BranchMinMaxKernel(
    AABBBranch_t* AABBBranches,
    AABBLeaf_t* AABBLeaves,
    uint32_t Level,
    uint32_t* BranchCounts,
    uint32_t* BranchOffsets,
    AABB_t* AABBs){
    const uint32_t StartIndex = BranchOffsets[Level] + blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t StopIndex = BranchOffsets[Level] + BranchCounts[Level];
    for (uint32_t BranchIndex = StartIndex; BranchIndex < StopIndex; BranchIndex += gridDim.x * blockDim.x){
        AABB_t Out;
        Out.Max = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        Out.Min = make_float4(INFINITY, INFINITY, INFINITY, INFINITY);
        AABBBranch_t AABBBranch = AABBBranches[BranchIndex];
        for (uint32_t ChildIndex = 0; ChildIndex < AABBBranch.BranchCount; ChildIndex++){
            const AABBBranch_t Child = AABBBranches[AABBBranch.BranchOffset + ChildIndex];
            const AABB_t AABB = AABBs[Child.BoxIndex];
            Max(Out.Max, AABB.Max);
            Min(Out.Min, AABB.Min);}
        for (uint32_t ChildIndex = 0; ChildIndex < AABBBranch.LeafCount; ChildIndex++){
            const AABBLeaf_t Child = AABBLeaves[AABBBranch.LeafOffset + ChildIndex];
            const AABB_t AABB = AABBs[Child.BoxIndex];
            Max(Out.Max, AABB.Max);
            Min(Out.Min, AABB.Min);}
        AABBs[AABBBranch.BoxIndex] = Out;}}
uint64_t BranchMinMaxUsage(uint32_t BranchCount, uint32_t NextBranchCount, uint32_t LeafCount){
    return (BranchCount + NextBranchCount) * (sizeof(AABBBranch_t) + sizeof(AABB_t)) + LeafCount * (sizeof(AABBLeaf_t) + sizeof(AABB_t));}
extern "C" {
Positions_t MallocPositions(uint64_t Count){
    Positions_t Out{};
    if (!Malloc(&Out.dData, Count)) return Out;
    Out.Count = Count;
    return Out;}
Indices_t MallocIndices(uint64_t Count){
    Indices_t Out{};
    if (!Malloc(&Out.dData, Count)) return Out;
    Out.Count = Count;
    return Out;}
Edges_t MallocEdges(uint64_t Count){
    Edges_t Out{};
    if (!Malloc(&Out.dData, Count)) return Out;
    Out.Count = Count;
    return Out;}
Normals_t MallocNormals(uint64_t Count){
    Normals_t Out{};
    if (!Malloc(&Out.dData, Count)) return Out;
    Out.Count = Count;
    return Out;}
VertexNormalsHeap_t MallocVertexNormalsHeap(uint64_t TriangleCount, uint64_t VertexCount, uint64_t SharpEdgeCount, uint64_t StackCount){
    VertexNormalsHeap_t Out{};
    const uint64_t TriangleReferenceCount = ((TriangleCount + TriangleCount * VerticesPerTriangle - 1) / StackCount + 1) * StackCount;
    uint64_t SharpEdgeReferenceCount = ((TriangleCount + SharpEdgeCount * VerticesPerEdge - 1) / StackCount + 1) * StackCount;
    const uint64_t ConservativeNormalCount = VertexCount + SharpEdgeCount;
    if (!Malloc(&Out.dTriangleReferences, TriangleReferenceCount)) return Out;
    if (!Malloc(&Out.dSharpEdgeReferences, SharpEdgeReferenceCount)) return Out;
    if (!Malloc(&Out.dOffset, StackCount)) return Out;
    if (!Malloc(&Out.dMaxTriangleReferences, 1)) return Out;
    if (!Malloc(&Out.dMaxSharpEdgeReferences, 1)) return Out;
    if (!Malloc(&Out.dNormalPtr, 1)) return Out;
    Out.TriangleCount = TriangleCount;
    Out.VertexCount = VertexCount;
    Out.ConservativeNormalCount = ConservativeNormalCount;
    Out.StackCount = StackCount;
    Out.TriangleReferenceCount = TriangleReferenceCount;
    Out.SharpEdgeReferenceCount = SharpEdgeReferenceCount;
    return Out;}
AABBHeap_t MallocAABBHeap(uint64_t TriangleCount){
    AABBHeap_t Out{};
    const uint32_t ConservativeLevelCount = floorf(log2f(TriangleCount));
    Out.FinalAverageCount = 256;
    Out.HistogramCount = max(static_cast<uint64_t>(8), TriangleCount);
    Out.LevelCount = floorf(log2f(TriangleCount));
    if (!Malloc(&Out.dTrianglePositions, TriangleCount)) return Out;
    if (!Malloc(&Out.dBlockAverages, Out.HistogramCount)) return Out;
    if (!Malloc(&Out.dFinalAverages, Out.FinalAverageCount)) return Out;
    if (!Malloc(&Out.dHistogramsA, Out.HistogramCount)) return Out;
    if (!Malloc(&Out.dHistogramsB, Out.HistogramCount)) return Out;
    if (!Malloc(&Out.dChildHistograms, Out.HistogramCount)) return Out;
    if (!Malloc(&Out.dHistogramOffsetsA, Out.HistogramCount)) return Out;
    if (!Malloc(&Out.dHistogramOffsetsB, Out.HistogramCount)) return Out;
    if (!Malloc(&Out.dChildHistogramOffsets, Out.HistogramCount)) return Out;
    if (!Malloc(&Out.dIsBranches, Out.HistogramCount)) return Out;
    if (!Malloc(&Out.dIsLeaves, Out.HistogramCount)) return Out;
    if (!Malloc(&Out.dBranchIndices, Out.HistogramCount)) return Out;
    if (!Malloc(&Out.dLeafIndices, Out.HistogramCount)) return Out;
    if (!Malloc(&Out.dAABBBranches, Out.HistogramCount)) return Out;
    if (!Malloc(&Out.dAABBLeaves, Out.HistogramCount)) return Out;
    if (!Malloc(&Out.dAABBs, Out.HistogramCount)) return Out;
    if (!Malloc(&Out.dTriangleHistogramIndices, TriangleCount)) return Out;
    if (!Malloc(&Out.dTriangleHistogramPositions, TriangleCount)) return Out;
    if (!Malloc(&Out.dReorderedIndicesA, TriangleCount)) return Out;
    if (!Malloc(&Out.dReorderedIndicesB, TriangleCount)) return Out;
    if (!Malloc(&Out.dTriangles, TriangleCount)) return Out;
    if (!Malloc(&Out.dBranchCounts, Out.LevelCount)) return Out;
    if (!Malloc(&Out.dBranchOffsets, Out.LevelCount)) return Out;
    if (!Malloc(&Out.dLeafCounts, Out.LevelCount)) return Out;
    if (!Malloc(&Out.dLeafOffsets, Out.LevelCount)) return Out;
    Out.hAABBHeader = static_cast<AABBHeader_t*>(malloc(sizeof(AABBHeader_t)));
    if (!Out.hAABBHeader) return Out;
    Out.TriangleCount = TriangleCount;
    return Out;}

int ToDevicePositions(Positions_t Positions, float3* hPosiitons){
    return ToDevice(Positions.dData, hPosiitons, Positions.Count);}
int ToDeviceIndices(Indices_t Indices, uint3* hIndices){
    return ToDevice(Indices.dData, hIndices, Indices.Count);}
int ToDeviceEdges(Edges_t Edges, uint2* hEdges){
    return ToDevice(Edges.dData, hEdges, Edges.Count);}
int ToDeviceNormals(Normals_t Normals, float3* hNormals){
    return ToDevice(Normals.dData, hNormals, Normals.Count);}

int ToHostPositions(float3* hPosiitons, Positions_t Positions){
    return ToHost(hPosiitons, Positions.dData, Positions.Count);}
int ToHostIndices(uint3* hIndices, Indices_t Indices){
    return ToHost(hIndices, Indices.dData, Indices.Count);}
int ToHostEdges(uint2* hEdges, Edges_t Edges){
    return ToHost(hEdges, Edges.dData, Edges.Count);}
int ToHostNormals(float3* hNormals, Normals_t Normals){
    return ToHost(hNormals, Normals.dData, Normals.Count);}
int ToHostAABBHeap(
    float3* hTrianglePositions,
    float3* hBlockAverages,
    float3* hFinalAverages,
    unsigned int* hHistogramsA,
    unsigned int* hHistogramsB,
    unsigned int* hChildHistograms,
    unsigned int* hHistogramOffsetsA,
    unsigned int* hHistogramOffsetsB,
    unsigned int* hChildHistogramOffsets,
    bool* hIsBranches,
    bool* hIsLeaves,
    unsigned int* hBranchIndices,
    unsigned int* hLeafIndices,
    AABBBranch_t* hAABBBranches,
    AABBLeaf_t* hAABBLeaves,
    AABB_t* hAABBs,
    uint32_t* hTriangleHistogramIndices,
    uint32_t* hTriangleHistogramPositions,
    unsigned int* hReorderedIndicesA,
    unsigned int* hReorderedIndicesB,
    Triangle_t* hTriangles,
    uint32_t* hBranchCounts,
    uint32_t* hBranchOffsets,
    uint32_t* hLeafCounts,
    uint32_t* hLeafOffsets,
    AABBHeader_t* hAABBHeader,
    AABBHeap_t AABBHeap){
    if (!ToHost(hTrianglePositions, AABBHeap.dTrianglePositions, AABBHeap.TriangleCount)) return 0;
    if (!ToHost(hBlockAverages, AABBHeap.dBlockAverages, AABBHeap.HistogramCount)) return 0;
    if (!ToHost(hFinalAverages, AABBHeap.dFinalAverages, AABBHeap.FinalAverageCount)) return 0;
    if (!ToHost(hHistogramsA, AABBHeap.dHistogramsA, AABBHeap.HistogramCount)) return 0;
    if (!ToHost(hHistogramsB, AABBHeap.dHistogramsB, AABBHeap.HistogramCount)) return 0;
    if (!ToHost(hChildHistograms, AABBHeap.dChildHistograms, AABBHeap.HistogramCount)) return 0;
    if (!ToHost(hHistogramOffsetsA, AABBHeap.dHistogramOffsetsA, AABBHeap.HistogramCount)) return 0;
    if (!ToHost(hHistogramOffsetsB, AABBHeap.dHistogramOffsetsB, AABBHeap.HistogramCount)) return 0;
    if (!ToHost(hChildHistogramOffsets, AABBHeap.dChildHistogramOffsets, AABBHeap.HistogramCount)) return 0;
    if (!ToHost(hIsBranches, AABBHeap.dIsBranches, AABBHeap.HistogramCount)) return 0;
    if (!ToHost(hIsLeaves, AABBHeap.dIsLeaves, AABBHeap.HistogramCount)) return 0;
    if (!ToHost(hBranchIndices, AABBHeap.dBranchIndices, AABBHeap.HistogramCount)) return 0;
    if (!ToHost(hLeafIndices, AABBHeap.dLeafIndices, AABBHeap.HistogramCount)) return 0;
    if (!ToHost(hAABBBranches, AABBHeap.dAABBBranches, AABBHeap.HistogramCount)) return 0;
    if (!ToHost(hAABBLeaves, AABBHeap.dAABBLeaves, AABBHeap.HistogramCount)) return 0;
    if (!ToHost(hAABBs, AABBHeap.dAABBs, AABBHeap.HistogramCount)) return 0;
    if (!ToHost(hTriangleHistogramIndices, AABBHeap.dTriangleHistogramIndices, AABBHeap.TriangleCount)) return 0;
    if (!ToHost(hTriangleHistogramPositions, AABBHeap.dTriangleHistogramPositions, AABBHeap.TriangleCount)) return 0;
    if (!ToHost(hReorderedIndicesA, AABBHeap.dReorderedIndicesA, AABBHeap.TriangleCount)) return 0;
    if (!ToHost(hReorderedIndicesB, AABBHeap.dReorderedIndicesB, AABBHeap.TriangleCount)) return 0;
    if (!ToHost(hTriangles, AABBHeap.dTriangles, AABBHeap.TriangleCount)) return 0;
    if (!ToHost(hBranchCounts, AABBHeap.dBranchCounts, AABBHeap.LevelCount)) return 0;
    if (!ToHost(hBranchOffsets, AABBHeap.dBranchOffsets, AABBHeap.LevelCount)) return 0;
    if (!ToHost(hLeafCounts, AABBHeap.dLeafCounts, AABBHeap.LevelCount)) return 0;
    if (!ToHost(hLeafOffsets, AABBHeap.dLeafOffsets, AABBHeap.LevelCount)) return 0;
    memcpy(hAABBHeader, AABBHeap.hAABBHeader, sizeof(AABBHeader_t));
    return 1;}
int FreePositions(Positions_t Positions){
    return Free(Positions.dData);}
int FreeIndices(Indices_t Indices){
    return Free(Indices.dData);}
int FreeEdges(Edges_t Edges){
    return Free(Edges.dData);}
int FreeNormals(Normals_t Normals){
    return Free(Normals.dData);}
int FreeVertexNormalsHeap(VertexNormalsHeap_t VertexNormalsHeap){
    if (!Free(VertexNormalsHeap.dTriangleReferences)) return 0;
    if (!Free(VertexNormalsHeap.dSharpEdgeReferences)) return 0;
    if (!Free(VertexNormalsHeap.dMaxTriangleReferences)) return 0;
    if (!Free(VertexNormalsHeap.dMaxSharpEdgeReferences)) return 0;
    if (!Free(VertexNormalsHeap.dNormalPtr)) return 0;
    return Free(VertexNormalsHeap.dOffset);}
int FreeAABBHeap(AABBHeap_t AABBHeap){
    if (!Free(AABBHeap.dTrianglePositions)) return 0;
    if (!Free(AABBHeap.dBlockAverages)) return 0;
    if (!Free(AABBHeap.dFinalAverages)) return 0;
    if (!Free(AABBHeap.dHistogramsA)) return 0;
    if (!Free(AABBHeap.dHistogramsB)) return 0;
    if (!Free(AABBHeap.dChildHistograms)) return 0;
    if (!Free(AABBHeap.dHistogramOffsetsA)) return 0;
    if (!Free(AABBHeap.dHistogramOffsetsB)) return 0;
    if (!Free(AABBHeap.dChildHistogramOffsets)) return 0;
    if (!Free(AABBHeap.dIsBranches)) return 0;
    if (!Free(AABBHeap.dIsLeaves)) return 0;
    if (!Free(AABBHeap.dBranchIndices)) return 0;
    if (!Free(AABBHeap.dLeafIndices)) return 0;
    if (!Free(AABBHeap.dAABBs)) return 0;
    if (!Free(AABBHeap.dAABBBranches)) return 0;
    if (!Free(AABBHeap.dAABBLeaves)) return 0;
    if (!Free(AABBHeap.dTriangleHistogramIndices)) return 0;
    if (!Free(AABBHeap.dTriangleHistogramPositions)) return 0;
    if (!Free(AABBHeap.dReorderedIndicesA)) return 0;
    if (!Free(AABBHeap.dReorderedIndicesB)) return 0;
    if (!Free(AABBHeap.dTriangles)) return 0;
    if (!Free(AABBHeap.dBranchCounts)) return 0;
    if (!Free(AABBHeap.dBranchOffsets)) return 0;
    if (!Free(AABBHeap.dLeafCounts)) return 0;
    if (!Free(AABBHeap.dLeafOffsets)) return 0;
    if (AABBHeap.hAABBHeader) free(AABBHeap.hAABBHeader);
    return 1;}
int CalculateTriangleNormals(Positions_t Positions, Indices_t Indices, Normals_t Normals){
    if (!Sync()) return 0;
    dim3 GridDim = dim3((Normals.Count - 1) / DefaultBlockDim.x + 1, 1, 1);
    CalculateTriangleNormalsKernel<<<GridDim, DefaultBlockDim>>>(Positions.dData, Indices.dData, Normals.dData, Normals.Count);
    if(!Sync(CalculateTriangleNormalsUsage(Normals.Count))) return 0;
    return 1;}
int CalculateVertexNormals(Indices_t PositionIndices, Normals_t TriangleNormals, Edges_t SharpEdges, Indices_t NormalIndices, Normals_t VertexNormals, VertexNormalsHeap_t Heap){
    if(!Sync()) return 0;
    // STEP 0: Initialize Linked List
    if(!Set(Heap.dTriangleReferences, Heap.TriangleReferenceCount, 0xFF)) return 0;
    if(!Set(Heap.dSharpEdgeReferences, Heap.SharpEdgeReferenceCount, 0xFF)) return 0;
    if(!Zero(Heap.dOffset, Heap.StackCount)) return 0;
    if(!Zero(Heap.dMaxTriangleReferences, 1)) return 0;
    if(!Zero(Heap.dMaxSharpEdgeReferences, 1)) return 0;
    if(!Zero(Heap.dNormalPtr, 1)) return 0;
    if(!Zero(NormalIndices.dData, NormalIndices.Count)) return 0;
    if(!Sync(sizeof(unsigned int) * (Heap.TriangleReferenceCount + Heap.SharpEdgeReferenceCount + Heap.StackCount) + sizeof(uint3) * NormalIndices.Count)) return 0;
    // STEP 1A: Make Linked List of triangles for each vertex
    {dim3 GridDim = dim3((PositionIndices.Count - 1) / DefaultBlockDim.x + 1, 1, 1);
    TrianglesToReferencesKernel<<<GridDim, DefaultBlockDim>>>(
        PositionIndices.dData,
        Heap.dTriangleReferences,
        Heap.dOffset,
        Heap.StackCount,
        PositionIndices.Count,
        Heap.VertexCount);
    if(!Sync(TriangleToReferencesUsage(PositionIndices.Count))) return 0;}
    // STEP 1B: Make Linked List of sharp edges for each vertex
    if (SharpEdges.Count > 0){
        if(!Zero(Heap.dOffset, Heap.StackCount)) return 0;
        if(!Sync(sizeof(unsigned int) * Heap.StackCount)) return 0;
        {dim3 GridDim = dim3((SharpEdges.Count - 1) / DefaultBlockDim.x + 1, 1, 1);
        SharpEdgesToReferencesKernel<<<GridDim, DefaultBlockDim>>>(
            SharpEdges.dData,
            Heap.dSharpEdgeReferences,
            Heap.dOffset,
            Heap.StackCount,
            SharpEdges.Count,
            Heap.VertexCount);
        if(!Sync(SharpEdgesToReferencesUsage(Heap.TriangleCount))) return 0;}}
    // STEP 2: For vertices find max triangle references and sharp edge references
    {dim3 BlockDim = dim3(MaxReferencesBlockSize, 1, 1);
    dim3 GridDim = dim3((Heap.VertexCount - 1) / MaxReferencesBlockSize + 1, 1, 1);
    bool UseSharpEdges = SharpEdges.Count > 0;
    MaxReferencesKernel<<<GridDim, BlockDim>>>(
        Heap.dTriangleReferences,
        UseSharpEdges,
        Heap.dSharpEdgeReferences,
        Heap.VertexCount,
        Heap.dMaxTriangleReferences,
        Heap.dMaxSharpEdgeReferences);
    if (!Sync(MaxReferencesUsage(Heap.VertexCount, Heap.TriangleCount, SharpEdges.Count))) return 0;}
    unsigned long long hMaxTriangleReferences = 0;
    unsigned long long hMaxSharpEdgeReferences = 0;
    if (!ToHost(&hMaxTriangleReferences, Heap.dMaxTriangleReferences, 1)) return 0;
    if (!Sync()) return 0;
    if(!ToHost(&hMaxSharpEdgeReferences, Heap.dMaxSharpEdgeReferences, 1)) return 0;
    if (!Sync()) return 0;
    // STEP 3: At each vertex organize triangle Linked Lists into groups using sharp edges
    // Calculate one normal per group and push result to a heap.
    {dim3 GridDim = dim3((Heap.VertexCount - 1) / DefaultBlockDim.x + 1, 1, 1);
    uint32_t SharedMemory = DefaultBlockDim.x * sizeof(uint32_t) * (5 * hMaxTriangleReferences + 2 * hMaxSharpEdgeReferences);
    VertexNormalsKernel<<<GridDim, DefaultBlockDim, SharedMemory>>>(
        Heap.dTriangleReferences,
        Heap.dSharpEdgeReferences,
        hMaxTriangleReferences,
        hMaxSharpEdgeReferences,
        Heap.VertexCount,
        TriangleNormals.dData,
        Heap.dNormalPtr,
        VertexNormals.dData,
        SharpEdges.dData,
        PositionIndices.dData,
        NormalIndices.dData);
    if(!Sync(ReferencesToVertexNormalsUsage(VertexNormals.Count))) return 0;
    return 1;}}
int BuildAABB(Positions_t Positions, Indices_t PositionIndices, Normals_t VertexNormals, Indices_t NormalIndices, AABBHeap_t AABBHeap){
    if(!Sync()) return 0;
    // STEP 0: Initialize
    if(!Zero(AABBHeap.dHistogramsA, AABBHeap.HistogramCount)) return 0;
    if(!Zero(AABBHeap.dHistogramOffsetsA, AABBHeap.HistogramCount)) return 0;
    if(!Zero(AABBHeap.dAABBBranches, AABBHeap.HistogramCount)) return 0;
    if(!Zero(AABBHeap.dAABBLeaves, AABBHeap.HistogramCount)) return 0;
    if(!Zero(AABBHeap.dAABBs, AABBHeap.HistogramCount)) return 0;
    const uint32_t TriangleCount = static_cast<uint32_t>(AABBHeap.TriangleCount);
    if(!E(cudaMemcpy(AABBHeap.dHistogramsA, &TriangleCount, sizeof(uint32_t), cudaMemcpyHostToDevice))) return 0;
    if(!Sync(AABBHeap.HistogramCount * (2 * sizeof(uint32_t) + sizeof(AABBBranch_t) + sizeof(AABBLeaf_t) + sizeof(AABB_t)))) return 0;

    {dim3 GridDim = dim3((AABBHeap.TriangleCount - 1) / DefaultBlockDim.x + 1, 1, 1);
    IotaKernel<<<GridDim, DefaultBlockDim>>>(AABBHeap.dReorderedIndicesA, 0, AABBHeap.TriangleCount);
    if (!Sync(IotaUsage(AABBHeap.TriangleCount))) return 0;}

    // STEP 1: Find triangle centers
    {dim3 GridDim = dim3((AABBHeap.TriangleCount - 1) / DefaultBlockDim.x + 1, 1, 1);
    TrianglePositionsKernel<<<GridDim, DefaultBlockDim>>>(
        Positions.dData,
        PositionIndices.dData,
        AABBHeap.dTrianglePositions,
        AABBHeap.TriangleCount);
    if (!Sync(TrianglePositionsUsage(AABBHeap.TriangleCount))) return 0;}

    // Step 2: Build AABBBranches, AABBLeaves, and AABBs
    uint32_t CurrentLevel = 0;
    vector<uint32_t> BranchCounts{};
    vector<uint32_t> BranchOffsets{};
    vector<uint32_t> BranchAABBOffsets{};
    vector<uint32_t> LeafCounts{};
    vector<uint32_t> LeafOffsets{};
    vector<uint32_t> LeafAABBOffsets{};
    unsigned int* dReorderedTriangleIndices = nullptr;
    BranchCounts.emplace_back(1);
    BranchOffsets.emplace_back(0);
    BranchAABBOffsets.emplace_back(0);
    LeafOffsets.emplace_back(0);
    LeafAABBOffsets.emplace_back(1);
    while(true){
        const uint32_t RemainingBins = BranchCounts[CurrentLevel];
        const uint32_t BlocksPerBin = (DefaultGridDim.x - 1) / RemainingBins + 1;
        const uint32_t RemainingChildren = RemainingBins * ChildCount;
        // STEP 2: Binning using a 2 page buffer
        unsigned int* dTriangleIndices = CurrentLevel % 2 == 0 ? AABBHeap.dReorderedIndicesA : AABBHeap.dReorderedIndicesB;
        dReorderedTriangleIndices = CurrentLevel % 2 == 0 ? AABBHeap.dReorderedIndicesB : AABBHeap.dReorderedIndicesA;
        unsigned int* dHistograms = CurrentLevel % 2 == 0 ? AABBHeap.dHistogramsA : AABBHeap.dHistogramsB;
        unsigned int* dHistogramOffsets = CurrentLevel % 2 == 0 ? AABBHeap.dHistogramOffsetsA : AABBHeap.dHistogramOffsetsB;
        unsigned int* dNextHistograms = CurrentLevel % 2 == 0 ? AABBHeap.dHistogramsB : AABBHeap.dHistogramsA;
        unsigned int* dNextHistogramOffsets = CurrentLevel % 2 == 0 ? AABBHeap.dHistogramOffsetsB : AABBHeap.dHistogramOffsetsA;
        if(!Zero(dNextHistograms, AABBHeap.HistogramCount)) return 0;
        if(!Zero(dNextHistogramOffsets, AABBHeap.HistogramCount)) return 0;
        if(!Zero(AABBHeap.dChildHistograms, AABBHeap.HistogramCount)) return 0;
        if(!Zero(AABBHeap.dChildHistogramOffsets, AABBHeap.HistogramCount)) return 0;
        if(!DeviceCopy(dReorderedTriangleIndices, dTriangleIndices, AABBHeap.TriangleCount)) return 0;

        // STEP 2A: Find average position
        float3* dAverages = AABBHeap.dBlockAverages;
        if(!Zero(AABBHeap.dBlockAverages, AABBHeap.HistogramCount)) return 0;
        if(!Sync(sizeof(unsigned int) * (4 * AABBHeap.HistogramCount + AABBHeap.TriangleCount))) return 0;

        {const dim3 GridDim = dim3(BlocksPerBin, RemainingBins, 1);
        ParallelAverageKernel<<<GridDim, DefaultBlockDim>>>(
            dHistogramOffsets,
            dHistograms,
            dTriangleIndices,
            AABBHeap.dTrianglePositions,
            AABBHeap.dBlockAverages);
        if (!Sync(ParallelAverageUsage(AABBHeap.TriangleCount, GridDim, DefaultBlockDim))) return 0;
        if (RemainingBins < 256){
            dAverages = AABBHeap.dFinalAverages;
            if(!Zero(AABBHeap.dFinalAverages, AABBHeap.FinalAverageCount)) return 0;
            if(!Sync(sizeof(unsigned int) * AABBHeap.FinalAverageCount)) return 0;
            const dim3 FinalGridDim = dim3(RemainingBins, 1, 1);
            const dim3 FinalBlockDim = dim3(BlocksPerBin, 1, 1);
            FinalAverageKernel<<<FinalGridDim, FinalBlockDim, BlocksPerBin * sizeof(float3)>>>(AABBHeap.dBlockAverages, AABBHeap.dFinalAverages);
            if (!Sync(FinalAverageUsage(FinalGridDim, FinalBlockDim))) return 0;}}

        // STEP 2B: Put each triangle in histogram
        {const dim3 GridDim = dim3(BlocksPerBin, RemainingBins, 1);
        TriangleToHistogramKernel<<<GridDim, DefaultBlockDim>>>(
            dAverages,
            dHistogramOffsets,
            dHistograms,
            dTriangleIndices,
            AABBHeap.dTrianglePositions,
            AABBHeap.dChildHistograms,
            AABBHeap.dTriangleHistogramIndices,
            AABBHeap.dTriangleHistogramPositions,
            AABBHeap.TriangleCount);
        if (!Sync(TriangleToHistogramUsage(AABBHeap.TriangleCount, GridDim, DefaultBlockDim))) return 0;}

        {dim3 GridDim = dim3(1, RemainingBins, 1);
        dim3 BlockDim = dim3(8, 1, 1);
        HistogramOffsetsKernel<<<GridDim, BlockDim>>>(dHistogramOffsets, AABBHeap.dChildHistograms, AABBHeap.dChildHistogramOffsets);
        if (!Sync(HistogramOffsetsUsage(RemainingBins))) return 0;}

        // STEP 2C: Is bin branch or leaf (or empty)
        {const dim3 GridDim = dim3((RemainingChildren - 1) / DefaultBlockDim.x + 1, 1, 1);
        HistogramBranchOrLeafKernel<<<GridDim, DefaultBlockDim>>>(
            AABBHeap.dChildHistograms,
            AABBHeap.dIsBranches,
            AABBHeap.dIsLeaves,
            RemainingChildren);
        if (!Sync(HistogramBranchOrLeafUsage(RemainingChildren))) return 0;}

        // STEP 2D: Find Histogram Offsets and empties removed branch index and leaf index
        const uint32_t Orders = ceil(log(RemainingChildren) / log(1024));
        for (uint32_t Order = 0; Order < Orders; Order++){
            uint32_t LevelHistogramCount = (RemainingChildren - 1 >> Order * 10) + 1;
            dim3 BlockDim = dim3(1024, 1, 1);
            dim3 GridDim = dim3((LevelHistogramCount - 1) / BlockDim.x + 1, 1, 1);
            UpBranchLeafIndicesKernel<<<GridDim, BlockDim>>>(
                AABBHeap.dIsBranches,
                AABBHeap.dIsLeaves,
                AABBHeap.dBranchIndices,
                AABBHeap.dLeafIndices,
                Order,
                RemainingChildren);
            if(!Sync(UpBranchLeafIndicesUsage(Order, RemainingChildren))) return 0;}
        if (Orders > 1){
            for (int32_t Order = Orders - 1; Order >= 0; Order--){
                uint32_t LevelHistogramCount = (AABBHeap.HistogramCount - 1 >> Order * 10) + 1;
                dim3 BlockDim = dim3(1024, 1, 1);
                dim3 GridDim = dim3((LevelHistogramCount - 1) / BlockDim.x + 1, 1, 1);
                DownBranchLeafIndicesKernel<<<GridDim, dim3(1024, 1, 1)>>>(
                    AABBHeap.dBranchIndices,
                    AABBHeap.dLeafIndices,
                    Order,
                    RemainingChildren);
                if(!Sync(DownBranchLeafIndicesUsage(Order, AABBHeap.HistogramCount))) return 0;}}

        // STEP 2E: Reorder triangles with histogram offsets
        {const dim3 GridDim = dim3(BlocksPerBin, RemainingBins, 1);
        ReorderTrianglesKernel<<<GridDim, DefaultBlockDim>>>(
            dHistogramOffsets,
            dHistograms,
            AABBHeap.dChildHistogramOffsets,
            dTriangleIndices,
            AABBHeap.dTriangleHistogramIndices,
            AABBHeap.dTriangleHistogramPositions,
            dReorderedTriangleIndices,
            AABBHeap.TriangleCount);
        if(!Sync(ReorderTrianglesUsage(AABBHeap.TriangleCount))) return 0;}

        // STEP 2F: reorder histograms with branch indices
        {const dim3 GridDim = dim3((RemainingChildren - 1) / DefaultBlockDim.x + 1, 1, 1);
        MakeNextHistogramsKernel<<<GridDim, DefaultBlockDim>>>(
            AABBHeap.dIsBranches,
            AABBHeap.dBranchIndices,
            AABBHeap.dChildHistograms,
            AABBHeap.dChildHistogramOffsets,
            dNextHistograms,
            dNextHistogramOffsets,
            RemainingChildren);
        if(!Sync(MakeNextHistogramsUsage(RemainingChildren))) return 0;}

        // STEP 2G: Pack branches, and leaves
        // Branches are appended to the end of of dAABBBranches and reference unique AABB
        // Leaves are appened to the end of dAABBLeaves and reference a unique AABB
        // AABBBranches reference first child AABBBranch and count
        // AABBLeaves reference first child Triangle and count
        {const dim3 GridDim = dim3((RemainingBins - 1) / DefaultBlockDim.x + 1, 1, 1);
        PackAABBKernel<<<GridDim, DefaultBlockDim>>>(
            AABBHeap.dChildHistograms,
            AABBHeap.dChildHistogramOffsets,
            AABBHeap.dIsBranches,
            AABBHeap.dIsLeaves,
            AABBHeap.dBranchIndices,
            AABBHeap.dLeafIndices,
            AABBHeap.dAABBBranches,
            AABBHeap.dAABBLeaves,
            RemainingBins,
            BranchAABBOffsets[CurrentLevel],
            LeafAABBOffsets[CurrentLevel],
            BranchOffsets[CurrentLevel],
            LeafOffsets[CurrentLevel]);
        if(!Sync(PackAABBUsage(RemainingBins))) return 0;}

        uint32_t LeafCount;
        uint32_t BranchCount;
        if (!E(cudaMemcpy(&LeafCount, AABBHeap.dLeafIndices + RemainingChildren - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost))) return 0;
        if (!E(cudaMemcpy(&BranchCount, AABBHeap.dBranchIndices + RemainingChildren - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost))) return 0;
        if (!Sync(2 * sizeof(uint32_t))) return 0;
        LeafCounts.emplace_back(LeafCount);
        if (BranchCount == 0)
            break;
        BranchCounts.emplace_back(BranchCount);
        BranchOffsets.emplace_back(BranchOffsets[CurrentLevel] + RemainingBins);
        LeafOffsets.emplace_back(LeafOffsets[CurrentLevel] + LeafCount);
        BranchAABBOffsets.emplace_back(LeafAABBOffsets[CurrentLevel] + LeafCount);
        LeafAABBOffsets.emplace_back(BranchAABBOffsets[CurrentLevel + 1] + BranchCount);
        CurrentLevel++;
        if (CurrentLevel == AABBHeap.LevelCount) return 0;}
    if (!E(cudaMemcpy(AABBHeap.dBranchCounts, BranchCounts.data(), BranchCounts.size() * sizeof(BranchCounts[0]), cudaMemcpyHostToDevice))) return 0;
    if (!E(cudaMemcpy(AABBHeap.dBranchOffsets, BranchOffsets.data(), BranchOffsets.size() * sizeof(BranchOffsets[0]), cudaMemcpyHostToDevice))) return 0;
    if (!E(cudaMemcpy(AABBHeap.dLeafCounts, LeafCounts.data(), LeafCounts.size() * sizeof(LeafCounts[0]), cudaMemcpyHostToDevice))) return 0;
    if (!E(cudaMemcpy(AABBHeap.dLeafOffsets, LeafOffsets.data(), LeafOffsets.size() * sizeof(LeafOffsets[0]), cudaMemcpyHostToDevice))) return 0;
    // Step 3 Reorder Triangles to match AABBLeaves layout.
    {dim3 GridDim = dim3((PositionIndices.Count - 1) / DefaultBlockDim.x + 1, 1, 1);
    CalculateTrianglesKernel<<<GridDim, DefaultBlockDim>>>(
        dReorderedTriangleIndices,
        PositionIndices.dData,
        Positions.dData,
        NormalIndices.dData,
        VertexNormals.dData,
        AABBHeap.dTriangles,
        PositionIndices.Count);
    if (!Sync(CalculateTrianglesUsage(PositionIndices.Count)));}
    // Step 4 Find Min and Max for Leaves
    {const dim3 GridDim = dim3(CurrentLevel + 1, 1, 1);
    LeafMinMaxKernel<<<GridDim, DefaultBlockDim>>>(
        AABBHeap.dAABBLeaves,
        AABBHeap.dTriangles,
        AABBHeap.dLeafOffsets,
        AABBHeap.dLeafCounts,
        AABBHeap.dAABBs);
    const uint32_t LeafAccumulate = accumulate(LeafCounts.begin(), LeafCounts.end(), 0);
    if (!Sync(LeafMinMaxUsage(LeafAccumulate, TriangleCount))) return 0;}
    // Step 5 Find min and max from the branches from the bottom level then going up.
    for (int32_t Level = CurrentLevel; Level >= 0; Level--){
        const dim3 GridDim = dim3((BranchCounts[Level] - 1) / DefaultBlockDim.x + 1, 1, 1);
        BranchMinMaxKernel<<<GridDim, DefaultBlockDim>>>(
            AABBHeap.dAABBBranches,
            AABBHeap.dAABBLeaves,
            Level,
            AABBHeap.dBranchCounts,
            AABBHeap.dBranchOffsets,
            AABBHeap.dAABBs);
        const uint32_t BranchCount = BranchCounts[Level];
        const uint32_t NextBranchCount = Level == CurrentLevel ? 0 : BranchCounts[Level + 1];
        const uint32_t LeafCount = LeafCounts[Level];
        if (!Sync(BranchMinMaxUsage(BranchCounts[Level], NextBranchCount, LeafCount))) return 1;}
    AABBHeap.hAABBHeader->BranchCount = BranchOffsets[CurrentLevel] + BranchCounts[CurrentLevel];
    AABBHeap.hAABBHeader->LeafCount = LeafOffsets[CurrentLevel] + LeafCounts[CurrentLevel];
    AABBHeap.hAABBHeader->AABBCount = LeafAABBOffsets[CurrentLevel] + LeafCounts[CurrentLevel];
    AABBHeap.hAABBHeader->MaxDepth = CurrentLevel;
    AABBHeap.hAABBHeader->TriangleCount = PositionIndices.Count;
    return 1;}}
