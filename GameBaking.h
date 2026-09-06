#pragma once
#include <vector_types.h>
#include <cinttypes>

constexpr uint64_t ChildCount = 8U;

typedef struct {
    float3* dData = nullptr;
    uint64_t Count = 0U;
} Positions_t;
typedef struct {
    uint3* dData = nullptr;
    uint64_t Count = 0U;
} Indices_t;
typedef struct {
    uint2* dData = nullptr;
    uint64_t Count = 0u;
} Edges_t;
typedef struct {
    float3* dData = nullptr;
    uint64_t Count = 0U;
} Normals_t;
typedef struct {
    unsigned long long* dTriangleReferences = nullptr;
    unsigned long long* dSharpEdgeReferences = nullptr;
    unsigned int* dOffset = nullptr;
    unsigned long long* dMaxTriangleReferences = nullptr;
    unsigned long long* dMaxSharpEdgeReferences = nullptr;
    unsigned int* dNormalPtr = nullptr;
    uint64_t TriangleCount = 0U;
    uint64_t VertexCount = 0U;
    uint64_t ConservativeNormalCount = 0U;
    uint64_t StackCount = 0U;
    uint64_t TriangleReferenceCount = 0U;
    uint64_t SharpEdgeReferenceCount = 0U;
} VertexNormalsHeap_t;
typedef struct {
    uint32_t BranchCount = 0U;
    uint32_t BranchOffset = 0U;
    uint32_t LeafCount = 0U;
    uint32_t LeafOffset = 0U;
    uint32_t BoxIndex = 0U;
} AABBBranch_t;
typedef struct {
    uint32_t TriangleCount = 0U;
    uint32_t TriangleOffset = 0U;
    uint32_t BoxIndex = 0U;
} AABBLeaf_t;
typedef struct {
    float4 Min = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 Max = {0.0f, 0.0f, 0.0f, 0.0f};
} AABB_t;
typedef struct {
    float4 V0 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 Edge0 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 Edge1 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 Normal0 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 Normal1 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 Normal2 = {0.0f, 0.0f, 0.0f, 0.0f};
} Triangle_t;
typedef struct {
    Triangle_t* dData = nullptr;
    uint64_t Count = 0U;
} Triangles_t;
typedef struct {
    uint64_t BranchCount = 0U;
    uint64_t LeafCount = 0U;
    uint64_t AABBCount = 0U;
    uint64_t MaxDepth = 0U;
    uint64_t TriangleCount = 0U;
} AABBHeader_t;
typedef struct {
    float3* dTrianglePositions = nullptr;
    float3* dBlockAverages = nullptr;
    float3* dFinalAverages = nullptr;
    unsigned int* dHistogramsA = nullptr;
    unsigned int* dHistogramsB = nullptr;
    unsigned int* dChildHistograms = nullptr;
    unsigned int* dHistogramOffsetsA = nullptr;
    unsigned int* dHistogramOffsetsB = nullptr;
    unsigned int* dChildHistogramOffsets = nullptr;
    bool* dIsBranches = nullptr;
    bool* dIsLeaves = nullptr;
    unsigned int* dBranchIndices = nullptr;
    unsigned int* dLeafIndices = nullptr;
    AABBBranch_t* dAABBBranches = nullptr;
    AABBLeaf_t* dAABBLeaves = nullptr;
    AABB_t* dAABBs = nullptr;
    unsigned int* dTriangleHistogramIndices = nullptr;
    unsigned int* dTriangleHistogramPositions = nullptr;
    unsigned int* dReorderedIndicesA = nullptr;
    unsigned int* dReorderedIndicesB = nullptr;
    Triangle_t* dTriangles = nullptr;
    uint32_t* dBranchCounts = nullptr;
    uint32_t* dBranchOffsets = nullptr;
    uint32_t* dLeafCounts = nullptr;
    uint32_t* dLeafOffsets = nullptr;
    AABBHeader_t* hAABBHeader = nullptr;
    uint64_t FinalAverageCount = 0U;
    uint64_t HistogramCount = 0U;
    uint64_t TriangleCount = 0U;
    uint64_t LevelCount = 0U;
} AABBHeap_t;

extern "C"{
    Positions_t MallocPositions(uint64_t Count);
    Indices_t MallocIndices(uint64_t Count);
    Edges_t MallocEdges(uint64_t Count);
    Normals_t MallocNormals(uint64_t Count);
    VertexNormalsHeap_t MallocVertexNormalsHeap(uint64_t TriangleCount, uint64_t VertexCount, uint64_t SharpEdgeCount, uint64_t StackCount);
    AABBHeap_t MallocAABBHeap(uint64_t TriangleCount);

    int ToDevicePositions(Positions_t Positions, float3* hPositions);
    int ToDeviceIndices(Indices_t Indices, uint3* hIndices);
    int ToDeviceEdges(Edges_t Edges, uint2* hEdges);
    int ToDeviceNormals(Normals_t Normals, float3* hNormals);

    int ToHostPositions(float3* hPositions, Positions_t Positions);
    int ToHostIndices(uint3* hIndices, Indices_t Indices);
    int ToHostEdges(uint2* hEdges, Edges_t Edges);
    int ToHostNormals(float3* hNormals, Normals_t Normals);
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
        uint32_t* hTriangleHistogramsIndices,
        uint32_t* hTriangleHistogramsPositions,
        unsigned int* hReorderedIndicesA,
        unsigned int* hReorderedIndicesB,
        Triangle_t* hTriangles,
        uint32_t* hBranchCounts,
        uint32_t* hBranchOffsetsCounts,
        uint32_t* hLeafCounts,
        uint32_t* hLeafOffsets,
        AABBHeader_t* AABBHeader,
        AABBHeap_t VertexNormalsHeap);

    int FreePositions(Positions_t Data);
    int FreeIndices(Indices_t Indices);
    int FreeEdges(Edges_t Edges);
    int FreeNormals(Normals_t Normal);
    int FreeVertexNormalsHeap(VertexNormalsHeap_t VertexNormalsHeap);
    int FreeAABBHeap(AABBHeap_t AABBHeap);
    
    // Real Functions
    int CalculateTriangleNormals(Positions_t Positions, Indices_t PositionIndices, Normals_t Normals);
    int CalculateVertexNormals(Indices_t PositionIndices, Normals_t TriangleNormals, Edges_t SharpEdges, Indices_t NormalIndices, Normals_t VertexNormals, VertexNormalsHeap_t Heap);
    int BuildAABB(Positions_t Positions, Indices_t PositionIndices, Normals_t VertexNormals, Indices_t NormalIndices, AABBHeap_t Heap);}
