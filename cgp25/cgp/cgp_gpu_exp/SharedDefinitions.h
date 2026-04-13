#ifndef SharedDefinitions_h
#define SharedDefinitions_h

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define SHARED_U32 unsigned int
#else
#include <cstdint>
#define SHARED_U32 uint32_t
#endif

#define METAL_MAX_OUTPUTS 16
#define METAL_MAX_ACTIVE_NODES 500
#define METAL_MAX_VYSTUPY 1024

struct MetalNode {
    SHARED_U32 in1;
    SHARED_U32 in2;
    SHARED_U32 fce;
    SHARED_U32 out_idx;
};

struct MetalChromosome {
    SHARED_U32 active_count;
    SHARED_U32 out_indices[METAL_MAX_OUTPUTS];
    MetalNode nodes[METAL_MAX_ACTIVE_NODES];
};

#undef SHARED_U32

#endif