#include <metal_stdlib>
#include "SharedDefinitions.h"

using namespace metal;

// The kernel function executed by the GPU
kernel void evaluate_cgp(
    device const MetalChromosome* population [[buffer(0)]],
    device const uint* tdata                 [[buffer(1)]],
    device const uint* valid_masks           [[buffer(2)]],
    device atomic_uint* fitness_scores       [[buffer(3)]],
    constant uint& param_in                  [[buffer(4)]],
    constant uint& param_out                 [[buffer(5)]],
    constant uint& data_blocks               [[buffer(6)]],
    constant uint& population_size           [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total_threads = population_size * data_blocks;
    if (gid >= total_threads) return;

    uint chrom_idx = gid / data_blocks;
    uint data_idx = gid % data_blocks;

    device const MetalChromosome& chrom = population[chrom_idx];

    uint total_vars = param_in + param_out;
    thread uint vystupy[METAL_MAX_VYSTUPY];

    uint batch_offset = data_idx * total_vars;
    for (uint i = 0; i < param_in; i++) {
        vystupy[i] = tdata[batch_offset + i];
    }

    for (uint i = 0; i < chrom.active_count; i++) {
        MetalNode node = chrom.nodes[i];
        uint in1 = vystupy[node.in1];
        uint in2 = vystupy[node.in2];
        uint res = 0;

        switch (node.fce) {
            case 0: res = in1; break;
            case 1: res = in1 & in2; break;
            case 2: res = in1 | in2; break;
            case 3: res = in1 ^ in2; break;
            case 4: res = ~in1; break;
            case 5: res = ~in2; break;
            case 6: res = in1 & (~in2); break;
            case 7: res = ~(in1 & in2); break;
            case 8: res = ~(in1 | in2); break;
        }
        vystupy[node.out_idx] = res;
    }

    uint total_correct = 0;
    uint current_mask = valid_masks[data_idx];
    device const uint* expected_outputs = tdata + batch_offset + param_in;

    for (uint i = 0; i < param_out; i++) {
        uint matched = ~(vystupy[chrom.out_indices[i]] ^ expected_outputs[i]);
        matched &= current_mask;
        total_correct += popcount(matched);
    }

    atomic_fetch_add_explicit(&fitness_scores[chrom_idx], total_correct, memory_order_relaxed);
}
