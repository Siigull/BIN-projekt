#ifndef METAL_EVAL_BRIDGE_H
#define METAL_EVAL_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#include "SharedDefinitions.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MetalEvalContext MetalEvalContext;

MetalEvalContext* metal_eval_create(const char* shader_path, char* err, size_t err_size);
bool metal_eval_evaluate(
    MetalEvalContext* ctx,
    const MetalChromosome* population,
    uint32_t population_size,
    const uint32_t* tdata,
    const uint32_t* valid_masks,
    uint32_t param_in,
    uint32_t param_out,
    uint32_t data_blocks,
    uint32_t* out_fitness,
    char* err,
    size_t err_size
);
void metal_eval_destroy(MetalEvalContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
