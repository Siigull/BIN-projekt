#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <string.h>

#include "metal_eval_bridge.h"

struct MetalEvalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
};

static void write_err(char* err, size_t err_size, const char* msg) {
    if (!err || err_size == 0) return;
    snprintf(err, err_size, "%s", msg ? msg : "Unknown error");
}

MetalEvalContext* metal_eval_create(const char* shader_path, char* err, size_t err_size) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            write_err(err, err_size, "Metal device not available");
            return nullptr;
        }

        if (!shader_path) {
            write_err(err, err_size, "Shader path is null");
            return nullptr;
        }

        NSString* shaderPath = [NSString stringWithUTF8String:shader_path];
        NSError* fileError = nil;
        NSString* source = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&fileError];
        if (!source) {
            NSString* msg = [NSString stringWithFormat:@"Failed to read shader %s: %@", shader_path, fileError.localizedDescription];
            write_err(err, err_size, msg.UTF8String);
            return nullptr;
        }

        NSError* libError = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&libError];
        if (!library) {
            NSString* msg = [NSString stringWithFormat:@"Failed to compile Metal shader: %@", libError.localizedDescription];
            write_err(err, err_size, msg.UTF8String);
            return nullptr;
        }

        id<MTLFunction> fn = [library newFunctionWithName:@"evaluate_cgp"];
        if (!fn) {
            write_err(err, err_size, "Kernel evaluate_cgp not found");
            return nullptr;
        }

        NSError* pipelineError = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&pipelineError];
        if (!pipeline) {
            NSString* msg = [NSString stringWithFormat:@"Failed to create compute pipeline: %@", pipelineError.localizedDescription];
            write_err(err, err_size, msg.UTF8String);
            return nullptr;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            write_err(err, err_size, "Failed to create Metal command queue");
            return nullptr;
        }

        MetalEvalContext* ctx = new MetalEvalContext();
        ctx->device = device;
        ctx->queue = queue;
        ctx->pipeline = pipeline;
        return ctx;
    }
}

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
) {
    @autoreleasepool {
        if (!ctx || !population || !tdata || !valid_masks || !out_fitness) {
            write_err(err, err_size, "Invalid null pointer argument");
            return false;
        }

        const size_t popBytes = (size_t)population_size * sizeof(MetalChromosome);
        const size_t tdataBytes = (size_t)data_blocks * (size_t)(param_in + param_out) * sizeof(uint32_t);
        const size_t maskBytes = (size_t)data_blocks * sizeof(uint32_t);
        const size_t fitBytes = (size_t)population_size * sizeof(uint32_t);

        id<MTLBuffer> popBuffer = [ctx->device newBufferWithBytes:population length:popBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> tdataBuffer = [ctx->device newBufferWithBytes:tdata length:tdataBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> maskBuffer = [ctx->device newBufferWithBytes:valid_masks length:maskBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> fitBuffer = [ctx->device newBufferWithLength:fitBytes options:MTLResourceStorageModeShared];
        if (!popBuffer || !tdataBuffer || !maskBuffer || !fitBuffer) {
            write_err(err, err_size, "Failed to create Metal buffers");
            return false;
        }

        memset(fitBuffer.contents, 0, fitBytes);

        id<MTLCommandBuffer> commandBuffer = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!commandBuffer || !encoder) {
            write_err(err, err_size, "Failed to create command buffer/encoder");
            return false;
        }

        [encoder setComputePipelineState:ctx->pipeline];
        [encoder setBuffer:popBuffer offset:0 atIndex:0];
        [encoder setBuffer:tdataBuffer offset:0 atIndex:1];
        [encoder setBuffer:maskBuffer offset:0 atIndex:2];
        [encoder setBuffer:fitBuffer offset:0 atIndex:3];
        [encoder setBytes:&param_in length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&param_out length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&data_blocks length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&population_size length:sizeof(uint32_t) atIndex:7];

        NSUInteger totalThreads = (NSUInteger)population_size * (NSUInteger)data_blocks;
        if (totalThreads == 0) {
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            return true;
        }

        NSUInteger width = ctx->pipeline.maxTotalThreadsPerThreadgroup;
        if (width > 256) width = 256;
        if (width == 0) width = 1;

        MTLSize grid = MTLSizeMake(totalThreads, 1, 1);
        MTLSize group = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:group];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            NSString* msg = [NSString stringWithFormat:@"Metal command failed: %@", commandBuffer.error.localizedDescription];
            write_err(err, err_size, msg.UTF8String);
            return false;
        }

        memcpy(out_fitness, fitBuffer.contents, fitBytes);
        return true;
    }
}

void metal_eval_destroy(MetalEvalContext* ctx) {
    delete ctx;
}
