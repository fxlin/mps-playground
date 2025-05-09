/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The code that registers a PyTorch custom operation.
*/


#include <torch/extension.h>
#include "CustomSoftshrink.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

torch::Tensor& dispatchSoftShrinkKernel(const torch::Tensor& input, torch::Tensor& output, float lambda) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Set the number of threads equal to the number of elements within the input tensor.
        int numThreads = input.numel();

        // xzl
        NSString * sourcePath = @"softshrink.metal";
        // xzl: below, 1st argument name omitted???
        NSString * src = [NSString stringWithContentsOfFile:sourcePath encoding:NSUTF8StringEncoding error:&error]; 
        if (error) {
            printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
        }

        // Load the custom soft shrink shader.
        id<MTLLibrary> myKernelLibrary = [device newLibraryWithSource:src
                                                                  options:nil
                                                                    error:&error];
        TORCH_CHECK(myKernelLibrary, "Failed to to create my kernel library, error: ", error.localizedDescription.UTF8String);                                                                    

        // xzl
        std::string kernel_name = std::string("softshrink_kernel_") + (input.scalar_type() == torch::kFloat ? "float" : "half");
        id<MTLFunction> mySoftShrinkFunction = [myKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
        TORCH_CHECK(mySoftShrinkFunction, "Failed to create function state object for ", kernel_name.c_str());

        // xzl 
        // Create a compute pipeline state object for the soft shrink kernel.
        id<MTLComputePipelineState> mysoftShrinkPSO = [device newComputePipelineStateWithFunction:mySoftShrinkFunction error:&error];
        TORCH_CHECK(mysoftShrinkPSO, error.localizedDescription.UTF8String);

        // Get a reference to the command buffer for the MPS stream.
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        // xzl 
        dispatch_sync(serialQueue, ^(){
            // Start a compute pass.
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

            // Encode the pipeline state object and its parameters.
            [computeEncoder setComputePipelineState:mysoftShrinkPSO];
            [computeEncoder setBuffer:getMTLBufferStorage(input) offset:input.storage_offset() * input.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:1];
            [computeEncoder setBytes:&lambda length:sizeof(float) atIndex:2];

            MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

            // Calculate a thread group size.
            NSUInteger threadGroupSize = mysoftShrinkPSO.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > numThreads) {
                threadGroupSize = numThreads;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            // Encode the compute command.
            [computeEncoder dispatchThreads:gridSize
                      threadsPerThreadgroup:threadgroupSize];

            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }

    return output;
}

// C++ op dispatching the Metal soft shrink shader.
torch::Tensor mps_softshrink(const torch::Tensor &input, float lambda = 0.5) {
    // Check whether the input tensor resides on the MPS device and whether it's contiguous.
    TORCH_CHECK(input.device().is_mps(), "input must be a MPS tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    // Check the supported data types for soft shrink.
    TORCH_CHECK(input.scalar_type() == torch::kFloat ||
                input.scalar_type() == torch::kHalf, "Unsupported data type: ", input.scalar_type());

    // Allocate the output, same shape as the input.
    torch::Tensor output = torch::empty_like(input);

    return dispatchSoftShrinkKernel(input, output, lambda);
}


torch::Tensor& dispatchGeluKernel(const torch::Tensor& input, torch::Tensor& output) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Set the number of threads equal to the number of elements within the input tensor.
        // int numThreads = input.numel()/4;   // float4
        int numThreads = input.numel();   // float4

        // xzl
        NSString * sourcePath = @"softshrink.metal";
        // xzl: below, 1st argument name omitted???
        NSString * src = [NSString stringWithContentsOfFile:sourcePath encoding:NSUTF8StringEncoding error:&error]; 
        if (error) {
            printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
        }

        // Load the custom soft shrink shader.
        id<MTLLibrary> myKernelLibrary = [device newLibraryWithSource:src
                                                                  options:nil
                                                                    error:&error];
        TORCH_CHECK(myKernelLibrary, "Failed to to create my kernel library, error: ", error.localizedDescription.UTF8String);                                                                    

        // xzl
        // id<MTLFunction> mySoftShrinkFunction = [myKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:"gelu_kernel"]];
        id<MTLFunction> mySoftShrinkFunction = [myKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:"relu_kernel"]];
        TORCH_CHECK(mySoftShrinkFunction, "Failed to create function state object for gelu");

        // xzl 
        // Create a compute pipeline state object for the soft shrink kernel.
        id<MTLComputePipelineState> mysoftShrinkPSO = [device newComputePipelineStateWithFunction:mySoftShrinkFunction error:&error];
        TORCH_CHECK(mysoftShrinkPSO, error.localizedDescription.UTF8String);

        // Get a reference to the command buffer for the MPS stream.
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        // xzl 
        dispatch_sync(serialQueue, ^(){
            // Start a compute pass.
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

            // Encode the pipeline state object and its parameters.
            [computeEncoder setComputePipelineState:mysoftShrinkPSO];
            [computeEncoder setBuffer:getMTLBufferStorage(input) offset:input.storage_offset() * input.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:1];

            MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

            // Calculate a thread group size.
            NSUInteger threadGroupSize = mysoftShrinkPSO.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > numThreads) {
                threadGroupSize = numThreads;
            }
            // MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
            MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);

            // Encode the compute command.
            [computeEncoder dispatchThreads:gridSize
                      threadsPerThreadgroup:threadgroupSize];

            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }

    return output;
}

// C++ op dispatching the Metal soft shrink shader.
torch::Tensor mps_gelu(const torch::Tensor &input) {
    // Check whether the input tensor resides on the MPS device and whether it's contiguous.
    TORCH_CHECK(input.device().is_mps(), "input must be a MPS tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    // Allocate the output, same shape as the input.
    torch::Tensor output = torch::empty_like(input);

    return dispatchGeluKernel(input, output);
}


// Create Python bindings for the Objective-C++ code.
// xzl ... so that python can find these funcs in "compiled" metal
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mps_softshrink", &mps_softshrink);
    m.def("mps_gelu", &mps_gelu);
}
