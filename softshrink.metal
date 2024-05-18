#include <metal_stdlib>
using namespace metal;

// SoftShrinkage(x) = x - lambda, if x > lambda
//                    x + lambda, if x < -lambda
//                    0,          otherwise
template<typename T>
kernel void softshrink_kernel(constant T*     input  [[buffer(0)]],
                              device   T*     output [[buffer(1)]],
                              constant float& lambda [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    output[index] = input[index] >  lambda ? input[index] - lambda :
                    input[index] < -lambda ? input[index] + lambda : 0;
}

template
[[host_name("softshrink_kernel_half")]]
kernel void softshrink_kernel<half>(constant half*  input  [[buffer(0)]],
                                    device   half*  output [[buffer(1)]],
                                    constant float& lambda [[buffer(2)]],
                                    uint index [[thread_position_in_grid]]);

template
[[host_name("softshrink_kernel_float")]]
kernel void softshrink_kernel<float>(constant float*  input  [[buffer(0)]],
                                     device   float*  output [[buffer(1)]],
                                     constant float& lambda  [[buffer(2)]],
                                     uint index [[thread_position_in_grid]]);


// from ggml-metal.metal 

constant float GELU_COEF_A     = 0.044715f;
constant float GELU_QUICK_COEF = -1.702f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

[[host_name("gelu_kernel")]]
kernel void kernel_gelu(
    device const float4 * src0 [[buffer(0)]],
    device       float4 * dst [[buffer(1)]],
    uint tpig [[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    // BEWARE !!!
    // Simply using "tanh" instead of "precise::tanh" will sometimes results in NaNs!
    // This was observed with Falcon 7B and 40B models
    //
    dst[tpig] = 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}


[[host_name("relu_kernel")]]
kernel void kernel_relu(
    device const float * src0 [[buffer(0)]],
    device       float * dst [[buffer(1)]],
    uint tpig [[thread_position_in_grid]]) {
    device const float & x = src0[tpig];
    
    dst[tpig] = max(x, 0.0f);
}
