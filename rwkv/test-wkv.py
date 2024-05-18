'''
Copyright © 2023 Apple Inc.

See LICENSE folder for this sample’s licensing information.

Abstract:
The code for compiling the custom pytorch extension.
'''

import torch
from torch import nn
import torch.utils.cpp_extension

HEAD_SIZE=64

# wraps native code as a py module(?
wkv5_metal = torch.utils.cpp_extension.load(
    name='wkv5',        # xzl: useful??
    sources=['wkv5_op.mm'],
    # verbose=True, 
    extra_cflags=['-std=c++17', f"-D_N_={HEAD_SIZE}"],
)

assert torch.backends.mps.is_available()
mps_device = torch.device("mps")  # Device object representing GPU.

if __name__ == "__main__":
    B=1
    T=512
    C=1024
    H=C//HEAD_SIZE
    assert(C%HEAD_SIZE==0)

    r=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format)
    k=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format)
    v=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format)
    y=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format)

    w=torch.empty((C,1), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format)
    u=torch.empty((C,1), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format)

    wkv5_metal.forward(B,T,C,H,r,k,v,w,u,y)