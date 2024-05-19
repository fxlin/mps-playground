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

B=8
T=512
C=768
H=C//HEAD_SIZE

# wraps native code as a py module(?
wkv5_metal = torch.utils.cpp_extension.load(
    name='wkv5',        # xzl: useful??
    sources=['wkv5_op.mm'],
    # verbose=True, 
    extra_cflags=['-std=c++17', f"-D_N_={HEAD_SIZE}"],
)

assert torch.backends.mps.is_available()
mps_device = torch.device("mps")  # Device object representing GPU.

def compute_rand():
    # r=torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16, 
    #     memory_format=torch.contiguous_format)
    r=torch.testing.make_tensor((B,T,C), device=mps_device, dtype=torch.bfloat16,
                              memory_format=torch.contiguous_format,low=0.0,high=1.0)
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

def load_compare():
    # load test file
    mydict = torch.load(f"/tmp/wkv-forwrad-{B}-{T}-{C}-20.pth",map_location='cpu')
    # breakpoint()
    r=mydict['r'].to(device=mps_device,memory_format=torch.contiguous_format)
    k=mydict['k'].to(device=mps_device,memory_format=torch.contiguous_format)
    v=mydict['v'].to(device=mps_device,memory_format=torch.contiguous_format)
    w=mydict['w'].to(device=mps_device,memory_format=torch.contiguous_format)
    u=mydict['u'].to(device=mps_device,memory_format=torch.contiguous_format)
    y=mydict['y'].to(device=mps_device,memory_format=torch.contiguous_format)

    ew = (-torch.exp(w.float())).contiguous()
    eew = (torch.exp(ew)).contiguous()
    # eew = eew.to(dtype=torch.bfloat16)

    # -- below ok -- # 
    ewc = (-torch.exp(w.cpu().float())).contiguous()
    eewc = (torch.exp(ewc)).contiguous()
    torch.testing.assert_close(eew.cpu(), eewc)

    yy = torch.empty((B,T,C), device=mps_device, dtype=torch.bfloat16, 
        memory_format=torch.contiguous_format)
    # breakpoint()
    wkv5_metal.forward(B,T,C,H,r,k,v,eew,u,yy)
    torch.testing.assert_close(y, yy)

if __name__ == "__main__":
    assert(C%HEAD_SIZE==0)
    load_compare()

