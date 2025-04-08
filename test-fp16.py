# do matmul in fp16, using pytorch and mps backend
# why instruments show high % of fp32???
'''
/Users/felixlin/workspace-mps/myenv-python312/bin/python3 test-fp16.py
'''
import torch
import time
import numpy as np
import random
import os
import gc
import sys
import math             

if not torch.backends.mps.is_available():
    print("MPS backend is not available on this system.")
    sys.exit(1)

# Set the device to MPS
device = torch.device("mps")

# Create two random matrices in fp16
matrix_a = torch.randn((1024, 1024), dtype=torch.float16, device=device)
matrix_b = torch.randn((1024, 1024), dtype=torch.float16, device=device)

# Measure execution time for matrix multiplication
start_time = time.time()
for _ in range(500):
    result = torch.matmul(matrix_a, matrix_b)
end_time = time.time()

# Print the result shape and execution time
print("Result shape:", result.shape, result.dtype)
print("Execution time:", end_time - start_time, "seconds")
