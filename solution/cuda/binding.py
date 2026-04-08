"""
Unused under binding = "torch".

The CUDA-track solution is built as a torch C++/CUDA extension; the framework
JIT-compiles solution/cuda/kernel.cu via torch.utils.cpp_extension.load and
calls the exported `kernel` symbol directly. This file is kept only because
pack_solution scans the cuda source directory for any auxiliary files; it has
no runtime role.
"""
