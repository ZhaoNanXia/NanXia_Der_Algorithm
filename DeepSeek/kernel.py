from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config


def act_quant_kernel(x_ptr, y_ptr, s_ptr, block_size):
    pid = tl.program_id(axis=0)


def act_quant(x, block_size):
    assert x.is_contiguous(), 'Input tensor must be contiguous'


def weight_dequant_kernel(x_ptr, y_ptr, s_ptr, m, n, block_size):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)


def weight_dequant(x, s, block_size):
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'


def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr, a_s_ptr, b_s_ptr, m, n, k, block_size_m, block_size_n, block_size_k):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)


def fp8_gemm(a, a_s, b, b_s):
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'