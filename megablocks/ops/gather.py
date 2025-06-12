# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
from stk.backend.autocast import custom_bwd, custom_fwd

from megablocks.backend import kernels


# Autograd wrapper for gather kernel.
class GatherOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ):
        ctx.save_for_backward(indices, bin_ids, bins)
        ctx.top_k = top_k
        if x.device.type == "npu":
            return gather_npu(x, indices, bin_ids, None, bins, top_k)
        else:
            return kernels.gather(x, indices, bin_ids, None, bins, top_k)

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()

        indices, bin_ids, bins = ctx.saved_tensors
        out = kernels.scatter(grad, indices, bin_ids, None, bins, ctx.top_k)
        return out, None, None, None, None, None


gather = GatherOp.apply

def assert_is_tensor(x, ndim):
    if x.ndim != ndim:
        raise ValueError(f'Expected {ndim}-tensor but got {x.ndim}-tensor')


def assert_is_matrix(x):
    assert_is_tensor(x, 2)


def assert_is_vector(x):
    if x.ndim != 1:
        raise ValueError(f'Expected 1-tensor but got {x.ndim}-tensor')


def assert_equal(a, b):
    if a != b:
        raise ValueError(f'Expected dimensions to be equal but got {a} and {b}.',)
    
def gather_npu(x, indices, bin_ids, weights, bins, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], x.shape[0] * top_k)
    assert_equal(bin_ids.shape[0], x.shape[0] * top_k)

    if weights is not None:
        assert_equal(weights.shape[0], x.shape[0] * top_k)

    # NOTE: There is no padding so the output rows equals the
    # input rows multiplied by top_k.
    output_rows = x.shape[0] * top_k
    out = torch.empty((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)
    padded_copy_pt(
        x,
        out,
        indices,
        bin_ids,
        weights,
        bins,
        bins,
        new_columns=x.shape[1],
        A_TO_B=True,
        TOP_K=top_k,
        SCALE=weights is not None,
    )
    return out


def padded_copy_pt(
    a: torch.Tensor,
    b: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    new_columns: int,  # 新增参数
    TOP_K: int,
    A_TO_B: bool = True,
    SCALE: bool = True,
):
    """
    实现与 Triton 内核相同的填充复制逻辑（含 new_columns 参数）
    
    参数:
        a (torch.Tensor): 输入张量 A，形状为 [num_rows_a, num_columns]
        b (torch.Tensor): 输出张量 B，形状为 [num_rows_b, new_columns]
        indices (torch.Tensor): 索引张量，长度为 n
        bin_ids (torch.Tensor): 分桶ID张量，长度为 n
        weights (torch.Tensor): 权重张量，长度为 max_index+1
        bins (torch.Tensor): 分桶累积行数，长度为 max_bin_id
        padded_bins (torch.Tensor): 分桶填充累积大小，长度与 bins 相同
        new_columns (int): 输出张量的列数（原 Triton 中的 NUM_COLUMNS）
        TOP_K (int): 复制因子
        A_TO_B (bool): 方向标志，True 表示 A->B，False 表示 B->A
        SCALE (bool): 是否缩放
    """
    n = indices.size(0)
    if n == 0:
        return
    
    device = indices.device
    
    # 计算前一个分桶的累积行数和填充大小
    prev_bin = torch.zeros(n, dtype=torch.long, device=device)
    padded_prev_bin = torch.zeros(n, dtype=torch.long, device=device)
    
    mask = bin_ids > 0
    if mask.any():
        bin_ids_minus_one = bin_ids[mask] - 1
        prev_bin[mask] = bins.index_select(0, bin_ids_minus_one)
        padded_prev_bin[mask] = padded_bins.index_select(0, bin_ids_minus_one)
    
    # 计算当前行在分桶内的偏移和输出索引
    i_tensor = torch.arange(n, device=device)
    offset_in_bin = i_tensor - prev_bin
    index_b_computed = offset_in_bin + padded_prev_bin

    # 列数处理（新增逻辑）
    num_columns = min(a.size(1), new_columns) if A_TO_B else min(b.size(1), new_columns)
    col_indices = torch.arange(num_columns, device=device)

    if A_TO_B:
        # 从 A 复制到 B
        input_rows = indices // TOP_K
        
        if SCALE:
            w = weights.index_select(0, indices).float().view(-1, 1)
            source_data = a.index_select(0, input_rows)[:, col_indices].float()
            scaled = (source_data * w).to(b.dtype)
        else:
            scaled = a.index_select(0, input_rows)[:, col_indices]
        
        # 确保目标索引不越界
        valid_mask = index_b_computed < b.size(0)
        if valid_mask.any():
            b.index_copy_(0, index_b_computed[valid_mask], scaled[valid_mask])
    else:
        # 从 B 复制到 A
        if SCALE:
            w = weights.index_select(0, indices).float().view(-1, 1)
            source_data = b.index_select(0, index_b_computed)[:, col_indices].float()
            scaled = (source_data * w).to(a.dtype)
        else:
            scaled = b.index_select(0, index_b_computed)[:, col_indices]
        
        # 直接写入目标位置
        a.index_copy_(0, indices, scaled)