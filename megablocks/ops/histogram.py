# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Any

# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

if not hasattr(torch, "npu"):
    # Wrap this in a try-block with better error message and
    # instructions for building the c++ operations.
    try:
        import megablocks_ops as ops  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("No module named 'megablocks_ops'.") from e


    # Autograd wrapper for histogram kernel.
    # NOTE: Does not support gradients.
    class HistogramOp(torch.autograd.Function):

        @staticmethod
        def forward(ctx: Any, x: torch.Tensor, max_val: float):
            return ops.histogram(x, max_val)
    histogram = HistogramOp.apply

else:
    def NpuHistogramOp(x: torch.Tensor, max_val: float):
        return torch.stack([torch.histc(y, max_val, 0, max_val - 1) for y in torch.split(x, 1)])

    histogram = NpuHistogramOp
