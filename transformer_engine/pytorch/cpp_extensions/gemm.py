# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for GEMM extensions"""

from typing import Iterable, Optional, Tuple, Union, List
import os
import torch
import transformer_engine_torch as tex
from ..constants import TE_DType
from ..utils import get_sm_count, _empty_tensor

from ..quantized_tensor import Quantizer
from ..tensor.storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from ..tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from ..tensor.utils import is_custom
from ..custom_recipes.gemm import custom_gemm
from ...debug.pytorch.debug_quantization import DebugQuantizer

__all__ = [
    "general_gemm",
    "general_grouped_gemm",
]


def validate_gemm_scale(scale: Optional[float], required: bool) -> float:
    """Validate whether a GEMM scaling factor is consistent with its usage"""
    if required:
        return scale if scale is not None else 1.0
    if scale not in (0.0, None):
        raise ValueError("scale must be zero")
    return 0.0


def general_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    workspace: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
    alpha: float = 1.0,
    beta: Optional[float] = None,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_split_accumulator: bool = False,
    grad: bool = False,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_type: tex.CommOverlapType = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
) -> Iterable[Optional[torch.Tensor]]:
    """GEMM supporting fp8 inputs."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    # assert quantization_params is None, "FP8 output not supported yet"

    alpha = validate_gemm_scale(alpha, True)
    beta = validate_gemm_scale(beta, accumulate)

    if ub_type is not None:
        assert ub is not None, (
            f"{'AG+GEMM' if ub_type == tex.CommOverlapType.AG else 'GEMM+RS'} overlap requires"
            + "a valid `ub` communicator object."
        )

    if ub is not None:
        assert ub_type is not None, "Comm+GEMM overlap requires a valid `comm_type` argument."
        if ub_type == tex.CommOverlapType.RS:
            if not (bulk_overlap and not ub.is_fp8_ubuf()):
                assert extra_output is not None, "GEMM+RS overlap requires extra output tensor."

    if out is not None:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    # If A or B are custom tensors -> dispatch to quantizers's qgemm implementation
    if is_custom(A) or is_custom(B):
        return custom_gemm(
            A,
            B,
            workspace,
            out_dtype,
            quantization_params,
            gelu,
            gelu_in,
            accumulate,
            layout,
            out,
            bias,
            use_split_accumulator,
            grad,
        )

    debug_quantizer = None
    if isinstance(quantization_params, DebugQuantizer):
        debug_quantizer = quantization_params
        quantization_params = quantization_params.parent_quantizer
        A = A.get_tensor(not transa)
        B = B.get_tensor(transb)

    # Use bfloat16 as default bias_dtype
    bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]

    if isinstance(A, Float8BlockwiseQTensorStorage) or isinstance(B, Float8BlockwiseQTensorStorage):
        # There is not use_split_accumulator == False
        # implementation for Float8BlockwiseQTensorStorage GEMM
        use_split_accumulator = True

        # Check that data format is supported
        if (
            A._data_format != tex.Float8BlockScaleTensorFormat.GEMM_READY
            or B._data_format != tex.Float8BlockScaleTensorFormat.GEMM_READY
        ):
            raise RuntimeError("GEMM with Float8BlockwiseQTensor requires GEMM_READY format")

    args = (
        A,
        transa,  # transa
        B,
        transb,  # transb
        out,
        quantization_params,
        TE_DType[out_dtype] if out_dtype is not None else None,
        bias,
        bias_dtype,
        gelu,
        gelu_in,
        grad,  # grad
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )
    kwargs = {
        "comm_overlap": ub,
        "comm_type": ub_type,
        "extra_output": extra_output,
        "bulk_overlap": bulk_overlap,
        "alpha": alpha,
        "beta": beta,
    }

    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)

    if debug_quantizer is not None:
        out = debug_quantizer.process_gemm_output(out)

    return out, bias_grad, gelu_input, extra_output


def general_grouped_gemm(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out: List[torch.Tensor],
    out_dtype: torch.dtype,
    workspaces: List[torch.Tensor],
    layout: str = "TN",
    m_splits: Optional[torch.Tensor] = None,
    m_splits_on_devie: bool = False,
    gelu: bool = False,
    grad=False,
    wgrad=False,
    accumulate: bool = False,
    accumulate_mask: Optional[torch.Tensor] = None,
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
    single_output=False,
) -> Tuple[List[torch.Tensor], ...]:
    """
    TN layout Grouped GEMM with fp8 inputs.
    """
    # print("===========general_grouped_gemm===========")
    # print("accumulate:", accumulate)
    # print(f"layout: {layout}")
    if isinstance(m_splits, list):
        m_splits = torch.tensor(m_splits)
    num_gemms = m_splits.size(0)
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    empty_tensor = _empty_tensor()
    empty_tensors = [empty_tensor] * num_gemms

    # Use bfloat16 as default bias_dtype
    gelu_input = empty_tensors
    out_dtype = TE_DType[out[0].dtype] if D_dtype is None else D_dtype

    sm_count = get_sm_count()
    if grad and use_bias:
        grad_bias = [
            torch.empty(B[i].shape[1], dtype=out[0].dtype, device="cuda") for i in range(num_gemms)
        ]
    else:
        grad_bias = empty_tensors
    bias = bias if use_bias else empty_tensors
    if use_bias:
        bias_dtype = TE_DType[grad_bias[0].dtype] if grad else TE_DType[bias[0].dtype]
    else:
        bias_dtype = TE_DType[torch.bfloat16]

    if gelu:
        gelu_input = [
            torch.empty_like(o, dtype=bias_dtype, memory_format=torch.contiguous_format)
            for o in out
        ]  # this should differ with respect to single output

    # print("===========call tex.te_general_grouped_gemm===========")
    # print("A[0]:",  A[0].get_metadata_debug())
    # print("B[0]:",  B[0].get_metadata_debug())
    if not m_splits_on_devie:
        bias = tex.te_general_grouped_gemm(
            A,
            transa,
            B,
            transb,
            out,
            out_dtype,
            m_splits,
            grad_bias if grad else bias,
            bias_dtype,
            single_output,
            gelu_input,  # this is pre_gelu_out
            grad,  # grad
            workspaces,
            workspaces[0].shape[0],
            accumulate,
            use_split_accumulator,
            sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count))),
        )
    else:
        assert isinstance(A[0], MXFP8TensorStorage) and isinstance(B[0], MXFP8TensorStorage), "Only MXFP8 A and B are supported when m_splits is on device"
        assert out[0].dtype == torch.bfloat16 or out[0].dtype == torch.float16 or (wgrad and out[0].dtype == torch.float32), "Only BF16, FP16 or FP32(only for wgrad accumulation) output is supported when m_splits is on device"
        assert not use_bias, "Bias is not supported when m_splits is on device"
        assert not gelu, "GELU is not supported when m_splits is on device"
        assert TE_DType[out[0].dtype] == out_dtype, "Output dtype mismatch: out[0].dtype=" + str(out[0].dtype) + ", out_dtype=" + str(out_dtype)
        bias = tex.te_general_device_initiated_grouped_gemm(
            A,
            transa,
            B,
            transb,
            out,
            out_dtype,
            m_splits,
            grad_bias if grad else bias,
            bias_dtype,
            single_output,
            gelu_input,  # this is pre_gelu_out
            grad,  # grad
            wgrad, # wgrad
            workspaces,
            workspaces[0].shape[0],
            accumulate,
            accumulate_mask,
            use_split_accumulator,
            sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count))),
        )

    return out, bias, gelu_input
