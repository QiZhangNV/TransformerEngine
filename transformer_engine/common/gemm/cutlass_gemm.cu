/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <float.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>
#include <mutex>
#include <vector>

#include "../common.h"
#include "../util/handle_manager.h"
#include "../util/logging.h"
#include "common/util/cuda_runtime.h"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/device_memory.h"

using namespace cute;

/*****
 * Fprop and Dgrad
 ******/

template <typename Sm1xxBlkScaledConfig, typename UnderlyingProblemShape, typename ElementA,
          typename ElementD, typename ElementSF, typename StrideA, typename StrideB,
          typename StrideD, typename LayoutSFA, typename LayoutSFB, bool transB>
__global__ void setGroupedGemmArguments(int num_experts, const int64_t *gemm_m_per_expert,
                                        int gemm_n, int gemm_k, ElementA *ptr_A, ElementSF *ptr_SFA,
                                        ElementD *ptr_D, UnderlyingProblemShape *problem_sizes,
                                        ElementA **ptr_A_list, ElementSF **ptr_SFA_list,
                                        StrideA *stride_A_list, LayoutSFA *layout_SFA_list,
                                        StrideB *stride_B_list, LayoutSFB *layout_SFB_list,
                                        ElementD **ptr_D_list, StrideD *stride_D_list) {
  int m_offset = 0;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int expert_id = 0; expert_id < num_experts; expert_id++) {
      int gemm_m = int(gemm_m_per_expert[expert_id]);
      problem_sizes[expert_id] = cute::make_shape(gemm_m, gemm_n, gemm_k);
      // printf("problem_sizes: %d, %d, %d\n", gemm_m, gemm_n, gemm_k);

      ptr_A_list[expert_id] = ptr_A + m_offset * gemm_k;
      ptr_SFA_list[expert_id] = ptr_SFA + m_offset * ((gemm_k + 127) / 128 * 4);
      stride_A_list[expert_id] = cute::make_stride(int64_t(gemm_k), _1{}, _0{});
      layout_SFA_list[expert_id] =
          Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(gemm_m, gemm_n, gemm_k, 1));

      if constexpr (transB) {
        stride_B_list[expert_id] = cute::make_stride(int64_t(gemm_k), _1{}, _0{});
      } else {
        stride_B_list[expert_id] = cute::make_stride(_1{}, int64_t(gemm_n), _0{});
      }
      layout_SFB_list[expert_id] =
          Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(gemm_m, gemm_n, gemm_k, 1));

      ptr_D_list[expert_id] = ptr_D + m_offset * gemm_n;
      stride_D_list[expert_id] = cute::make_stride(int64_t(gemm_n), _1{}, _0{});

      m_offset += gemm_m;
    }
  }
}

template <typename T, typename TSF, typename WeightType, typename WeightTypeSF, typename OutputType,
          bool TransB>
void generic_moe_gemm_kernelLauncher(T *A, TSF *SFA, WeightType **B_list, WeightTypeSF **SFB_list,
                                     OutputType *D, const int64_t *gemm_m_per_expert, int gemm_n,
                                     int gemm_k, int num_experts, size_t workspaceSize,
                                     void *workspace, cudaStream_t stream,
                                     int *kernel_occupancy = nullptr) {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group
  using ElementInput = cutlass::float_e4m3_t;  // Element type for Input matrix operands
  using ElementSF = cutlass::float_ue8m0_t;    // Element type for SF matrix operands
  using ElementC = cutlass::bfloat16_t;

  using ElementA = cutlass::mx_float8_t<ElementInput>;  // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;            // Layout type for A matrix operand
  constexpr int AlignmentA = 32;  // Alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = cutlass::mx_float8_t<ElementInput>;  // Element type for B matrix operand
  using LayoutB =
      cute::conditional_t<TransB, cutlass::layout::ColumnMajor,
                          cutlass::layout::RowMajor>;  // Layout type for B matrix operand
  constexpr int AlignmentB = 32;  // Alignment of A matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using ElementD = ElementC;                  // Element type for D matrix operands
  using LayoutC = cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
  constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<
                ElementC>::value;  // Alignment of C matrix in units of elements (up to 16 bytes)
  constexpr int AlignmentD =
      128 / cutlass::sizeof_bits<
                ElementD>::value;    // Alignment of D matrix in units of elements (up to 16 bytes)
  using ElementAccumulator = float;  // Element type for internal accumulation

  // Core kernel configurations
  using ArchTag =
      cutlass::arch::Sm100;  // Tag indicating the minimum SM that supports the intended feature
  using EpilogueOperatorClass = cutlass::arch::OpClassTensorOp;  // Epilogue Operator class tag
  using MainloopOperatorClass =
      cutlass::arch::OpClassBlockScaledTensorOp;  // Mainloop Operator class tag
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto;  // Stage count maximized based on the tile size

  // Runtime Cluster Shape
  using ClusterShape = Shape<int32_t, int32_t, _1>;

  struct MMA2SMConfig {
    using MmaTileShape = Shape<_256, _256, _128>;
    using KernelSchedule =
        cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;  // Kernel to launch
    using EpilogueSchedule =
        cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;  // Epilogue to launch
  };

  using CollectiveEpilogue2SM = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, EpilogueOperatorClass, typename MMA2SMConfig::MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator, void,
      LayoutC *, AlignmentC, ElementD, LayoutC *, AlignmentD,
      typename MMA2SMConfig::EpilogueSchedule
      // , FusionOperation  // Enable for SF Output
      >::CollectiveOp;
  using CollectiveMainloop2SM = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, MainloopOperatorClass, ElementA, LayoutA *, AlignmentA, ElementB, LayoutB *,
      AlignmentB, ElementAccumulator, typename MMA2SMConfig::MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue2SM::SharedStorage))>,
      typename MMA2SMConfig::KernelSchedule>::CollectiveOp;
  using GemmKernel2SM = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop2SM,
                                                             CollectiveEpilogue2SM>;
  using GemmGrouped = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2SM>;

  using StrideA = typename GemmGrouped::GemmKernel::InternalStrideA;
  using StrideB = typename GemmGrouped::GemmKernel::InternalStrideB;
  using StrideC = typename GemmGrouped::GemmKernel::InternalStrideC;
  using StrideD = typename GemmGrouped::GemmKernel::InternalStrideD;

  using LayoutSFA = typename GemmGrouped::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename GemmGrouped::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using Sm1xxBlkScaledConfig =
      typename GemmGrouped::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

  auto get_aligned_offset = [](size_t current_offset, size_t alignment) -> size_t {
    return (current_offset + alignment - 1) & ~(alignment - 1);
  };

  if (workspace == nullptr) {
    throw std::runtime_error("[FT Error][MoE Runner] workspace is null");
  }

  size_t offset = 0;
  typename GemmGrouped::ElementA *ptr_A = reinterpret_cast<typename GemmGrouped::ElementA *>(A);
  typename GemmGrouped::ElementA **ptr_A_list = reinterpret_cast<typename GemmGrouped::ElementA **>(
      reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(typename GemmGrouped::ElementA *), 128);

  typename GemmGrouped::ElementB **ptr_B_list = reinterpret_cast<typename GemmGrouped::ElementB **>(
      reinterpret_cast<char *>(workspace) + offset);
  cudaMemcpyAsync(ptr_B_list, B_list, num_experts * sizeof(typename GemmGrouped::ElementB *),
                  cudaMemcpyHostToDevice, stream);
  offset = get_aligned_offset(offset + num_experts * sizeof(typename GemmGrouped::ElementB *), 128);

  typename GemmGrouped::ElementD *ptr_D = reinterpret_cast<typename GemmGrouped::ElementD *>(D);
  typename GemmGrouped::ElementD **ptr_D_list = reinterpret_cast<typename GemmGrouped::ElementD **>(
      reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(typename GemmGrouped::ElementD *), 128);

  typename GemmGrouped::GemmKernel::ElementSF *ptr_SFA =
      reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF *>(SFA);
  typename GemmGrouped::GemmKernel::ElementSF **ptr_SFA_list =
      reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF **>(
          reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(
      offset + num_experts * sizeof(typename GemmGrouped::GemmKernel::ElementSF *), 128);

  typename GemmGrouped::GemmKernel::ElementSF **ptr_SFB_list =
      reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF **>(
          reinterpret_cast<char *>(workspace) + offset);
  cudaMemcpyAsync(ptr_SFB_list, SFB_list,
                  num_experts * sizeof(typename GemmGrouped::GemmKernel::ElementSF *),
                  cudaMemcpyHostToDevice, stream);
  offset = get_aligned_offset(
      offset + num_experts * sizeof(typename GemmGrouped::GemmKernel::ElementSF *), 128);

  StrideA *stride_A_list =
      reinterpret_cast<StrideA *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(StrideA), 128);

  StrideB *stride_B_list =
      reinterpret_cast<StrideB *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(StrideB), 128);

  StrideD *stride_D_list =
      reinterpret_cast<StrideD *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(StrideD), 128);

  LayoutSFA *layout_SFA_list =
      reinterpret_cast<LayoutSFA *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(LayoutSFA), 128);

  LayoutSFB *layout_SFB_list =
      reinterpret_cast<LayoutSFB *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(LayoutSFB), 128);

  ProblemShape::UnderlyingProblemShape *problem_sizes =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape *>(reinterpret_cast<char *>(workspace) +
                                                               offset);
  offset =
      get_aligned_offset(offset + num_experts * sizeof(ProblemShape::UnderlyingProblemShape), 128);

  setGroupedGemmArguments<Sm1xxBlkScaledConfig, ProblemShape::UnderlyingProblemShape,
                          typename GemmGrouped::ElementA, typename GemmGrouped::ElementD,
                          typename GemmGrouped::GemmKernel::ElementSF, StrideA, StrideB, StrideD,
                          LayoutSFA, LayoutSFB, TransB><<<1, 32, 0, stream>>>(
      num_experts, gemm_m_per_expert, gemm_n, gemm_k, ptr_A, ptr_SFA, ptr_D, problem_sizes,
      ptr_A_list, ptr_SFA_list, stride_A_list, layout_SFA_list, stride_B_list, layout_SFB_list,
      ptr_D_list, stride_D_list);

  typename GemmGrouped::Arguments args;
  decltype(args.epilogue.thread) fusion_args;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  // Set alpha and beta to 1 and 0 for the fusion operation
  fusion_args.alpha = 1;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.dAlpha = {_0{}, _0{}, 0};
  fusion_args.beta = 0;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dBeta = {_0{}, _0{}, 0};

  cutlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  if (!is_static_v<ClusterShape>) {
    hw_info.cluster_shape = dim3(4, 4, 1);
    hw_info.cluster_shape_fallback = dim3(2, 1, 1);
  }

  typename GemmGrouped::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = RasterOrderOptions::AlongN;

  args = typename GemmGrouped::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes, nullptr},
      {const_cast<const typename GemmGrouped::ElementA **>(ptr_A_list), stride_A_list,
       const_cast<const typename GemmGrouped::ElementB **>(ptr_B_list), stride_B_list,
       const_cast<const typename GemmGrouped::GemmKernel::ElementSF **>(ptr_SFA_list),
       layout_SFA_list,
       const_cast<const typename GemmGrouped::GemmKernel::ElementSF **>(ptr_SFB_list),
       layout_SFB_list},
      {fusion_args, nullptr, stride_D_list, ptr_D_list, stride_D_list},
      hw_info,
      scheduler};

  GemmGrouped gemm;

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = GemmGrouped::get_workspace_size(args);
  if (workspaceSize < offset + workspace_size) {  // 16MB limit
    throw std::runtime_error("Calculated workspace size (" +
                             std::to_string(offset + workspace_size) + ") exceeds buffer size (" +
                             std::to_string(workspaceSize) + ")\n");
  }

  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg = "MoE kernel will fail for params. Error: " +
                          std::string(cutlassGetStatusString(can_implement));
    throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
  }

  auto init_status = gemm.initialize(args, reinterpret_cast<char *>(workspace) + offset);
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to initialize cutlass grouped gemm. Error: " +
                          std::string(cutlassGetStatusString(init_status));
    throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
  }

  auto run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to run cutlass grouped gemm. Error: " +
                          std::string(cutlassGetStatusString(run_status));
    throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
  }
}

// Only mxfp8 is supported for now
// A is single Tensor, B is splited tensor list, D is single Tensor
void nvte_cutlass_grouped_gemm(const NVTETensor *A, const NVTETensor *B, NVTETensor *D,
                               const int64_t *m_splits, const NVTETensor *bias,
                               NVTETensor *pre_gelu_out, const int num_gemms, bool transa,
                               bool transb, bool grad, NVTETensor *workspace, size_t workspaceSize,
                               bool accumulate, bool use_split_accumulator, int math_sm_count,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_cutlass_grouped_gemm);
  using namespace transformer_engine;
  // printf("===========nvte_cutlass_grouped_gemm===========\n");
  // printf("transa: %d, transb: %d\n", transa, transb);
  // printf("grad: %d\n", grad);

  // Process A
  const transformer_engine::Tensor *inputA = convertNVTETensor(A[0]);
  if (transa) {
    NVTE_CHECK(inputA->has_columnwise_data(), "Input A is missing column-wise usage");
  } else {
    NVTE_CHECK(inputA->has_data(), "Input A is missing row-wise usage");
  }
  __nv_fp8_e4m3 *inputA_ptr = transa
                                  ? reinterpret_cast<__nv_fp8_e4m3 *>(inputA->columnwise_data.dptr)
                                  : reinterpret_cast<__nv_fp8_e4m3 *>(inputA->data.dptr);
  __nv_fp8_e8m0 *inputA_SF_ptr =
      transa ? reinterpret_cast<__nv_fp8_e8m0 *>(inputA->columnwise_scale_inv.dptr)
             : reinterpret_cast<__nv_fp8_e8m0 *>(inputA->scale_inv.dptr);

  // Process B
  __nv_fp8_e4m3 *inputB_ptr_list[num_gemms];
  __nv_fp8_e8m0 *inputB_SF_ptr_list[num_gemms];
  for (size_t i = 0; i < num_gemms; i++) {
    const transformer_engine::Tensor *inputB = convertNVTETensor(B[i]);
    if (transb) {
      NVTE_CHECK(inputB->has_data(), "Input B is missing row-wise usage");
    } else {
      NVTE_CHECK(inputB->has_columnwise_data(), "Input B is missing column-wise usage");
    }
    inputB_ptr_list[i] = transb ? reinterpret_cast<__nv_fp8_e4m3 *>(inputB->data.dptr)
                                : reinterpret_cast<__nv_fp8_e4m3 *>(inputB->columnwise_data.dptr);
    inputB_SF_ptr_list[i] =
        transb ? reinterpret_cast<__nv_fp8_e8m0 *>(inputB->scale_inv.dptr)
               : reinterpret_cast<__nv_fp8_e8m0 *>(inputB->columnwise_scale_inv.dptr);
  }

  // Process D
  const transformer_engine::Tensor *outputD = convertNVTETensor(D[0]);
  NVTE_CHECK(outputD->has_data(), "Input D is missing row-wise usage");
  __nv_bfloat16 *outputD_ptr = reinterpret_cast<__nv_bfloat16 *>(outputD->data.dptr);

  // Get GEMM shape
  const int gemm_k = transa ? inputA->flat_first_dim() : inputA->flat_last_dim();
  const int gemm_n =
      transb ? convertNVTETensor(B[0])->flat_first_dim() : convertNVTETensor(B[0])->flat_last_dim();
  //   printf("num_gemms: %d\n", num_gemms);
  //   printf("gemm_n: %d, gemm_k: %d\n", gemm_n, gemm_k);
  if ((gemm_k & 0x1F) != 0) {
    throw std::runtime_error("gemm_k of grouped gemm with variable M must be a multiple of 32.");
  }

  if (transb) {
    generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e8m0, __nv_fp8_e4m3, __nv_fp8_e8m0,
                                    __nv_bfloat16, true>(
        inputA_ptr, inputA_SF_ptr, inputB_ptr_list, inputB_SF_ptr_list, outputD_ptr,
        m_splits,  // gemm_m splits
        gemm_n,    // gemm_n
        gemm_k,    // gemm_k
        num_gemms, workspaceSize, convertNVTETensor(workspace[0])->data.dptr, stream);
  } else {
    generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e8m0, __nv_fp8_e4m3, __nv_fp8_e8m0,
                                    __nv_bfloat16, false>(
        inputA_ptr, inputA_SF_ptr, inputB_ptr_list, inputB_SF_ptr_list, outputD_ptr,
        m_splits,  // gemm_m splits
        gemm_n,    // gemm_n
        gemm_k,    // gemm_k
        num_gemms, workspaceSize, convertNVTETensor(workspace[0])->data.dptr, stream);
  }
}

/*****
 * Wgrad
 ******/

template <typename Sm1xxBlkScaledConfig, typename UnderlyingProblemShape, typename ElementA,
          typename ElementB, typename ElementD, typename ElementSF, typename StrideA,
          typename StrideB, typename StrideD, typename LayoutSFA, typename LayoutSFB, bool transD>
__global__ void setGroupedGemmWgradArguments(
    int num_experts, int gemm_m, int gemm_n, const int64_t *gemm_k_per_expert, int total_gemm_k,
    ElementA *ptr_A, ElementSF *ptr_SFA, ElementB *ptr_B, ElementSF *ptr_SFB,
    UnderlyingProblemShape *problem_sizes, ElementA **ptr_A_list, ElementSF **ptr_SFA_list,
    StrideA *stride_A_list, LayoutSFA *layout_SFA_list, ElementB **ptr_B_list,
    ElementSF **ptr_SFB_list, StrideB *stride_B_list, LayoutSFB *layout_SFB_list,
    ElementD **ptr_D_list, StrideD *stride_D_list, bool accumulate_D) {
  // printf("===========wgrad setGroupedGemmWgradArguments===========\n");
  // printf("transD: %d\n", transD);
  int k_offset = 0;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int expert_id = 0; expert_id < num_experts; expert_id++) {
      int gemm_k = int(gemm_k_per_expert[expert_id]);
      if (gemm_k == 0) {
        // If gemm_k is 0, we need to set the problem_sizes to 0, 0, 0 to skip the gemm
        problem_sizes[expert_id] = cute::make_shape(0, 0, 0);
        if (!accumulate_D) {
          for (int i = 0; i < gemm_m * gemm_n; i++) {
            ptr_D_list[expert_id][i] = ElementD(0);
          }
        }
        continue;
      }
      problem_sizes[expert_id] = cute::make_shape(gemm_m, gemm_n, gemm_k);
      // printf("wgrad problem_sizes: %d, %d, %d\n", gemm_m, gemm_n, gemm_k);

      ptr_A_list[expert_id] = ptr_A + gemm_m * k_offset;
      ptr_SFA_list[expert_id] = ptr_SFA + 128 * ((k_offset + 127) / 128 * 4);
      stride_A_list[expert_id] = cute::make_stride(_1{}, int64_t(gemm_m), _0{});
      auto temp_sfa_layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
          cute::make_shape(gemm_m, gemm_n, total_gemm_k, 1));
      layout_SFA_list[expert_id] = cute::make_layout(
          get<0>(temp_sfa_layout),
          make_layout(get<0>(get<1>(temp_sfa_layout)),
                      make_layout(gemm_k / 128, get<1>(get<1>(temp_sfa_layout.stride())))),

          get<2>(temp_sfa_layout));

      ptr_B_list[expert_id] = ptr_B + gemm_n * k_offset;
      ptr_SFB_list[expert_id] = ptr_SFB + 128 * ((k_offset + 127) / 128 * 4);
      stride_B_list[expert_id] = cute::make_stride(_1{}, int64_t(gemm_n), _0{});
      auto temp_sfb_layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
          cute::make_shape(gemm_m, gemm_n, total_gemm_k, 1));
      layout_SFB_list[expert_id] = cute::make_layout(
          get<0>(temp_sfb_layout),
          make_layout(get<0>(get<1>(temp_sfb_layout)),
                      make_layout(gemm_k / 128, get<1>(get<1>(temp_sfb_layout.stride())))),
          get<2>(temp_sfb_layout));

      if constexpr (transD) {
        stride_D_list[expert_id] = cute::make_stride(_1{}, int64_t(gemm_m), _0{});
      } else {
        stride_D_list[expert_id] = cute::make_stride(int64_t(gemm_n), _1{}, _0{});
      }

      k_offset += gemm_k;
    }
  }
}

template <typename T, typename TSF, typename WeightType, typename WeightTypeSF, typename OutputType,
          bool TransD>
void generic_moe_gemm_wgrad_kernelLauncher(T *A, TSF *SFA, WeightType *B, WeightTypeSF *SFB,
                                           void **D_list, int gemm_m, int gemm_n,
                                           const int64_t *gemm_k_per_expert, int total_gemm_k,
                                           int num_experts, bool accumulate_D, size_t workspaceSize,
                                           void *workspace, cudaStream_t stream,
                                           int *kernel_occupancy = nullptr) {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group
  using ElementInput = cutlass::float_e4m3_t;  // Element type for Input matrix operands
  using ElementSF = cutlass::float_ue8m0_t;    // Element type for SF matrix operands
  using ElementC =
      cute::conditional_t<cute::is_same_v<OutputType, __nv_bfloat16>, cutlass::bfloat16_t, float>;

  using ElementA = cutlass::mx_float8_t<ElementInput>;  // Element type for A matrix operand
  using LayoutA = cutlass::layout::ColumnMajor;         // Layout type for A matrix operand
  constexpr int AlignmentA = 32;  // Alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = cutlass::mx_float8_t<ElementInput>;  // Element type for B matrix operand
  using LayoutB = cutlass::layout::RowMajor;            // Layout type for B matrix operand
  constexpr int AlignmentB = 32;  // Alignment of A matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using ElementD = ElementC;  // Element type for D matrix operands
  using LayoutC = typename cutlass::platform::conditional<
      TransD, cutlass::layout::ColumnMajor,
      cutlass::layout::RowMajor>::type;  // Layout type for C and D matrix operands
  constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<
                ElementC>::value;  // Alignment of C matrix in units of elements (up to 16 bytes)
  constexpr int AlignmentD =
      128 / cutlass::sizeof_bits<
                ElementD>::value;    // Alignment of D matrix in units of elements (up to 16 bytes)
  using ElementAccumulator = float;  // Element type for internal accumulation

  // Core kernel configurations
  using ArchTag =
      cutlass::arch::Sm100;  // Tag indicating the minimum SM that supports the intended feature
  using EpilogueOperatorClass = cutlass::arch::OpClassTensorOp;  // Epilogue Operator class tag
  using MainloopOperatorClass =
      cutlass::arch::OpClassBlockScaledTensorOp;  // Mainloop Operator class tag
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto;  // Stage count maximized based on the tile size

  // Runtime Cluster Shape
  using ClusterShape = Shape<int32_t, int32_t, _1>;

  struct MMA2SMConfig {
    using MmaTileShape = Shape<_256, _256, _128>;
    using KernelSchedule =
        cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;  // Kernel to launch
    using EpilogueSchedule =
        cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;  // Epilogue to launch
  };

  using CollectiveEpilogue2SM = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, EpilogueOperatorClass, typename MMA2SMConfig::MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC *, AlignmentC, ElementD, LayoutC *, AlignmentD,
      typename MMA2SMConfig::EpilogueSchedule
      // , FusionOperation  // Enable for SF Output
      >::CollectiveOp;
  using CollectiveMainloop2SM = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, MainloopOperatorClass, ElementA, LayoutA *, AlignmentA, ElementB, LayoutB *,
      AlignmentB, ElementAccumulator, typename MMA2SMConfig::MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue2SM::SharedStorage))>,
      typename MMA2SMConfig::KernelSchedule>::CollectiveOp;
  using GemmKernel2SM = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop2SM,
                                                             CollectiveEpilogue2SM>;
  using GemmGrouped = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2SM>;

  using StrideA = typename GemmGrouped::GemmKernel::InternalStrideA;
  using StrideB = typename GemmGrouped::GemmKernel::InternalStrideB;
  using StrideC = typename GemmGrouped::GemmKernel::InternalStrideC;
  using StrideD = typename GemmGrouped::GemmKernel::InternalStrideD;

  using LayoutSFA = typename GemmGrouped::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename GemmGrouped::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using Sm1xxBlkScaledConfig =
      typename GemmGrouped::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

  // Helper function to calculate aligned offset
  auto get_aligned_offset = [](int current_offset, int alignment) -> int {
    return (current_offset + alignment - 1) & ~(alignment - 1);
  };

  if (workspace == nullptr) {
    throw std::runtime_error("[FT Error][MoE Runner] workspace is null");
  }

  int offset = 0;
  typename GemmGrouped::ElementA *ptr_A = reinterpret_cast<typename GemmGrouped::ElementA *>(A);
  typename GemmGrouped::ElementA **ptr_A_list = reinterpret_cast<typename GemmGrouped::ElementA **>(
      reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(typename GemmGrouped::ElementA *), 128);

  typename GemmGrouped::ElementB *ptr_B = reinterpret_cast<typename GemmGrouped::ElementB *>(B);
  typename GemmGrouped::ElementB **ptr_B_list = reinterpret_cast<typename GemmGrouped::ElementB **>(
      reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(typename GemmGrouped::ElementB *), 128);

  typename GemmGrouped::ElementD **ptr_D_list = reinterpret_cast<typename GemmGrouped::ElementD **>(
      reinterpret_cast<char *>(workspace) + offset);
  cudaMemcpyAsync(ptr_D_list, D_list, num_experts * sizeof(typename GemmGrouped::ElementD *),
                  cudaMemcpyHostToDevice, stream);
  offset = get_aligned_offset(offset + num_experts * sizeof(typename GemmGrouped::ElementD *), 128);

  typename GemmGrouped::GemmKernel::ElementSF *ptr_SFA =
      reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF *>(SFA);
  typename GemmGrouped::GemmKernel::ElementSF **ptr_SFA_list =
      reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF **>(
          reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(
      offset + num_experts * sizeof(typename GemmGrouped::GemmKernel::ElementSF *), 128);

  typename GemmGrouped::GemmKernel::ElementSF *ptr_SFB =
      reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF *>(SFB);
  typename GemmGrouped::GemmKernel::ElementSF **ptr_SFB_list =
      reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF **>(
          reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(
      offset + num_experts * sizeof(typename GemmGrouped::GemmKernel::ElementSF *), 128);

  StrideA *stride_A_list =
      reinterpret_cast<StrideA *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(StrideA), 128);
  StrideB *stride_B_list =
      reinterpret_cast<StrideB *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(StrideB), 128);
  StrideD *stride_D_list =
      reinterpret_cast<StrideD *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(StrideD), 128);

  LayoutSFA *layout_SFA_list =
      reinterpret_cast<LayoutSFA *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(LayoutSFA), 128);
  LayoutSFB *layout_SFB_list =
      reinterpret_cast<LayoutSFB *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(LayoutSFB), 128);

  ProblemShape::UnderlyingProblemShape *problem_sizes =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape *>(reinterpret_cast<char *>(workspace) +
                                                               offset);
  offset =
      get_aligned_offset(offset + num_experts * sizeof(ProblemShape::UnderlyingProblemShape), 128);

  setGroupedGemmWgradArguments<Sm1xxBlkScaledConfig, ProblemShape::UnderlyingProblemShape,
                               typename GemmGrouped::ElementA, typename GemmGrouped::ElementB,
                               typename GemmGrouped::ElementD,
                               typename GemmGrouped::GemmKernel::ElementSF, StrideA, StrideB,
                               StrideD, LayoutSFA, LayoutSFB, TransD><<<1, 32, 0, stream>>>(
      num_experts, gemm_m, gemm_n, gemm_k_per_expert, total_gemm_k, ptr_A, ptr_SFA, ptr_B, ptr_SFB,
      problem_sizes, ptr_A_list, ptr_SFA_list, stride_A_list, layout_SFA_list, ptr_B_list,
      ptr_SFB_list, stride_B_list, layout_SFB_list, ptr_D_list, stride_D_list, accumulate_D);

  // Check for CUDA errors after kernel launch
  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess) {
    std::string err_msg = "Failed to run setGroupedGemmWgradArguments. CUDA Error: " +
                          std::string(cudaGetErrorString(cuda_error));
    throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
  }

  typename GemmGrouped::Arguments args;
  decltype(args.epilogue.thread) fusion_args;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  // Set alpha and beta
  fusion_args.alpha = 1;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.dAlpha = {_0{}, _0{}, 0};
  fusion_args.beta = accumulate_D ? 1 : 0;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dBeta = {_0{}, _0{}, 0};

  cutlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  if (!is_static_v<ClusterShape>) {
    hw_info.cluster_shape = dim3(4, 4, 1);
    hw_info.cluster_shape_fallback = dim3(2, 1, 1);
  }

  typename GemmGrouped::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = RasterOrderOptions::AlongN;

  args = typename GemmGrouped::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes, nullptr},
      {const_cast<const typename GemmGrouped::ElementA **>(ptr_A_list), stride_A_list,
       const_cast<const typename GemmGrouped::ElementB **>(ptr_B_list), stride_B_list,
       const_cast<const typename GemmGrouped::GemmKernel::ElementSF **>(ptr_SFA_list),
       layout_SFA_list,
       const_cast<const typename GemmGrouped::GemmKernel::ElementSF **>(ptr_SFB_list),
       layout_SFB_list},
      {fusion_args,
       accumulate_D ? const_cast<const typename GemmGrouped::ElementC **>(ptr_D_list) : nullptr,
       stride_D_list, ptr_D_list, stride_D_list},
      hw_info,
      scheduler};

  GemmGrouped gemm;

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = GemmGrouped::get_workspace_size(args);
  if (workspaceSize < offset + workspace_size) {  // 16MB limit
    throw std::runtime_error("Calculated workspace size (" +
                             std::to_string(offset + workspace_size) + ") exceeds buffer size (" +
                             std::to_string(workspaceSize) + ")\n");
  }

  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg = "MoE kernel will fail for params. Error: " +
                          std::string(cutlassGetStatusString(can_implement));
    throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
  }

  auto init_status = gemm.initialize(args, reinterpret_cast<char *>(workspace) + offset);
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to initialize cutlass grouped gemm. Error: " +
                          std::string(cutlassGetStatusString(init_status));
    throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
  }

  auto run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to run cutlass grouped gemm. Error: " +
                          std::string(cutlassGetStatusString(run_status));
    throw std::runtime_error("[FT Error][MoE Runner] " + err_msg);
  }
}

void nvte_cutlass_grouped_gemm_wgrad(const NVTETensor *A, const NVTETensor *B, NVTETensor *D,
                                     const int64_t *m_splits, const NVTETensor *bias,
                                     NVTETensor *pre_gelu_out, const int num_gemms, bool transa,
                                     bool transb, NVTETensor *workspace, size_t workspaceSize,
                                     bool accumulate, bool use_split_accumulator, int math_sm_count,
                                     cudaStream_t stream) {
  NVTE_API_CALL(nvte_cutlass_grouped_gemm_wgrad);
  using namespace transformer_engine;
  //   printf("===========nvte_cutlass_grouped_gemm===========\n");
  //   printf("transa: %d, transb: %d\n", transa, transb);
  //   printf("accumulate: %d\n", accumulate);
  NVTE_CHECK(transa && !transb, "wgrad grouped gemm currently only support TN");

  // Process A
  const transformer_engine::Tensor *inputA = convertNVTETensor(A[0]);
  if (transa) {
    NVTE_CHECK(inputA->has_columnwise_data(), "Input A is missing column-wise usage");
  } else {
    NVTE_CHECK(inputA->has_data(), "Input A is missing row-wise usage");
  }
  __nv_fp8_e4m3 *inputA_ptr = transa
                                  ? reinterpret_cast<__nv_fp8_e4m3 *>(inputA->columnwise_data.dptr)
                                  : reinterpret_cast<__nv_fp8_e4m3 *>(inputA->data.dptr);
  __nv_fp8_e8m0 *inputA_SF_ptr =
      transa ? reinterpret_cast<__nv_fp8_e8m0 *>(inputA->columnwise_scale_inv.dptr)
             : reinterpret_cast<__nv_fp8_e8m0 *>(inputA->scale_inv.dptr);

  // Process B
  const transformer_engine::Tensor *inputB = convertNVTETensor(B[0]);
  if (transb) {
    NVTE_CHECK(inputB->has_data(), "Input B is missing row-wise usage");
  } else {
    NVTE_CHECK(inputB->has_columnwise_data(), "Input B is missing column-wise usage");
  }
  __nv_fp8_e4m3 *inputB_ptr = transb
                                  ? reinterpret_cast<__nv_fp8_e4m3 *>(inputB->data.dptr)
                                  : reinterpret_cast<__nv_fp8_e4m3 *>(inputB->columnwise_data.dptr);
  __nv_fp8_e8m0 *inputB_SF_ptr =
      transb ? reinterpret_cast<__nv_fp8_e8m0 *>(inputB->scale_inv.dptr)
             : reinterpret_cast<__nv_fp8_e8m0 *>(inputB->columnwise_scale_inv.dptr);

  // Process D
  void *outputD_ptr_list[num_gemms];
  for (size_t i = 0; i < num_gemms; i++) {
    const transformer_engine::Tensor *outputD = convertNVTETensor(D[i]);
    NVTE_CHECK(outputD->has_data(), "Input D is missing row-wise usage");
    outputD_ptr_list[i] = outputD->data.dptr;
  }

  // Get GEMM shape
  const int gemm_m = transa ? inputA->flat_last_dim() : inputA->flat_first_dim();
  const int gemm_n = transb ? inputB->flat_first_dim() : inputB->flat_last_dim();
  const int total_gemm_k = transa ? inputA->flat_first_dim() : inputA->flat_last_dim();
  //   printf("num_gemms: %d\n", num_gemms);
  //   printf("gemm_m: %d, gemm_n: %d\n", gemm_m, gemm_n);
  //   printf("total_gemm_k: %d\n", total_gemm_k);
  if ((gemm_m & 0x1F) != 0 || (gemm_n & 0xF) != 0) {
    throw std::runtime_error(
        "gemm_m and gemm_n of grouped gemm with variable K must be multiples of 32.");
  }

  // printf("inputA_SF_ptr: \n");
  // Print_tensor<<<1, 1>>>(inputA_SF_ptr, 128, gemm_k/32);
  // cudaDeviceSynchronize();

  //   printf("B_SF_ptr: \n");
  //   Print_tensor<<<1, 1>>>(B_SF_ptr, gemm_n, 256/32);
  //   cudaDeviceSynchronize();

  bool transD = true;  // transD should be the same as transB in fprop, currently is always true
  if (transD) {
    if (accumulate) {
      generic_moe_gemm_wgrad_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e8m0, __nv_fp8_e4m3,
                                            __nv_fp8_e8m0, float, true>(
          inputA_ptr, inputA_SF_ptr, inputB_ptr, inputB_SF_ptr, outputD_ptr_list, gemm_m, gemm_n,
          m_splits, total_gemm_k, num_gemms, accumulate, workspaceSize,
          convertNVTETensor(workspace[0])->data.dptr, stream);
    } else {
      generic_moe_gemm_wgrad_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e8m0, __nv_fp8_e4m3,
                                            __nv_fp8_e8m0, __nv_bfloat16, true>(
          inputA_ptr, inputA_SF_ptr, inputB_ptr, inputB_SF_ptr, outputD_ptr_list, gemm_m, gemm_n,
          m_splits, total_gemm_k, num_gemms, accumulate, workspaceSize,
          convertNVTETensor(workspace[0])->data.dptr, stream);
    }
  } else {
    if (accumulate) {
      generic_moe_gemm_wgrad_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e8m0, __nv_fp8_e4m3,
                                            __nv_fp8_e8m0, float, false>(
          inputA_ptr, inputA_SF_ptr, inputB_ptr, inputB_SF_ptr, outputD_ptr_list, gemm_m, gemm_n,
          m_splits, total_gemm_k, num_gemms, accumulate, workspaceSize,
          convertNVTETensor(workspace[0])->data.dptr, stream);
    } else {
      generic_moe_gemm_wgrad_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e8m0, __nv_fp8_e4m3,
                                            __nv_fp8_e8m0, __nv_bfloat16, false>(
          inputA_ptr, inputA_SF_ptr, inputB_ptr, inputB_SF_ptr, outputD_ptr_list, gemm_m, gemm_n,
          m_splits, total_gemm_k, num_gemms, accumulate, workspaceSize,
          convertNVTETensor(workspace[0])->data.dptr, stream);
    }
  }
}