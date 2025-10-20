#pragma once

#include <ATen/ATen.h>
#include "cutlass/epilogue/threadblock/broadcast_load_epilogue.hpp"

/*
   This file defines custom epilogues for fusing channel scales, token scales,
   bias, and activation zero-points onto a GEMM operation using the
   CUTLASS 2.x API, for sm80 (Ampere) NVIDIA GPUs.

   Epilogues must contain a public type named EVTCompute of type Sm80EVT,
   as well as a static prepare_args function that constructs an
   EVTCompute::Arguments struct.
*/

using namespace cute;

/*
 * This class provides the common load descriptors for the
 * ScaledEpilogue[...] classes
 */
template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogueBase {
 protected:
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  template <typename T>
  using ColOrScalarLoad =
      cutlass::epilogue::threadblock::VisitorColOrScalarBroadcast<
          OutputTileThreadMap, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowOrScalarLoad =
      cutlass::epilogue::threadblock::VisitorRowOrScalarBroadcast<
          OutputTileThreadMap, T, Stride<Int<0>, Int<1>, Int<0>>>;

  template <typename T>
  using ColLoad = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, T, Stride<Int<0>, Int<1>, Int<0>>>;

  template <typename T>
  using RowOrZeroLoad =
      cutlass::epilogue::threadblock::VisitorRowOrZeroBroadcast<
          OutputTileThreadMap, T, Stride<Int<0>, Int<1>, Int<0>>>;

  // This utility function constructs the arguments for the load descriptors
  // from a tensor. It can handle both row and column, as well as row/column or
  // scalar cases.
  template <typename Descriptor, typename T>
  static auto args_from_tensor(at::Tensor const& tensor) {
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = static_cast<T*>(tensor.data_ptr());
    if constexpr (std::is_same_v<Descriptor, ColOrScalarLoad<T>> ||
                  std::is_same_v<Descriptor, RowOrScalarLoad<T>>) {
      return Arguments{data_ptr, tensor.numel() != 1};
    } else {
      // it would technically work but no use case as data_ptr is never nullptr
      static_assert(!std::is_same_v<Descriptor, RowOrZeroLoad<T>>);
      return Arguments{data_ptr};
    }
  }

  // This overload handles the case where there might not be a tensor, in which
  // case a nullptr is passed and a constant (0) is used.
  template <typename Descriptor, typename T>
  static auto args_from_tensor(std::optional<at::Tensor> const& tensor) {
    static_assert(std::is_same_v<Descriptor, RowOrZeroLoad<T>>);
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = tensor ? static_cast<T*>(tensor->data_ptr()) : nullptr;
    return Arguments{data_ptr};
  }
};

/*
 This epilogue function defines a quantized GEMM operation similar to
 torch._scaled_mm.

 A and B may be both either int8 or fp8_e4m3. A can be quantized per-tensor or
 per-row. B can be quantized per-tensor or per-column.
 Any combination of per-tensor and per-row or column is supported.
 A and B must have symmetric quantization (zero point == 0).

 So the GEMM operation is D = (a_scales * A) (b_scales * B), where the
 scales are applied elementwise with numpy-style broadcasting.

 ScaleA and ScaleB define the epilogue functions that apply the scales for
 the A and B operands respectively. These scales may be either per-tensor or
 per row or column.
*/
template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogue
    : private ScaledEpilogueBase<ElementD, OutputTileThreadMap> {
 private:
  using SUPER = ScaledEpilogueBase<ElementD, OutputTileThreadMap>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;

  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(at::Tensor const& a_scales,
                                   at::Tensor const& b_scales) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, {}};
  }
};