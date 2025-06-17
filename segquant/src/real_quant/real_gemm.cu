#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_array.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/arch/mma_sm80.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <string>

#ifdef SEGQUANT_MIX
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/threadblock/default_mma.h"
#else
namespace cutlass
{
namespace arch
{

struct OpMultiplyAddFp16Int4 {};

} // namespace arch
} // namespace cutlass
#endif

#include "quantizer.cuh"
#include "dequantizer.cuh"
#include "seg_utils.h"

#define BLOCK_SIZE 128

#define AT_DISPATCH_CASE_FLOATING_TYPES(...)                                                       \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                                             \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                                              \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                                                \
    AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

template <typename T>
struct CutlassElementOutputType;

template <>
struct CutlassElementOutputType<__nv_fp8_e4m3> {
    using type = cutlass::float_e4m3_t;
};

template <>
struct CutlassElementOutputType<int8_t> {
    using type = int8_t;
};

template <>
struct CutlassElementOutputType<cutlass::int4b_t> {
    using type = cutlass::int4b_t;
};

template <>
struct CutlassElementOutputType<at::Half> {
    using type = cutlass::half_t;
};

template <>
struct CutlassElementOutputType<at::BFloat16> {
    using type = cutlass::bfloat16_t;
};

template <>
struct CutlassElementOutputType<float> {
    using type = float;
};

template <typename A, typename B>
struct CutlassElementAccumulatorType;

template <>
struct CutlassElementAccumulatorType<int8_t, int8_t> {
    using type = int32_t;
};

template <>
struct CutlassElementAccumulatorType<int8_t, cutlass::int4b_t> {
    using type = int32_t;
};

template <>
struct CutlassElementAccumulatorType<cutlass::int4b_t, cutlass::int4b_t> {
    using type = int32_t;
};

template <>
struct CutlassElementAccumulatorType<__nv_fp8_e4m3, __nv_fp8_e4m3> {
    using type = float;
};

template <typename B>
struct CutlassElementAccumulatorType<at::Half, B> {
    using type = float;
};

template <typename B>
struct CutlassElementAccumulatorType<at::BFloat16, B> {
    using type = float;
};

template <typename A, typename B>
struct CutlassArchType {
    using arch = cutlass::arch::Sm80;
};
template <>
struct CutlassArchType<__nv_fp8_e4m3, __nv_fp8_e4m3> {
    using arch = cutlass::arch::Sm89;
};

template <typename A, typename B>
struct OpType {
    using type = cutlass::arch::OpClassTensorOp;
};

template <typename A, typename B>
struct ShapeType;

template <>
struct ShapeType<int8_t, int8_t> {
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
};

template <>
struct ShapeType<cutlass::int4b_t, cutlass::int4b_t> {
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 128>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
};

template <>
struct ShapeType<int8_t, cutlass::int4b_t> {
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
};

template <>
struct ShapeType<at::Half, int8_t> {
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
};

template <>
struct ShapeType<__nv_fp8_e4m3, __nv_fp8_e4m3> {
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
};

template <>
struct ShapeType<at::Half, cutlass::int4b_t> {
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
};

template <typename A, typename B, bool IsSame = std::is_same<A, B>::value>
struct StagesImpl;
template <typename A, typename B>
struct StagesImpl<A, B, true> {
    static constexpr int value = 3;
};
template <typename A, typename B>
struct StagesImpl<A, B, false> {
    static constexpr int value = 4;
};
template <typename A, typename B>
struct Stages : StagesImpl<A, B> {};

template <typename A, typename B>
struct AddType {
    using type = cutlass::arch::OpMultiplyAdd;
};
template <>
struct AddType<int8_t, int8_t> {
    using type = cutlass::arch::OpMultiplyAddSaturate;
};

template <>
struct AddType<cutlass::int4b_t, cutlass::int4b_t> {
    using type = cutlass::arch::OpMultiplyAddSaturate;
};

template <>
struct AddType<int8_t, cutlass::int4b_t> {
    using type = cutlass::arch::OpMultiplyAddMixedInputUpcast;
};

template <>
struct AddType<at::Half, int8_t> {
    using type = cutlass::arch::OpMultiplyAddMixedInputUpcast;
};

template <>
struct AddType<at::Half, cutlass::int4b_t> {
    using type = cutlass::arch::OpMultiplyAddFp16Int4;
};

template <>
struct AddType<__nv_fp8_e4m3, __nv_fp8_e4m3> {
    using type = cutlass::arch::OpMultiplyAdd;
    // fast but precision not ok
    // using type = cutlass::arch::OpMultiplyAddFastAccum;
};

template <typename A, typename B>
struct AlignNums {
    static constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<A>::value;
    static constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<B>::value;
};

//////////////////////////////////////////////////////////////////////
////////// CUTLASS Kernels
//////////////////////////////////////////////////////////////////////
template <typename AType, typename BType, typename CType>
void launch_universal_gemm_scaled(
    cudaStream_t stream,
    void const* A, void const* B, void* C,
    int M, int N, int K,
    cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm,
    float alpha = 1.0f, float beta = 0.0f,
    int batch_count=1,
    int64_t batch_stride_A=0,
    int64_t batch_stride_B=0,
    int64_t batch_stride_C=0,
    int64_t batch_stride_D=0) {

    using ElementInputA = typename CutlassElementOutputType<AType>::type;
    using ElementInputB = typename CutlassElementOutputType<BType>::type;
    using ElementOutput = typename CutlassElementOutputType<CType>::type;
    using ElementAccumulator = typename CutlassElementAccumulatorType<AType, BType>::type;
    using ElementCompute = float;
    using CutlassOp = typename OpType<AType, BType>::type;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    using CutlassArch = typename CutlassArchType<AType, BType>::arch;
    constexpr int NumPerThread = 128 / cutlass::sizeof_bits<ElementOutput>::value;

    if (K % AlignNums<AType, BType>::kAlignmentA != 0) {
        throw std::runtime_error(
            "K (" + std::to_string(K) + ") must be divisible by alignmentA (" +
            std::to_string(AlignNums<AType, BType>::kAlignmentA) + ") for AType");
    }
    
    if (K % AlignNums<AType, BType>::kAlignmentB != 0) {
        throw std::runtime_error(
            "K (" + std::to_string(K) + ") must be divisible by alignmentB (" +
            std::to_string(AlignNums<AType, BType>::kAlignmentB) + ") for BType");
    }

    using GemmUniversal = cutlass::gemm::device::GemmUniversal<
        ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        CutlassOp,
        CutlassArch,
        typename ShapeType<AType, BType>::ThreadblockShape,
        typename ShapeType<AType, BType>::WarpShape,
        typename ShapeType<AType, BType>::InstructionShape,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            NumPerThread,
            ElementAccumulator,
            ElementCompute
        >,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        Stages<AType, BType>::value,
        AlignNums<AType, BType>::kAlignmentA,
        AlignNums<AType, BType>::kAlignmentB,
        typename AddType<AType, BType>::type
    >;

    typename GemmUniversal::Arguments args{
        mode,
        {M, N, K},
        batch_count,
        {alpha, beta},
        reinterpret_cast<void const *>(A),
        reinterpret_cast<void const *>(B),
        reinterpret_cast<void const *>(C),
        reinterpret_cast<void *>(C),
        batch_stride_A,
        batch_stride_B,
        batch_stride_C,
        batch_stride_D,
        K,
        K,
        N,
        N
    };

    GemmUniversal gemm;
    cutlass::Status status = gemm(args, nullptr, stream);
    cudaError_t err = cudaGetLastError();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string("GEMM launch failed, cutlass status: ") + cutlass::cutlassGetStatusString(status) + ", cuda error: " + cudaGetErrorString(err));
    }
}


template <typename AType, typename BType, typename CType>
void launch_gemm_scaled(
    const AType *A, const BType *B, CType *C,
    int M, int N, int K,
    float scale_x, float scale_w,
    float beta,
    cudaStream_t stream) {

    float scale = 1.0f / (scale_x * scale_w);
    launch_universal_gemm_scaled<AType, BType, CType>(
        stream,
        A, B, C,
        M, N, K,
        cutlass::gemm::GemmUniversalMode::kGemm,
        scale, beta);
}

template <typename AType, typename BType, typename CType>
void launch_batched_gemm_scaled(
    const AType *A, const BType *B, CType *C,
    int M, int N, int K,
    int batch_count,
    float scale_x, float scale_w,
    float beta,
    cudaStream_t stream) {

    float scale = 1.0f / (scale_x * scale_w);
    launch_universal_gemm_scaled<AType, BType, CType>(
        stream,
        A, B, C,
        M, N, K,
        cutlass::gemm::GemmUniversalMode::kBatched,
        scale, beta,
        batch_count,
        M * K,
        0,
        M * N,
        M * N
    );
}

template <typename AType, typename BType, typename CType>
void launch_array_gemm_scaled(
    const AType** A, const BType** B, CType** C,
    int M, int N, int K,
    int batch_count,
    cudaStream_t stream) {

    launch_universal_gemm_scaled<AType, BType, CType>(
        stream,
        A, B, C,
        M, N, K,
        cutlass::gemm::GemmUniversalMode::kArray,
        1.0f, 0.0f,
        batch_count
    );
}


//////////////////////////////////////////////////////////////////////
////////// Real Quantize weight
//////////////////////////////////////////////////////////////////////
template<typename T>
void real_quantized_quantize_weights(at::Tensor weights, at::Tensor outputs, float scale_w);

template <>
void real_quantized_quantize_weights<__nv_fp8_e4m3>(at::Tensor weights, at::Tensor outputs, float scale_w) {
    size_t numel = weights.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "real_quantize_scaled_kernel", [&] {
        real_quantize_scaled_kernel<scalar_t, __nv_fp8_e4m3, uint8_t><<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            weights.data_ptr<scalar_t>(), scale_w, numel, 
            outputs.data_ptr<uint8_t>()
        );
    });
}

template <>
void real_quantized_quantize_weights<int8_t>(at::Tensor weights, at::Tensor outputs, float scale_w) {
    size_t numel = weights.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "real_quantize_scaled_kernel", [&] {
        real_quantize_scaled_kernel<scalar_t, int8_t, int8_t><<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            weights.data_ptr<scalar_t>(), scale_w, numel, 
            outputs.data_ptr<int8_t>()
        );
    });
}

template <>
void real_quantized_quantize_weights<cutlass::int4b_t>(at::Tensor weights, at::Tensor outputs, float scale_w) {
    size_t numel = weights.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "real_quantize_scaled_kernel", [&] {
        real_quantize_scaled_kernel<scalar_t, cutlass::int4b_t, uint8_t><<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            weights.data_ptr<scalar_t>(), scale_w, numel, 
            outputs.data_ptr<uint8_t>()
        );
    });
}

//////////////////////////////////////////////////////////////////////
////////// Real Quantize weight: axis version
//////////////////////////////////////////////////////////////////////
template<typename T>
void real_quantized_quantize_weights(at::Tensor weights, at::Tensor outputs, at::Tensor scale_w);

template <>
void real_quantized_quantize_weights<int8_t>(at::Tensor weights, at::Tensor outputs, at::Tensor scale_w) {
    size_t numel = weights.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto last_features = weights.sizes()[1]; // in

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "real_quantize_scaled_kernel", [&] {
        real_quantize_scaled_kernel<scalar_t, int8_t, int8_t><<<
            (numel + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4), BLOCK_SIZE, 0, stream>>>(
            weights.data_ptr<scalar_t>(),
            scale_w.data_ptr<float>(),
            last_features,
            numel,
            outputs.data_ptr<int8_t>());
    });
}

template <>
void real_quantized_quantize_weights<__nv_fp8_e4m3>(at::Tensor weights, at::Tensor outputs, at::Tensor scale_w) {
    size_t numel = weights.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto last_features = weights.sizes()[1];

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "real_quantize_scaled_kernel", [&] {
        real_quantize_scaled_kernel<scalar_t, __nv_fp8_e4m3, uint8_t><<<
            (numel + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4), BLOCK_SIZE, 0, stream>>>(
            weights.data_ptr<scalar_t>(),
            scale_w.data_ptr<float>(),
            last_features,
            numel,
            outputs.data_ptr<uint8_t>());
    });
}

template <>
void real_quantized_quantize_weights<cutlass::int4b_t>(at::Tensor weights, at::Tensor outputs, at::Tensor scale_w) {
    size_t numel = weights.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto last_features = weights.sizes()[1];

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "real_quantize_scaled_kernel", [&] {
        real_quantize_scaled_kernel<scalar_t, cutlass::int4b_t, uint8_t><<<
            (numel + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4), BLOCK_SIZE, 0, stream>>>(
            weights.data_ptr<scalar_t>(),
            scale_w.data_ptr<float>(),
            last_features,
            numel,
            outputs.data_ptr<uint8_t>());
    });
}

//////////////////////////////////////////////////////////////////////
////////// Call GEMM Pipepine
//////////////////////////////////////////////////////////////////////
template<typename input_type, typename weight_type, typename scale_x_type, typename scale_w_type>
at::Tensor real_quantized_gemm_scaled(at::Tensor inputs, at::Tensor weights, scale_x_type scale_x, scale_w_type scale_w) {
    auto inputs_sizes = inputs.sizes();
    auto weights_sizes = weights.sizes();
    auto input_rank = inputs_sizes.size();
    int64_t batch_count = 1;
    if (input_rank > 2) {
        // batch gemm
        TORCH_CHECK(inputs.is_contiguous(), "inputs must be contiguous");
        for (int64_t i = 0; i < input_rank - 2; ++i) {
            batch_count *= inputs.size(i);
        }
    }
    int64_t M = inputs_sizes[input_rank - 2]; // (..., M, K)
    int64_t K = inputs_sizes[input_rank - 1];
    int64_t N = std::is_same<weight_type, cutlass::int4b_t>::value ? weights_sizes[0] * 2 / K : weights_sizes[0]; // (N, K)
    if constexpr (!std::is_same<weight_type, cutlass::int4b_t>::value) {
        if (weights_sizes[1] != K) {
            std::ostringstream oss;
            oss << "real_quantized_gemm_scaled: weights tensor must have shape [N, K], but got weights shape ["
                << weights_sizes[0] << ", " << weights_sizes[1] << "] and inputs shape [..., "
                << inputs_sizes[input_rank - 2] << ", " << inputs_sizes[input_rank - 1] << "]";
            throw std::runtime_error(oss.str());
        }
    }

    if constexpr (std::is_same<input_type, at::Half>::value) {
        if constexpr (std::is_same<scale_x_type, float>::value) {
            scale_x = 1.0f;
        }
        else {
            scale_x = at::ones_like(scale_x);
        }
    }

    // create output tensor
    auto options = inputs.options();
    std::vector<int64_t> output_sizes(inputs_sizes.begin(), inputs_sizes.end() - 1);
    output_sizes.push_back(N);
    auto outputs = at::empty(output_sizes, options);
    // quantized tensors
    using StoreInputType = typename StoreType<input_type>::type;
    using StoreWeightType = typename StoreType<weight_type>::type;
    using CUDAStoreInputType = typename CUDAStoreType<input_type>::type;

    auto Xq_tensor = std::is_same<input_type, cutlass::int4b_t>::value
        ? at::empty({(inputs.numel() + 1) / 2}, options.dtype(c10::CppTypeToScalarType<StoreInputType>::value))
        : at::empty_like(inputs, options.dtype(c10::CppTypeToScalarType<StoreInputType>::value));
    input_type* Xq = reinterpret_cast<input_type*>(Xq_tensor.template data_ptr<StoreInputType>());
    weight_type* Wq = reinterpret_cast<weight_type*>(weights.template data_ptr<StoreWeightType>());

    auto stream = c10::cuda::getCurrentCUDAStream();
    size_t numel_x = inputs.numel();
    if constexpr (std::is_same<scale_x_type, float>::value) {
        // axis none
        AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "real_quantize_scaled_kernel", [&] {
            real_quantize_scaled_kernel<scalar_t, input_type, CUDAStoreInputType><<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
                inputs.data_ptr<scalar_t>(), scale_x, numel_x, reinterpret_cast<CUDAStoreInputType*>(Xq_tensor.template data_ptr<StoreInputType>())
            );
        });
    }
    else {
        // axis = -1
        AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "real_quantize_scaled_kernel", [&] {
            real_quantize_scaled_kernel<scalar_t, input_type, CUDAStoreInputType><<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
                inputs.data_ptr<scalar_t>(), scale_x.template data_ptr<float>(), K, numel_x, reinterpret_cast<CUDAStoreInputType*>(Xq_tensor.template data_ptr<StoreInputType>())
            );
        });
    }

    if constexpr (std::is_same<scale_x_type, float>::value && std::is_same<scale_w_type, float>::value) {
        // dequant is unnecessary
        if (batch_count > 1) {
            // batched gemm
            AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "launch_batched_gemm_scaled", [&] {
                launch_batched_gemm_scaled<input_type, weight_type, scalar_t>(Xq, Wq, outputs.data_ptr<scalar_t>(), M, N, K, batch_count, scale_x, scale_w, 0.0f, stream);
            });
        }
        else {
            // single gemm
            AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "launch_gemm_scaled", [&] {
                launch_gemm_scaled<input_type, weight_type, scalar_t>(Xq, Wq, outputs.data_ptr<scalar_t>(), M, N, K, scale_x, scale_w, 0.0f, stream);
            });
        }
    }
    else {
        auto tmp_options = outputs.options().dtype(torch::kFloat32);
        auto Yq = at::empty_like(outputs, tmp_options);
        size_t numel_y = outputs.numel();

        float new_scale_x = 1.0f;
        if constexpr (std::is_same<scale_x_type, float>::value) {
            new_scale_x = scale_x;
        }

        float new_scale_w = 1.0f;
        if constexpr (std::is_same<scale_w_type, float>::value) {
            new_scale_w = scale_w;
        }

        if (batch_count > 1) {
            // batched gemm
            AT_DISPATCH_FLOATING_TYPES(Yq.scalar_type(), "launch_batched_gemm_scaled", [&] {
                launch_batched_gemm_scaled<input_type, weight_type, scalar_t>(Xq, Wq, Yq.data_ptr<scalar_t>(), M, N, K, batch_count, new_scale_x, new_scale_w, 0.0f, stream);
            });
        }
        else {
            // single gemm
            AT_DISPATCH_FLOATING_TYPES(Yq.scalar_type(), "launch_gemm_scaled", [&] {
                launch_gemm_scaled<input_type, weight_type, scalar_t>(Xq, Wq, Yq.data_ptr<scalar_t>(), M, N, K, new_scale_x, new_scale_w, 0.0f, stream);
            });
        }

        if constexpr (std::is_same<scale_x_type, float>::value) {
            // dequant with scale_w
            AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "real_dequantize_scaled_kernel", [&] {
                real_dequantize_scaled_kernel<scalar_t, 0><<<numel_y / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
                    Yq.data_ptr<float>(),
                    outputs.data_ptr<scalar_t>(), 
                    scale_w.template data_ptr<float>(),
                    N,
                    numel_y
                );
            });
        }
        else if constexpr (std::is_same<scale_w_type, float>::value) {
            // dequant with scale_x
            AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "real_dequantize_scaled_kernel", [&] {
                real_dequantize_scaled_kernel<scalar_t, 1><<<numel_y / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
                    Yq.data_ptr<float>(),
                    outputs.data_ptr<scalar_t>(), 
                    scale_x.template data_ptr<float>(),
                    N,
                    numel_y
                );
            });
        }
        else {
            // dequant with scale_x and scale_w
            AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "real_dequantize_scaled_kernel", [&] {
                real_dequantize_scaled_kernel<scalar_t><<<numel_y / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
                    Yq.data_ptr<float>(),
                    outputs.data_ptr<scalar_t>(), 
                    scale_x.template data_ptr<float>(), scale_w.template data_ptr<float>(),
                    N,
                    numel_y
                );
            });
        }
    }

    return outputs;
}

template<typename input_type, typename weight_type, typename scale_x_type, typename scale_w_type>
at::Tensor real_quantized_gemm_dual_scaled(at::Tensor inputs, at::Tensor weights,
                                scale_x_type pos_scale_x, scale_x_type neg_scale_x,
                                scale_w_type scale_w) {
    
    auto inputs_sizes = inputs.sizes();
    auto weights_sizes = weights.sizes();
    auto input_rank = inputs_sizes.size();
    int64_t batch_count = 1;
    if (input_rank > 2) {
        // batch gemm
        TORCH_CHECK(inputs.is_contiguous(), "inputs must be contiguous");
        for (int64_t i = 0; i < input_rank - 2; ++i) {
            batch_count *= inputs.size(i);
        }
    }
    int64_t M = inputs_sizes[input_rank - 2]; // (..., M, K)
    int64_t K = inputs_sizes[input_rank - 1];
    int64_t N = std::is_same<weight_type, cutlass::int4b_t>::value ? weights_sizes[0] * 2 / K : weights_sizes[0]; // (N, K)
    if constexpr (!std::is_same<weight_type, cutlass::int4b_t>::value) {
        if (weights_sizes[1] != K) {
            std::ostringstream oss;
            oss << "real_quantized_e4m3fy_gemm_dual_scaled: weights tensor must have shape [N, K], but got weights shape ["
                << weights_sizes[0] << ", " << weights_sizes[1] << "] and inputs shape [..., "
                << inputs_sizes[input_rank - 2] << ", " << inputs_sizes[input_rank - 1] << "]";
            throw std::runtime_error(oss.str());
        }
    }

    if constexpr (std::is_same<input_type, at::Half>::value) {
        if constexpr (std::is_same<scale_x_type, float>::value) {
            pos_scale_x = neg_scale_x = 1.0f;
        }
        else {
            pos_scale_x = at::ones_like(pos_scale_x);
            neg_scale_x = at::ones_like(neg_scale_x);
        }
    }

    // create output tensor
    auto options = inputs.options();
    std::vector<int64_t> output_sizes(inputs_sizes.begin(), inputs_sizes.end() - 1);
    output_sizes.push_back(N);
    auto outputs = at::empty(output_sizes, options);
    using StoreInputType = typename StoreType<input_type>::type;
    using StoreWeightType = typename StoreType<weight_type>::type;
    using CUDAStoreInputType = typename CUDAStoreType<input_type>::type;

    auto Xp_tensor = std::is_same<input_type, cutlass::int4b_t>::value
        ? at::empty({(inputs.numel() + 1) / 2}, options.dtype(c10::CppTypeToScalarType<StoreInputType>::value))
        : at::empty_like(inputs, options.dtype(c10::CppTypeToScalarType<StoreInputType>::value));
    
    auto Xn_tensor = std::is_same<input_type, cutlass::int4b_t>::value
        ? at::empty({(inputs.numel() + 1) / 2}, options.dtype(c10::CppTypeToScalarType<StoreInputType>::value))
        : at::empty_like(inputs, options.dtype(c10::CppTypeToScalarType<StoreInputType>::value));

    input_type* Xp = reinterpret_cast<input_type*>(Xp_tensor.template data_ptr<StoreInputType>());
    input_type* Xn = reinterpret_cast<input_type*>(Xn_tensor.template data_ptr<StoreInputType>());
    weight_type* Wq = reinterpret_cast<weight_type*>(weights.template data_ptr<StoreWeightType>());

    auto stream = c10::cuda::getCurrentCUDAStream();
    size_t numel_x = inputs.numel();
    // X = Xp + Xn
    if constexpr (std::is_same<scale_x_type, float>::value) {
        // axis none
        AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "real_quantize_dual_scaled_kernel", [&] {
            real_quantize_dual_scaled_kernel<scalar_t, input_type, CUDAStoreInputType><<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
                inputs.data_ptr<scalar_t>(), pos_scale_x, neg_scale_x, numel_x,
                reinterpret_cast<CUDAStoreInputType*>(Xp_tensor.template data_ptr<StoreInputType>()),
                reinterpret_cast<CUDAStoreInputType*>(Xn_tensor.template data_ptr<StoreInputType>())
            );
        });
    }
    else {
        // axis = -1
        AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "real_quantize_dual_scaled_kernel", [&] {
            real_quantize_dual_scaled_kernel<scalar_t, input_type, CUDAStoreInputType><<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
                inputs.data_ptr<scalar_t>(), pos_scale_x.template data_ptr<float>(), neg_scale_x.template data_ptr<float>(), K,
                numel_x,
                reinterpret_cast<CUDAStoreInputType*>(Xp_tensor.template data_ptr<StoreInputType>()),
                reinterpret_cast<CUDAStoreInputType*>(Xn_tensor.template data_ptr<StoreInputType>())
            );
        });
    }

    auto tmp_options = outputs.options().dtype(torch::kFloat32);
    auto Y_p = at::empty_like(outputs, tmp_options);
    auto Y_n = at::empty_like(outputs, tmp_options);
    std::vector<void const *> ptr_A_batched_host;
    std::vector<void const *> ptr_B_batched_host;
    std::vector<void*> ptr_C_batched_host;

    for (int64_t i = 0; i < batch_count; ++i) {
        // For batched gemm, we need to prepare pointers for each batch
        ptr_A_batched_host.push_back(reinterpret_cast<void const *>(Xp + i * M * K));
        ptr_A_batched_host.push_back(reinterpret_cast<void const *>(Xn + i * M * K));
        ptr_B_batched_host.push_back(reinterpret_cast<void const *>(Wq));
        ptr_B_batched_host.push_back(reinterpret_cast<void const *>(Wq));
        AT_DISPATCH_FLOATING_TYPES(Y_p.scalar_type(), "prepare_ptr_C_batched_host", [&] {
            ptr_C_batched_host.push_back(reinterpret_cast<void*>(Y_p.data_ptr<scalar_t>() + i * M * N));
            ptr_C_batched_host.push_back(reinterpret_cast<void*>(Y_n.data_ptr<scalar_t>() + i * M * N));
        });
    }

    // Allocate device memory for batched GEMM
    cutlass::DeviceAllocation<void const *> ptr_A_batched;
    cutlass::DeviceAllocation<void const *> ptr_B_batched;
    cutlass::DeviceAllocation<void       *> ptr_C_batched;

    ptr_A_batched.reset(ptr_A_batched_host.size());
    ptr_B_batched.reset(ptr_A_batched_host.size());
    ptr_C_batched.reset(ptr_A_batched_host.size());

    ptr_A_batched.copy_from_host(ptr_A_batched_host.data());
    ptr_B_batched.copy_from_host(ptr_B_batched_host.data());
    ptr_C_batched.copy_from_host(ptr_C_batched_host.data());

    AT_DISPATCH_FLOATING_TYPES(Y_p.scalar_type(), "launch_array_gemm_scaled", [&] {
        launch_array_gemm_scaled<input_type, weight_type, scalar_t>(
            reinterpret_cast<const input_type**>(ptr_A_batched.get()),
            reinterpret_cast<const weight_type**>(ptr_B_batched.get()),
            reinterpret_cast<scalar_t**>(ptr_C_batched.get()),
            M, N, K,
            2 * batch_count,
            stream);
    });

    // Y = Y_p + Y_n
    size_t numel_y = outputs.numel();

    if constexpr (std::is_same<scale_x_type, float>::value && std::is_same<scale_w_type, float>::value) {
        // input axis = None, weight axis = None
        AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "real_dequantize_dual_scaled_kernel", [&] {
            real_dequantize_dual_scaled_kernel<scalar_t><<<numel_y / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
                Y_p.data_ptr<float>(), 
                Y_n.data_ptr<float>(), 
                outputs.data_ptr<scalar_t>(), 
                pos_scale_x, neg_scale_x,
                scale_w, 
                numel_y
            );
        });
    }
    else if constexpr (std::is_same<scale_w_type, float>::value) {
        // scale x is tensor, row_flag = 1
        AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "real_dequantize_dual_scaled_kernel", [&] {
            real_dequantize_dual_scaled_kernel<scalar_t><<<numel_y / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
                Y_p.data_ptr<float>(), 
                Y_n.data_ptr<float>(), 
                outputs.data_ptr<scalar_t>(), 
                pos_scale_x.template data_ptr<float>(), neg_scale_x.template data_ptr<float>(),
                scale_w,
                N,
                numel_y
            );
        });
    }
    else if constexpr (std::is_same<scale_x_type, float>::value) {
        // scale w is tensor, row_flag = 0
        AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "real_dequantize_dual_scaled_kernel", [&] {
            real_dequantize_dual_scaled_kernel<scalar_t><<<numel_y / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
                Y_p.data_ptr<float>(), 
                Y_n.data_ptr<float>(), 
                outputs.data_ptr<scalar_t>(), 
                pos_scale_x, neg_scale_x,
                scale_w.template data_ptr<float>(),
                N,
                numel_y
            );
        });
    }
    else {
        // all tensor
        AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "real_dequantize_dual_scaled_kernel", [&] {
            real_dequantize_dual_scaled_kernel<scalar_t><<<numel_y / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
                Y_p.data_ptr<float>(), 
                Y_n.data_ptr<float>(), 
                outputs.data_ptr<scalar_t>(), 
                pos_scale_x.template data_ptr<float>(), neg_scale_x.template data_ptr<float>(),
                scale_w.template data_ptr<float>(),
                N,
                numel_y
            );
        });
    }

    return outputs;
}



//////////////////////////////////////////////////////////////////////
////////// Template Instantiation
//////////////////////////////////////////////////////////////////////
#define INT8_AW_PAIRS \
    X(int8_t, int8_t)

#define INT4_AW_PAIRS \
    X(cutlass::int4b_t, cutlass::int4b_t)

#define FP8_AW_PAIRS \
    X(__nv_fp8_e4m3, __nv_fp8_e4m3)

#define MIX_AW_PAIRS \
    X(at::Half, cutlass::int4b_t)

#define INSTANTIATE(A, W, SX, SW) \
    template at::Tensor real_quantized_gemm_scaled<A, W, SX, SW>(at::Tensor, at::Tensor, SX, SW); \
    template at::Tensor real_quantized_gemm_dual_scaled<A, W, SX, SW>(at::Tensor, at::Tensor, SX, SX, SW);

#define EXPAND_SW(A, W, SX) \
    INSTANTIATE(A, W, SX, float) \
    INSTANTIATE(A, W, SX, at::Tensor)

#define EXPAND_SX(A, W) \
    EXPAND_SW(A, W, float) \
    EXPAND_SW(A, W, at::Tensor)

#define X(A, W) EXPAND_SX(A, W)
#ifdef SEGQUANT_INT8
INT8_AW_PAIRS
#endif
#ifdef SEGQUANT_FP8
FP8_AW_PAIRS
#endif
#ifdef SEGQUANT_INT4
INT4_AW_PAIRS
#endif
#ifdef SEGQUANT_MIX
MIX_AW_PAIRS
#endif
#undef X