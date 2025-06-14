#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_array.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/util/device_memory.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <string>
#include "quantizer.cuh"
#include "dequantizer.cuh"

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

template <typename T>
struct CutlassElementAccumulatorType;

template <>
struct CutlassElementAccumulatorType<int8_t> {
    using type = int32_t;
};

template <>
struct CutlassElementAccumulatorType<__nv_fp8_e4m3> {
    using type = float;
};

template <>
struct CutlassElementAccumulatorType<cutlass::int4b_t> {
    using type = int32_t;
};

template <typename T>
struct CutlassArchType;

template <>
struct CutlassArchType<int8_t> {
    using arch = cutlass::arch::Sm80;
};

template <>
struct CutlassArchType<__nv_fp8_e4m3> {
    using arch = cutlass::arch::Sm89;
};

template <>
struct CutlassArchType<cutlass::int4b_t> {
    using arch = cutlass::arch::Sm80;
};

template <typename T>
struct StoreType;

template <>
struct StoreType<int8_t> {
    using type = int8_t;
};

template <>
struct StoreType<__nv_fp8_e4m3> {
    using type = uint8_t;
};

template <>
struct StoreType<cutlass::int4b_t> {
    using type = uint8_t;
};

template <typename T>
struct ShapeType;

template <>
struct ShapeType<int8_t> {
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
};

template <>
struct ShapeType<__nv_fp8_e4m3> {
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
};

template <>
struct ShapeType<cutlass::int4b_t> {
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 128>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
};

//////////////////////////////////////////////////////////////////////
////////// CUTLASS Kernels
//////////////////////////////////////////////////////////////////////
template <typename ABType, typename CType>
void launch_gemm_scaled(
    const ABType *A, const ABType *B, CType *C,
    int M, int N, int K,
    float scale_x, float scale_w,
    float beta,
    cudaStream_t stream) {

    using ElementInputA = typename CutlassElementOutputType<ABType>::type;
    using ElementInputB = typename CutlassElementOutputType<ABType>::type;
    using ElementOutput = typename CutlassElementOutputType<CType>::type;
    using ElementAccumulator = typename CutlassElementAccumulatorType<ABType>::type;
    using ElementCompute = float;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    using CutlassArch = typename CutlassArchType<ABType>::arch;
    constexpr int NumPerThread = 128 / cutlass::sizeof_bits<ElementOutput>::value;
    constexpr int AlignNum = 16;

    if (K % AlignNum != 0) {
        throw std::runtime_error("K dimension (" + std::to_string(K) + ") is not aligned to " + std::to_string(AlignNum));
    }
    if (N % AlignNum != 0) {
        throw std::runtime_error("N dimension (" + std::to_string(N) + ") is not aligned to " + std::to_string(AlignNum));
    }

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        CutlassArch,
        typename ShapeType<ABType>::ThreadblockShape,
        typename ShapeType<ABType>::WarpShape,
        typename ShapeType<ABType>::InstructionShape,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            NumPerThread,
            ElementAccumulator,
            ElementCompute
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3,
        AlignNum,
        AlignNum
    >;

    Gemm gemm;

    float scale = 1.0f / (scale_x * scale_w);

    typename Gemm::Arguments args{
        {M, N, K},
        {reinterpret_cast<const ElementInputA*>(A), K},
        {reinterpret_cast<const ElementInputB*>(B), K},
        {reinterpret_cast<const ElementOutput*>(C), N},
        {reinterpret_cast<ElementOutput*>(C), N},
        {scale, beta}
    };

    cutlass::Status status = gemm(args, nullptr, stream);
    cudaError_t err = cudaGetLastError();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string("GEMM launch failed, cutlass status: ") + cutlass::cutlassGetStatusString(status) + ", cuda error: " + cudaGetErrorString(err));
    }
}

template <typename ABType, typename CType>
void launch_batched_gemm_scaled(
    const ABType *A, const ABType *B, CType *C,
    int M, int N, int K,
    int batch_count,
    float scale_x, float scale_w,
    float beta,
    cudaStream_t stream) {

    using ElementInputA = typename CutlassElementOutputType<ABType>::type;
    using ElementInputB = typename CutlassElementOutputType<ABType>::type;
    using ElementOutput = typename CutlassElementOutputType<CType>::type;
    using ElementAccumulator = typename CutlassElementAccumulatorType<ABType>::type;
    using ElementCompute = float;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    using CutlassArch = typename CutlassArchType<ABType>::arch;
    constexpr int NumPerThread = 128 / cutlass::sizeof_bits<ElementOutput>::value;
    constexpr int AlignNum = 16;

    if (K % AlignNum != 0) {
        throw std::runtime_error("K dimension (" + std::to_string(K) + ") is not aligned to " + std::to_string(AlignNum));
    }
    if (N % AlignNum != 0) {
        throw std::runtime_error("N dimension (" + std::to_string(N) + ") is not aligned to " + std::to_string(AlignNum));
    }

    using Gemm = cutlass::gemm::device::GemmBatched<
        ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        CutlassArch,
        typename ShapeType<ABType>::ThreadblockShape,
        typename ShapeType<ABType>::WarpShape,
        typename ShapeType<ABType>::InstructionShape,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            NumPerThread,
            ElementAccumulator,
            ElementCompute
        >,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        3,
        AlignNum,
        AlignNum
    >;

    Gemm gemm;

    float scale = 1.0f / (scale_x * scale_w);

    typename Gemm::Arguments args{
        {M, N, K},
        {reinterpret_cast<const ElementInputA*>(A), K},
        M * K,
        {reinterpret_cast<const ElementInputB*>(B), K},
        0,
        {reinterpret_cast<const ElementOutput*>(C), N},
        M * N,
        {reinterpret_cast<ElementOutput*>(C), N},
        M * N,
        {scale, beta},
        batch_count
    };

    cutlass::Status status = gemm(args, nullptr, stream);
    cudaError_t err = cudaGetLastError();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string("GEMM launch failed, cutlass status: ") + cutlass::cutlassGetStatusString(status) + ", cuda error: " + cudaGetErrorString(err));
    }
}

template <typename ABType, typename CType>
void launch_array_gemm_scaled(
    const ABType** A, const ABType** B, CType** C,
    int M, int N, int K,
    int batch_count,
    cudaStream_t stream) {

    using ElementInputA = typename CutlassElementOutputType<ABType>::type;
    using ElementInputB = typename CutlassElementOutputType<ABType>::type;
    using ElementOutput = typename CutlassElementOutputType<CType>::type;
    using ElementAccumulator = typename CutlassElementAccumulatorType<ABType>::type;
    using ElementCompute = float;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    using CutlassArch = typename CutlassArchType<ABType>::arch;
    constexpr int NumPerThread = 128 / cutlass::sizeof_bits<ElementOutput>::value;
    constexpr int AlignNum = 16;

    if (K % AlignNum != 0) {
        throw std::runtime_error("K dimension (" + std::to_string(K) + ") is not aligned to " + std::to_string(AlignNum));
    }
    if (N % AlignNum != 0) {
        throw std::runtime_error("N dimension (" + std::to_string(N) + ") is not aligned to " + std::to_string(AlignNum));
    }

    using GemmArray = cutlass::gemm::device::GemmArray<
        ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        CutlassArch,
        typename ShapeType<ABType>::ThreadblockShape,
        typename ShapeType<ABType>::WarpShape,
        typename ShapeType<ABType>::InstructionShape,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            NumPerThread,
            ElementAccumulator,
            ElementCompute
        >,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        3,
        AlignNum,
        AlignNum
    >;

    typename GemmArray::Arguments args{
        {M, N, K},
        reinterpret_cast<ElementInputA const * const *>(A), K,
        reinterpret_cast<ElementInputB const * const *>(B), K,
        reinterpret_cast<ElementOutput const * const *>(C), N,
        reinterpret_cast<ElementOutput * const *>(C), N,
        {1.0f, 0.0f},
        batch_count   
    };

    GemmArray gemm;
    cutlass::Status status = gemm(args, nullptr, stream);
    cudaError_t err = cudaGetLastError();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string("GEMM launch failed, cutlass status: ") + cutlass::cutlassGetStatusString(status) + ", cuda error: " + cudaGetErrorString(err));
    }
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

template<typename T>
void real_quantized_quantize_weights_axis(at::Tensor weights, at::Tensor outputs, int axis, at::Tensor scale_w);

template <>
void real_quantized_quantize_weights_axis<int8_t>(at::Tensor weights, at::Tensor outputs, int axis, at::Tensor scale_w) {
    size_t numel = weights.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto last_features = weights.sizes()[1];

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "real_quantize_scaled_axis_kernel", [&] {
        switch (axis) {
            case 0:
                real_quantize_scaled_axis_kernel<scalar_t, int8_t, int8_t, 0><<<
                    (numel + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4), BLOCK_SIZE, 0, stream>>>(
                    weights.data_ptr<scalar_t>(),
                    scale_w.data_ptr<float>(),
                    last_features,
                    numel,
                    outputs.data_ptr<int8_t>());
                break;
            case 1:
                real_quantize_scaled_axis_kernel<scalar_t, int8_t, int8_t, 1><<<
                    (numel + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4), BLOCK_SIZE, 0, stream>>>(
                    weights.data_ptr<scalar_t>(),
                    scale_w.data_ptr<float>(),
                    last_features,
                    numel,
                    outputs.data_ptr<int8_t>());
                break;
            case -1:
                real_quantize_scaled_axis_kernel<scalar_t, int8_t, int8_t, -1><<<
                    (numel + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4), BLOCK_SIZE, 0, stream>>>(
                    weights.data_ptr<scalar_t>(),
                    scale_w.data_ptr<float>(),
                    last_features,
                    numel,
                    outputs.data_ptr<int8_t>());
                break;
            default:
                AT_ERROR("Unsupported axis: ", axis);
        }
    });
}

template <>
void real_quantized_quantize_weights_axis<__nv_fp8_e4m3>(at::Tensor weights, at::Tensor outputs, int axis, at::Tensor scale_w) {
    size_t numel = weights.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto last_features = weights.sizes()[1];

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "real_quantize_scaled_axis_kernel", [&] {
        switch (axis) {
            case 0:
                real_quantize_scaled_axis_kernel<scalar_t, __nv_fp8_e4m3, uint8_t, 0><<<
                    (numel + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4), BLOCK_SIZE, 0, stream>>>(
                    weights.data_ptr<scalar_t>(),
                    scale_w.data_ptr<float>(),
                    last_features,
                    numel,
                    outputs.data_ptr<int8_t>());
                break;
            case 1:
                real_quantize_scaled_axis_kernel<scalar_t, __nv_fp8_e4m3, uint8_t, 1><<<
                    (numel + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4), BLOCK_SIZE, 0, stream>>>(
                    weights.data_ptr<scalar_t>(),
                    scale_w.data_ptr<float>(),
                    last_features,
                    numel,
                    outputs.data_ptr<int8_t>());
                break;
            case -1:
                real_quantize_scaled_axis_kernel<scalar_t, __nv_fp8_e4m3, uint8_t, -1><<<
                    (numel + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4), BLOCK_SIZE, 0, stream>>>(
                    weights.data_ptr<scalar_t>(),
                    scale_w.data_ptr<float>(),
                    last_features,
                    numel,
                    outputs.data_ptr<int8_t>());
                break;
            default:
                AT_ERROR("Unsupported axis: ", axis);
        }
    });
}

template <>
void real_quantized_quantize_weights_axis<cutlass::int4b_t>(at::Tensor weights, at::Tensor outputs, int axis, at::Tensor scale_w) {
    size_t numel = weights.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto last_features = weights.sizes()[1];

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "real_quantize_scaled_axis_kernel", [&] {
        switch (axis) {
            case 0:
                real_quantize_scaled_axis_kernel<scalar_t, cutlass::int4b_t, uint8_t, 0><<<
                    (numel + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4), BLOCK_SIZE, 0, stream>>>(
                    weights.data_ptr<scalar_t>(),
                    scale_w.data_ptr<float>(),
                    last_features,
                    numel,
                    outputs.data_ptr<int8_t>());
                break;
            case 1:
                real_quantize_scaled_axis_kernel<scalar_t, cutlass::int4b_t, uint8_t, uint8_t, 1><<<
                    (numel + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4), BLOCK_SIZE, 0, stream>>>(
                    weights.data_ptr<scalar_t>(),
                    scale_w.data_ptr<float>(),
                    last_features,
                    numel,
                    outputs.data_ptr<int8_t>());
                break;
            case -1:
                real_quantize_scaled_axis_kernel<scalar_t, cutlass::int4b_t, uint8_t, uint8_t, -1><<<
                    (numel + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4), BLOCK_SIZE, 0, stream>>>(
                    weights.data_ptr<scalar_t>(),
                    scale_w.data_ptr<float>(),
                    last_features,
                    numel,
                    outputs.data_ptr<int8_t>());
                break;
            default:
                AT_ERROR("Unsupported axis: ", axis);
        }
    });
}

//////////////////////////////////////////////////////////////////////
////////// Call GEMM Pipepine
//////////////////////////////////////////////////////////////////////
template<typename T>
at::Tensor real_quantized_gemm_scaled(at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w) {
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
    int64_t N = std::is_same<T, cutlass::int4b_t>::value ? weights_sizes[0] * 2 / K : weights_sizes[0]; // (N, K)
    if constexpr (!std::is_same<T, cutlass::int4b_t>::value) {
        if (weights_sizes[1] != K) {
            std::ostringstream oss;
            oss << "real_quantized_e4m3fy_gemm_scaled: weights tensor must have shape [N, K], but got weights shape ["
                << weights_sizes[0] << ", " << weights_sizes[1] << "] and inputs shape [..., "
                << inputs_sizes[input_rank - 2] << ", " << inputs_sizes[input_rank - 1] << "]";
            throw std::runtime_error(oss.str());
        }
    }

    // create output tensor
    auto options = inputs.options();
    std::vector<int64_t> output_sizes(inputs_sizes.begin(), inputs_sizes.end() - 1);
    output_sizes.push_back(N);
    auto outputs = at::empty(output_sizes, options);
    // quantized tensors
    using StoreT = typename StoreType<T>::type;
    auto Xq_tensor = std::is_same<T, cutlass::int4b_t>::value
        ? at::empty({(inputs.numel() + 1) / 2}, options.dtype(c10::CppTypeToScalarType<StoreT>::value))
        : at::empty_like(inputs, options.dtype(c10::CppTypeToScalarType<StoreT>::value));
    T* Xq = reinterpret_cast<T*>(Xq_tensor.template data_ptr<StoreT>());
    T* Wq = reinterpret_cast<T*>(weights.template data_ptr<StoreT>());

    auto stream = c10::cuda::getCurrentCUDAStream();
    size_t numel_x = inputs.numel();
    AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "real_quantize_scaled_kernel", [&] {
        real_quantize_scaled_kernel<scalar_t, T, StoreT><<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            inputs.data_ptr<scalar_t>(), scale_x, numel_x, Xq_tensor.template data_ptr<StoreT>()
        );
    });

    // scale can be fusioned
    if (batch_count > 1) {
        // batched gemm
        AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "launch_fp8_batched_gemm_scaled", [&] {
            launch_batched_gemm_scaled<T, scalar_t>(Xq, Wq, outputs.data_ptr<scalar_t>(), M, N, K, batch_count, scale_x, scale_w, 0.0f, stream);
        });
    }
    else {
        // single gemm
        AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "launch_fp8_gemm_scaled", [&] {
            launch_gemm_scaled<T, scalar_t>(Xq, Wq, outputs.data_ptr<scalar_t>(), M, N, K, scale_x, scale_w, 0.0f, stream);
        });
    }
    
    return outputs;
}

template<typename T>
at::Tensor real_quantized_gemm_dual_scaled(at::Tensor inputs, at::Tensor weights,
                                float pos_scale_x, float neg_scale_x,
                                float scale_w) {
    
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
    int64_t N = std::is_same<T, cutlass::int4b_t>::value ? weights_sizes[0] * 2 / K : weights_sizes[0]; // (N, K)
    if constexpr (!std::is_same<T, cutlass::int4b_t>::value) {
        if (weights_sizes[1] != K) {
            std::ostringstream oss;
            oss << "real_quantized_e4m3fy_gemm_dual_scaled: weights tensor must have shape [N, K], but got weights shape ["
                << weights_sizes[0] << ", " << weights_sizes[1] << "] and inputs shape [..., "
                << inputs_sizes[input_rank - 2] << ", " << inputs_sizes[input_rank - 1] << "]";
            throw std::runtime_error(oss.str());
        }
    }

    // create output tensor
    auto options = inputs.options();
    std::vector<int64_t> output_sizes(inputs_sizes.begin(), inputs_sizes.end() - 1);
    output_sizes.push_back(N);
    auto outputs = at::empty(output_sizes, options);
    using StoreT = typename StoreType<T>::type;
    auto Xp_tensor = std::is_same<T, cutlass::int4b_t>::value
        ? at::empty({(inputs.numel() + 1) / 2}, options.dtype(c10::CppTypeToScalarType<StoreT>::value))
        : at::empty_like(inputs, options.dtype(c10::CppTypeToScalarType<StoreT>::value));
    
    auto Xn_tensor = std::is_same<T, cutlass::int4b_t>::value
        ? at::empty({(inputs.numel() + 1) / 2}, options.dtype(c10::CppTypeToScalarType<StoreT>::value))
        : at::empty_like(inputs, options.dtype(c10::CppTypeToScalarType<StoreT>::value));

    T* Xp = reinterpret_cast<T*>(Xp_tensor.template data_ptr<typename StoreType<T>::type>());
    T* Xn = reinterpret_cast<T*>(Xn_tensor.template data_ptr<typename StoreType<T>::type>());
    T* Wq = reinterpret_cast<T*>(weights.template data_ptr<typename StoreType<T>::type>());

    auto stream = c10::cuda::getCurrentCUDAStream();
    size_t numel_x = inputs.numel();
    // X = Xp + Xn
    AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "real_quantize_dual_scaled_kernel", [&] {
        real_quantize_dual_scaled_kernel<scalar_t, T, StoreT><<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            inputs.data_ptr<scalar_t>(), pos_scale_x, neg_scale_x, numel_x, Xp_tensor.template data_ptr<typename StoreType<T>::type>(), Xn_tensor.template data_ptr<typename StoreType<T>::type>());
    });

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
        launch_array_gemm_scaled<T, scalar_t>(
            reinterpret_cast<const T**>(ptr_A_batched.get()),
            reinterpret_cast<const T**>(ptr_B_batched.get()),
            reinterpret_cast<scalar_t**>(ptr_C_batched.get()),
            M, N, K,
            2 * batch_count,
            stream);
    });

    // Y = Y_p + Y_n
    size_t numel_y = outputs.numel();
    AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "real_dequantize_dual_scaled_kernel", [&] {
        real_dequantize_dual_scaled_kernel<scalar_t><<<numel_y / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            Y_p.data_ptr<float>(), 
            Y_n.data_ptr<float>(), 
            outputs.data_ptr<scalar_t>(), 
            pos_scale_x * scale_w, 
            neg_scale_x * scale_w, 
            numel_y
        );
    });

    return outputs;
}

#ifdef SEGQUANT_FP8
template at::Tensor real_quantized_gemm_scaled<__nv_fp8_e4m3>(at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w);
template at::Tensor real_quantized_gemm_dual_scaled<__nv_fp8_e4m3>(at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, float scale_w);
#endif

#ifdef SEGQUANT_INT8
template at::Tensor real_quantized_gemm_scaled<int8_t>(at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w);
template at::Tensor real_quantized_gemm_dual_scaled<int8_t>(at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, float scale_w);
#endif

#ifdef SEGQUANT_INT4
template at::Tensor real_quantized_gemm_scaled<cutlass::int4b_t>(at::Tensor inputs, at::Tensor weights, float scale_x, float scale_w);
template at::Tensor real_quantized_gemm_dual_scaled<cutlass::int4b_t>(at::Tensor inputs, at::Tensor weights, float pos_scale_x, float neg_scale_x, float scale_w);
#endif