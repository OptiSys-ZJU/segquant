#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_array.h>
#include <cutlass/util/device_memory.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <string>
#include "quantizer_fp8.cuh"
#include "../dequantized.cuh"

#define BLOCK_SIZE 128

#define AT_DISPATCH_CASE_FLOATING_TYPES(...)                                                       \
    AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)                                            \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                                             \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                                              \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                                                \
    AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

template <typename T>
struct CutlassElementOutputType;

template <>
struct CutlassElementOutputType<at::Half> {
    using type = cutlass::half_t;
};

template <>
struct CutlassElementOutputType<float> {
    using type = float;
};

template <>
struct CutlassElementOutputType<at::BFloat16> {
    using type = cutlass::bfloat16_t;
};

template <>
struct CutlassElementOutputType<double> {
    using type = double;
};

template <typename T>
void launch_fp8_gemm_scaled(
    const __nv_fp8_e4m3 *A, const __nv_fp8_e4m3 *B, T *C,
    int M, int N, int K,
    float scale_x, float scale_w,
    float beta,
    cudaStream_t stream) {

    using ElementInputA = cutlass::float_e4m3_t;
    using ElementInputB = cutlass::float_e4m3_t;
    using ElementOutput = typename CutlassElementOutputType<T>::type;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    constexpr int NumPerThread = 4;
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
        cutlass::arch::Sm89,
        cutlass::gemm::GemmShape<128, 64, 128>,
        cutlass::gemm::GemmShape<64, 32, 128>,
        cutlass::gemm::GemmShape<16, 8, 32>,
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

template <typename T>
void launch_fp8_array_gemm_scaled(
    const __nv_fp8_e4m3** A, const __nv_fp8_e4m3** B, T** C,
    int M, int N, int K,
    int batch_count,
    cudaStream_t stream) {

    using ElementInputA = cutlass::float_e4m3_t;
    using ElementInputB = cutlass::float_e4m3_t;
    using ElementOutput = typename CutlassElementOutputType<T>::type;
    using ElementAccumulator = float;
    using ElementCompute = float;
    constexpr int NumPerThread = 4;

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using GemmArray = cutlass::gemm::device::GemmArray<
        ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm89,
        cutlass::gemm::GemmShape<128, 64, 128>,
        cutlass::gemm::GemmShape<64, 32, 128>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            NumPerThread,
            ElementAccumulator,
            ElementCompute
        >
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
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Array GEMM launch failed");
    }
}

at::Tensor real_quantized_e4m3fy_quantize_weights(at::Tensor weights, float scale_w) {
    auto options = weights.options().dtype(at::kByte);
    auto quantized_weights = at::empty(weights.sizes(), options);

    size_t numel = weights.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "real_quantize_e4m3fy_scaled_kernel", [&] {
        real_quantize_e4m3fy_scaled_kernel<<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            weights.data_ptr<scalar_t>(), scale_w, numel, 
            quantized_weights.data_ptr<uint8_t>()
        );
    });

    return quantized_weights;
}

at::Tensor real_quantized_e4m3fy_gemm_scaled(at::Tensor inputs, at::Tensor weights, 
    float scale_x, float scale_w) {

    auto inputs_sizes = inputs.sizes();
    auto weights_sizes = weights.sizes();
    int64_t M = inputs_sizes[0]; // (M, K)
    int64_t K = inputs_sizes[1];
    int64_t N = weights_sizes[0]; // (N, K)
    if (weights_sizes[1] != K) {
        throw std::runtime_error("weights tensor must have shape [N, K]");
    }

    auto options = inputs.options();
    auto outputs = at::empty({M, N}, options);
    auto Xq_tensor = at::empty({M, K}, options.dtype(at::kByte));
    __nv_fp8_e4m3* Xq = reinterpret_cast<__nv_fp8_e4m3*>(Xq_tensor.data_ptr<uint8_t>());
    __nv_fp8_e4m3* Wq = reinterpret_cast<__nv_fp8_e4m3*>(weights.data_ptr<uint8_t>());

    auto stream = c10::cuda::getCurrentCUDAStream();
    size_t numel_x = inputs.numel();
    AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "real_quantize_e4m3fy_scaled_kernel", [&] {
        real_quantize_e4m3fy_scaled_kernel<<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            inputs.data_ptr<scalar_t>(), scale_x, numel_x, Xq
        );
    });

    // scale can be fusioned
    AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "launch_fp8_gemm_scaled", [&] {
        launch_fp8_gemm_scaled<scalar_t>(Xq, Wq, outputs.data_ptr<scalar_t>(), M, N, K, scale_x, scale_w, 0.0f, stream);
    });

    return outputs;
}

at::Tensor real_quantized_e4m3fy_gemm_dual_scaled(at::Tensor inputs, at::Tensor weights,
                                float pos_scale_x, float neg_scale_x,
                                float scale_w) {
    
    __nv_fp8_e4m3 *Xp, *Xn, *Wq;
    
    auto inputs_sizes = inputs.sizes();
    auto weights_sizes = weights.sizes();
    int64_t M = inputs_sizes[0];
    int64_t K = inputs_sizes[1];
    int64_t N = weights_sizes[0];

    auto options = inputs.options();
    auto outputs = at::empty({M, N}, options);

    size_t numel_x = inputs.numel();
    size_t numel_y = outputs.numel();

    auto stream = c10::cuda::getCurrentCUDAStream();

    cudaMallocAsync(&Xp, numel_x * sizeof(__nv_fp8_e4m3), stream);
    cudaMallocAsync(&Xn, numel_x * sizeof(__nv_fp8_e4m3), stream);
    
    // X = Xp + Xn
    AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "real_quantize_e4m3fy_dual_scaled_kernel", [&] {
        real_quantize_e4m3fy_dual_scaled_kernel<<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            inputs.data_ptr<scalar_t>(), pos_scale_x, neg_scale_x, numel_x, Xp, Xn);
    });

    Wq = reinterpret_cast<__nv_fp8_e4m3*>(weights.data_ptr<uint8_t>());

    auto Y_p = at::empty_like(outputs);
    auto Y_n = at::empty_like(outputs);
    
    std::vector<void const *> ptr_A_batched_host{Xp, Xn};
    std::vector<void const *> ptr_B_batched_host{Wq, Wq};
    std::vector<void*> ptr_C_batched_host;
    AT_DISPATCH_FLOATING_TYPES(Y_p.scalar_type(), "prepare_ptr_C_batched_host", [&] {
        ptr_C_batched_host = {
            static_cast<void*>(Y_p.data_ptr<scalar_t>()),
            static_cast<void*>(Y_n.data_ptr<scalar_t>())
        };
    });

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

    AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "launch_fp8_array_gemm_scaled", [&] {
        launch_fp8_array_gemm_scaled<scalar_t>(
            reinterpret_cast<const __nv_fp8_e4m3**>(ptr_A_batched.get()),
            reinterpret_cast<const __nv_fp8_e4m3**>(ptr_B_batched.get()),
            reinterpret_cast<scalar_t**>(ptr_C_batched.get()),
            M, N, K,
            2,
            stream);
    });

    // Y = Y_p + Y_n
    AT_DISPATCH_FLOATING_TYPES(outputs.scalar_type(), "real_dequantize_dual_scaled_kernel", [&] {
        real_dequantize_dual_scaled_kernel<<<numel_y / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            Y_p.data_ptr<scalar_t>(), 
            Y_n.data_ptr<scalar_t>(), 
            outputs.data_ptr<scalar_t>(), 
            pos_scale_x * scale_w, 
            neg_scale_x * scale_w, 
            numel_y
        );
    });

    cudaFreeAsync(Xp, stream);
    cudaFreeAsync(Xn, stream);

    return outputs;
}
