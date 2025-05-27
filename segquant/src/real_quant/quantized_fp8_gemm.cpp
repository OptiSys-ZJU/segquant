#include <cutlass/gemm/device/gemm.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "quantizer_fp8.cuh"
#include "dequantized.cuh"

#define BLOCK_SIZE 128

#define AT_DISPATCH_CASE_FLOATING_TYPES(...)                                                       \
    AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)                                            \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                                             \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                                              \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                                                \
    AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

template <typename T>
void launch_fp8_gemm_scaled(
    __nv_fp8_e4m3 *A, __nv_fp8_e4m3 *B, T *C,
    int M, int N, int K,
    float scale_x, float scale_w,
    float beta,
    cudaStream_t stream) {

    using ElementInputA = cutlass::float_e4m3_t;
    using ElementInputB = cutlass::float_e4m3_t;
    using ElementOutput = T;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    constexpr int NumPerThread = 4;

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
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            NumPerThread,
            ElementAccumulator,
            ElementCompute
        >
    >;

    Gemm gemm;

    float scale = 1.0f / (scale_x * scale_w);

    typename Gemm::Arguments args{
        {M, N, K},
        {reinterpret_cast<ElementInputA*>(A), K},
        {reinterpret_cast<ElementInputB*>(B), K},
        {C, N},
        {C, N},
        {scale, beta}
    };

    cutlass::Status status = gemm(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("GEMM launch failed");
    }
}

template <typename T>
void launch_fp8_batched_gemm_scaled(
    __nv_fp8_e4m3* A[], __nv_fp8_e4m3* B[], T* C[],
    int M, int N, int K,
    int batch_count,
    cudaStream_t stream) {

    using ElementInputA = cutlass::float_e4m3_t;
    using ElementInputB = cutlass::float_e4m3_t;
    using ElementOutput = T;
    using ElementAccumulator = float;
    using ElementCompute = float;
    constexpr int NumPerThread = 4;

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using GemmBatched = cutlass::gemm::device::GemmBatched<
        ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm89,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            NumPerThread,
            ElementAccumulator,
            ElementCompute
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2  // Stages
    >;

    typename GemmBatched::Arguments args{
        {M, N, K},
        A,
        B,
        C,
        C,
        batch_count,
        K,  // lda
        K,  // ldb
        N,  // ldc
        N,  // ldd
        {1.0f, 0.0f}
    };

    GemmBatched gemm;
    cutlass::Status status = gemm(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Batched GEMM launch failed");
    }
}

at::Tensor real_quantized_e4m3fy_quantize_weights(at::Tensor weights, float scale_w) {
    auto options = weights.options().dtype(at::kByte);
    auto quantized_weights = torch::empty(weights.sizes(), options);

    size_t numel = weights.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(weights.type().scalarType(), "real_quantize_e4m3fy_weight_kernel", [&] {
        real_quantize_e4m3fy_weight_kernel<<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            weights.data_ptr<scalar_t>(), scale_w, numel, 
            quantized_weights.data_ptr<uint8_t>()
        );
    });

    return quantized_weights;
}

at::Tensor real_quantized_e4m3fy_gemm_scaled(at::Tensor inputs, at::Tensor weights, 
    float scale_x, float scale_w) {

    __nv_fp8_e4m3 *Xq, *Wq;

    auto inputs_sizes = inputs.sizes();
    auto weights_sizes = weights.sizes();
    int64_t M = inputs_sizes[0];
    int64_t K = inputs_sizes[1];
    int64_t N = weights_sizes[0];

    auto options = inputs.options();
    auto outputs = torch::empty({M, N}, options);

    size_t numel_x = inputs.numel();
    size_t numel_y = outputs.numel();

    auto stream = c10::cuda::getCurrentCUDAStream();

    cudaMallocAsync(&Xq, numel_x * sizeof(__nv_fp8_e4m3), stream);

    AT_DISPATCH_FLOATING_TYPES(inputs.type().scalarType(), "real_quantize_e4m3fy_scaled_kernel", [&] {
        real_quantize_e4m3fy_scaled_kernel<<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            inputs.data_ptr<scalar_t>(), scale_x, numel_x, Xq
        );
    });

    Wq = reinterpret_cast<__nv_fp8_e4m3*>(weights.data_ptr<uint8_t>());

    // scale can be fusioned
    launch_fp8_gemm_scaled(Xq, Wq, outputs.data_ptr<scalar_t>(), 
                            M, N, K, scale_x, scale_w, 0.0f, stream);

    cudaFreeAsync(Xq, stream);

    return outputs;
}

at::Tensor real_quantized_e4m3fy_gemm_scaled(at::Tensor inputs, at::Tensor weights, at::Tensor bias,
    float scale_x, float scale_w) {

    __nv_fp8_e4m3 *Xq, *Wq;

    auto inputs_sizes = inputs.sizes();
    auto weights_sizes = weights.sizes();
    int64_t M = inputs_sizes[0];
    int64_t K = inputs_sizes[1];
    int64_t N = weights_sizes[0];

    auto options = inputs.options();
    auto outputs = bias.expand({M, N}).contiguous().clone();

    size_t numel_x = inputs.numel();
    size_t numel_y = outputs.numel();

    auto stream = c10::cuda::getCurrentCUDAStream();

    cudaMallocAsync(&Xq, numel_x * sizeof(__nv_fp8_e4m3), stream);

    AT_DISPATCH_FLOATING_TYPES(inputs.type().scalarType(), "real_quantize_e4m3fy_scaled_kernel", [&] {
        real_quantize_e4m3fy_scaled_kernel<<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            inputs.data_ptr<scalar_t>(), scale_x, numel_x, Xq
        );
    });

    Wq = reinterpret_cast<__nv_fp8_e4m3*>(weights.data_ptr<uint8_t>());

    // scale can be fusioned
    launch_fp8_gemm_scaled(Xq, Wq, outputs.data_ptr<scalar_t>(), 
                            M, N, K, scale_x, scale_w, 1.0f, stream);

    cudaFreeAsync(Xq, stream);

    return outputs;
}

at::Tensor real_quantized_e4m3fy_gemm_dual_scaled(at::Tensor inputs, at::Tensor weights,
                                float pos_scale_x, float neg_scale_x,
                                float scale_w) {
    
    __nv_fp8_e4m3 *Xp, *Xn, *Wp, *Wn;
    
    auto inputs_sizes = inputs.sizes();
    auto weights_sizes = weights.sizes();
    int64_t M = inputs_sizes[0];
    int64_t K = inputs_sizes[1];
    int64_t N = weights_sizes[0];

    auto options = inputs.options();
    auto outputs = torch::empty({M, N}, options);

    size_t numel_x = inputs.numel();
    size_t numel_y = outputs.numel();

    auto stream = c10::cuda::getCurrentCUDAStream();

    cudaMallocAsync(&Xp, numel_x * sizeof(__nv_fp8_e4m3), stream);
    cudaMallocAsync(&Xn, numel_x * sizeof(__nv_fp8_e4m3), stream);
    
    // X = Xp + Xn
    AT_DISPATCH_FLOATING_TYPES(inputs.type().scalarType(), "real_quantize_e4m3fy_dual_scaled_split_kernel", [&] {
        real_quantize_e4m3fy_dual_scaled_split_kernel<<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            inputs.data_ptr<scalar_t>(), pos_scale_x, neg_scale_x, numel_x, Xp, Xn);
    });

    Wq = reinterpret_cast<__nv_fp8_e4m3*>(weights.data_ptr<uint8_t>());

    auto Y_p = at::empty_like(outputs);
    auto Y_n = at::empty_like(outputs);
    
    std::vector<void const *> ptr_A_batched_host{Xp, Xn};
    std::vector<void const *> ptr_B_batched_host{Wq, Wq};
    std::vector<void       *> ptr_C_batched_host{Y_p.data_ptr<scalar_t>(), Y_n.data_ptr<scalar_t>()};
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

    launch_fp8_batched_gemm_scaled(
        ptr_A_batched.get(), ptr_B_batched.get(), ptr_C_batched.get(),
        M, N, K,
        2,
        stream);

    // Y = Y_p + Y_n
    AT_DISPATCH_FLOATING_TYPES(outputs.type().scalarType(), "real_dequantize_dual_scaled_kernel", [&] {
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

at::Tensor real_quantized_e4m3fy_gemm_dual_scaled(at::Tensor inputs, at::Tensor weights, at::Tensor bias,
    float pos_scale_x, float neg_scale_x,
    float scale_w) {

    __nv_fp8_e4m3 *Xp, *Xn, *Wp, *Wn;

    auto inputs_sizes = inputs.sizes();
    auto weights_sizes = weights.sizes();
    int64_t M = inputs_sizes[0];
    int64_t K = inputs_sizes[1];
    int64_t N = weights_sizes[0];

    auto options = inputs.options();
    auto outputs = torch::empty({M, N}, options);

    size_t numel_x = inputs.numel();
    size_t numel_y = outputs.numel();

    auto stream = c10::cuda::getCurrentCUDAStream();

    cudaMallocAsync(&Xp, numel_x * sizeof(__nv_fp8_e4m3), stream);
    cudaMallocAsync(&Xn, numel_x * sizeof(__nv_fp8_e4m3), stream);

    // X = Xp - Xn
    AT_DISPATCH_FLOATING_TYPES(inputs.type().scalarType(), "real_quantize_e4m3fy_dual_scaled_split_kernel", [&] {
        real_quantize_e4m3fy_dual_scaled_split_kernel<<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
        inputs.data_ptr<scalar_t>(), pos_scale_x, neg_scale_x, numel_x, Xp, Xn);
    });

    Wq = reinterpret_cast<__nv_fp8_e4m3*>(weights.data_ptr<uint8_t>());

    auto Y_p = at::empty_like(outputs);
    auto Y_n = at::empty_like(outputs);

    std::vector<void const *> ptr_A_batched_host{Xp, Xn};
    std::vector<void const *> ptr_B_batched_host{Wq, Wq};
    std::vector<void       *> ptr_C_batched_host{Y_p.data_ptr<scalar_t>(), Y_n.data_ptr<scalar_t>()};
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

    launch_fp8_batched_gemm_scaled(
        ptr_A_batched.get(), ptr_B_batched.get(), ptr_C_batched.get(),
        M, N, K,
        2,
        stream);

    // Y = Y_p - Y_n
    AT_DISPATCH_FLOATING_TYPES(outputs.type().scalarType(), "real_dequantize_dual_scaled_kernel", [&] {
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

    return outputs + bias;
}
