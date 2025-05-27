#include <cutlass/gemm/device/gemm.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "quantizer_fp8.cuh"

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
    using NumPerThread = 4;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

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
    size_t numel_w = inputs.numel();
    size_t numel_y = outputs.numel();

    auto stream = c10::cuda::getCurrentCUDAStream();

    cudaMallocAsync(&Xq, numel_x * sizeof(__nv_fp8_e4m3), stream);

    cudaMallocAsync(&Wq, numel_w * sizeof(__nv_fp8_e4m3), stream);

    AT_DISPATCH_FLOATING_TYPES(inputs.type().scalarType(), "real_quantize_e4m3fy_split_cuda", [&] {
        real_quantize_e4m3fy_scaled_kernel<<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            inputs.data_ptr<scalar_t>(), scale_x, numel_x, Xq
        );
    });

    AT_DISPATCH_FLOATING_TYPES(weights.type().scalarType(), "real_quantize_e4m3fy_split_cuda", [&] {
        real_quantize_e4m3fy_scaled_kernel<<<numel_w / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            weights.data_ptr<scalar_t>(), scale_w, numel_w, Wq
        );
    });

    // scale can be fusioned
    launch_fp8_gemm_scaled(Xq, Wq, outputs.data_ptr<scalar_t>(), 
                            M, N, K, scale_x, scale_w, 0.0f, stream);

    cudaFreeAsync(Xq, stream);
    cudaFreeAsync(Wq, stream);

    return outputs;
}

at::Tensor real_quantized_e4m3fy_gemm_scaled_bias(at::Tensor inputs, at::Tensor weights, at::Tensor bias,
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
    size_t numel_w = inputs.numel();
    size_t numel_y = outputs.numel();

    auto stream = c10::cuda::getCurrentCUDAStream();

    cudaMallocAsync(&Xq, numel_x * sizeof(__nv_fp8_e4m3), stream);

    cudaMallocAsync(&Wq, numel_w * sizeof(__nv_fp8_e4m3), stream);

    AT_DISPATCH_FLOATING_TYPES(inputs.type().scalarType(), "real_quantize_e4m3fy_split_cuda", [&] {
        real_quantize_e4m3fy_scaled_kernel<<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            inputs.data_ptr<scalar_t>(), scale_x, numel_x, Xq
        );
    });

    AT_DISPATCH_FLOATING_TYPES(weights.type().scalarType(), "real_quantize_e4m3fy_split_cuda", [&] {
        real_quantize_e4m3fy_scaled_kernel<<<numel_w / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            weights.data_ptr<scalar_t>(), scale_w, numel_w, Wq
        );
    });

    // scale can be fusioned
    launch_fp8_gemm_scaled(Xq, Wq, outputs.data_ptr<scalar_t>(), 
                            M, N, K, scale_x, scale_w, 1.0f, stream);

    cudaFreeAsync(Xq, stream);
    cudaFreeAsync(Wq, stream);

    return outputs;
}

at::Tensor real_quantized_e4m3fy_gemm_dual_scaled(at::Tensor inputs, at::Tensor weights,
                                float pos_scale_x, float neg_scale_x,
                                float pos_scale_w, float neg_scale_w) {
    
    __nv_fp8_e4m3 *Xp, *Xn, *Wp, *Wn;
    
    auto inputs_sizes = inputs.sizes();
    auto weights_sizes = weights.sizes();
    int64_t M = inputs_sizes[0];
    int64_t K = inputs_sizes[1];
    int64_t N = weights_sizes[0];

    auto options = inputs.options();
    auto outputs = torch::empty({M, N}, options);

    size_t numel_x = inputs.numel();
    size_t numel_w = inputs.numel();
    size_t numel_y = outputs.numel();

    auto stream = c10::cuda::getCurrentCUDAStream();

    cudaMallocAsync(&Xp, numel_x * sizeof(__nv_fp8_e4m3), stream);
    cudaMallocAsync(&Xn, numel_x * sizeof(__nv_fp8_e4m3), stream);

    cudaMallocAsync(&Wp, numel_w * sizeof(__nv_fp8_e4m3), stream);
    cudaMallocAsync(&Wn, numel_w * sizeof(__nv_fp8_e4m3), stream);
    
    // X = Xp - Xn, W = Wp - Wn
    AT_DISPATCH_FLOATING_TYPES(inputs.type().scalarType(), "real_quantize_e4m3fy_split_cuda", [&] {
        real_quantize_e4m3fy_dual_scaled_split_kernel<<<numel_x / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            inputs.data_ptr<scalar_t>(), pos_scale_x, neg_scale_x, numel_x, Xp, Xn);
    });

    AT_DISPATCH_FLOATING_TYPES(weights.type().scalarType(), "real_quantize_e4m3fy_split_cuda", [&] {
        real_quantize_e4m3fy_dual_scaled_split_kernel<<<numel_w / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            weights.data_ptr<scalar_t>(), pos_scale_w, neg_scale_w, numel_w, Wp, Wn);
    });

    auto Y_pp = at::empty_like(outputs);
    auto Y_nn = at::empty_like(outputs);
    auto Y_pn = at::empty_like(outputs);
    auto Y_np = at::empty_like(outputs);
    
    // todo
    launch_fp8_gemm(Xp, Wp, Y_pp, M, N, K);
    launch_fp8_gemm(Xn, Wn, Y_nn, M, N, K);
    launch_fp8_gemm(Xp, Wn, Y_pn, M, N, K);
    launch_fp8_gemm(Xn, Wp, Y_np, M, N, K);

    // Y = Y_pp - Y_pn - Y_np + Y_nn
    AT_DISPATCH_FLOATING_TYPES(outputs.type().scalarType(), "real_dequantize_e4m3fy_combine_cuda", [&] {
        real_dequantize_e4m3fy_dual_scaled_combine_kernel<<<numel_y / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
            Y_pp.data_ptr<scalar_t>(), 
            Y_nn.data_ptr<scalar_t>(), 
            Y_pn.data_ptr<scalar_t>(), 
            Y_np.data_ptr<scalar_t>(), 
            outputs.data_ptr<scalar_t>(), 
            pos_scale_x * pos_scale_w, 
            pos_scale_x * neg_scale_w, 
            neg_scale_x * pos_scale_w, 
            neg_scale_x * neg_scale_x, 
            numel_y
        );
    });

    cudaFreeAsync(Xp, stream);
    cudaFreeAsync(Xn, stream);
    cudaFreeAsync(Wp, stream);
    cudaFreeAsync(Wn, stream);

    return outputs;
}
