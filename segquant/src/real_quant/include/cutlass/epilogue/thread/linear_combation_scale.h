#pragma once

namespace cutlass {
namespace epilogue {
namespace thread {

template <
    typename ElementOutput_,      // Data type of output tensor.
    int Count,                    // Number of elements computed per operation.
    typename ElementAccumulator_, // Data type of accumulator
    typename ElementCompute_,     // Data type for internal computations
>
class LinearCombinationScaleVector {
public:
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    using ElementScalingFactor = ElementAccumulator_;

    static int const kCount = Count;
    using FragmentOutput = Array<ElementOutput, kCount>;
    using FragmentAccumulator = Array<ElementAccumulator, kCount>;
    using FragmentCompute = Array<ElementCompute, kCount>;

    struct Params {
        ElementScalingFactor const* scale_a_ptr;
        ElementScalingFactor const* scale_b_ptr;
        
        CUTLASS_HOST_DEVICE
        Params() : scale_a_ptr(nullptr), scale_b_ptr(nullptr) {}
        CUTLASS_HOST_DEVICE
        Params(ElementScalingFactor const* scale_a_ptr_, ElementScalingFactor const* scale_b_ptr_)
            : scale_a_ptr(scale_a_ptr_), scale_b_ptr(scale_b_ptr_) {}
    };
private:
    Params params_;
public:
    CUTLASS_HOST_DEVICE
    LinearCombinationScale(Params const& params): params_(params) {

    }

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
        // Convert source to interal compute numeric type
        NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;
        
        // Convert to destination numeric type
        NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

        FragmentCompute converted_accumulator = accumulator_converter(accumulator);
        FragmentCompute intermediate;
        multiplies<FragmentCompute> mul_accumulator;
        intermediate = mul_accumulator(alpha_, converted_accumulator);
        return destination_converter(intermediate);
    }
};

}



} // namespace epilogue
} // namespace cutlass