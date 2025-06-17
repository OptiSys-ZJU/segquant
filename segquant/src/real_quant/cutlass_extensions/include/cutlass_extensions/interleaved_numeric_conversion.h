/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*!
    \file
    \brief Boost-like numeric conversion operator for int8 and CUTLASS int4b_t interleaved in a register
*/

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/numeric_types.h"

namespace cutlass
{

// This converter is meant to be used with data interleaved in a 32-bit register where the even elements are in the low
// bits and the odd elemeents are in the high bits of the register. In addition, it assumes elements were originally
// signed and had a bias of 2**(b-1) added (where b is the number of bits in the type) to make all numbers unsigned.
// This converter will uninterleave the data and subtract the bias while converting to the result type.
template <typename T, typename S, int N>
struct FastInterleavedAndBiasedNumericArrayConverter
{
};

template <>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, int4b_t, 8>
{
    using result_type = Array<half_t, 8>;
    using source_type = Array<int4b_t, 8>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        result_type result;

        // todo PTX
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int value = static_cast<int>(source[i]); // sign-extended int4 → int
            float fval = static_cast<float>(value);  // int → float
            result[i] = static_cast<half_t>(fval);   // float → fp16
        }

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

template <int N>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, int4b_t, N>
{
    static constexpr int VEC_WIDTH = 8;
    static_assert(!(N % VEC_WIDTH), "N must be multiple of 8.");

    using result_type = Array<half_t, N>;
    using source_type = Array<int4b_t, N>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        using scalar_result_type = typename result_type::Element;
        using scalar_source_type = typename source_type::Element;
        FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
            convert_vector_;

        result_type result;
        using vec_result = Array<scalar_result_type, VEC_WIDTH>;
        using vec_source = Array<scalar_source_type, VEC_WIDTH>;

        vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
        vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / VEC_WIDTH; ++i)
        {
            result_ptr[i] = convert_vector_(source_ptr[i]);
        }

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
