/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "cutlass_extensions/gemm/threadblock/default_mix_mma_multistage.h"

namespace cutlass
{
namespace gemm
{
namespace threadblock
{

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp), fp16 activation & int4 weight
template <
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator>
struct DefaultMma<cutlass::half_t, LayoutA, kAlignmentA, int4b_t, LayoutB, kAlignmentB, ElementAccumulator,
    layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2, Operator>
{

private:
    using Mma = MixMma<half_t, LayoutA, kAlignmentA, int4b_t, LayoutB, kAlignmentB,
        ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape,
        WarpShape, InstructionShape, 2, Operator>;

public:
    // Define the MmaCore components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major output (OperatorClass TensorOp), fp16 activation & int4 weight
template <
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    ///
    int kStages,
    /// Shared memory clear option
    SharedMemoryClearOption SharedMemoryClear>
struct DefaultMma<cutlass::half_t, LayoutA, kAlignmentA, int4b_t, LayoutB, kAlignmentB, ElementAccumulator,
    layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape, InstructionShape, kStages, Operator,
    false, SharedMemoryClear>
{

private:
    using Mma = MixMma<half_t, LayoutA, kAlignmentA, int4b_t, LayoutB, kAlignmentB,
        ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape,
        WarpShape, InstructionShape, kStages, Operator, SharedMemoryClear>;

public:
    // Define the MmaCore components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
