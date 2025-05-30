"""
This module provides the `BaseSplitter` class, a utility for splitting and concatenating
tensors based on specified dimensions and segmentation modes. It supports two modes:
"input" and "weight", which determine the behavior of the tensor operations.

Classes:
    BaseSplitter: A class for handling tensor splitting and concatenation operations
                  based on the specified segmentation mode and split sizes.

Dependencies:
    - typing.List
    - typing.Literal
    - torch
"""

from typing import List, Literal
import torch


class BaseSplitter:
    """
    BaseSplitter is a utility class designed to handle the splitting and concatenation of tensors
    based on specified dimensions. It supports two segmentation modes: "input" and "weight".
    Attributes:
        split_sizes (List[int]): A list of sizes to split the tensor into.
        seg_input (bool): A boolean indicating whether the segmentation mode is "input"
                          (True) or "weight" (False).
    """

    def __init__(self, split_sizes, seg_mode: Literal["input", "weight"] = "weight"):
        self.split_sizes = split_sizes
        self.seg_input = seg_mode == "input"

    def split_output(self, output: torch.Tensor):
        """
        Splits the output tensor into chunks based on `split_sizes` along the last dimension
        if the segmentation mode is "weight". Raises a ValueError if the mode is "input".
        """
        if not self.seg_input:
            return output.split(self.split_sizes, dim=-int(not self.seg_input))
        raise ValueError("Spliter: split_output not work")

    def concat_output(self, output_chunks: List[torch.Tensor]):
        """
        Concatenates a list of output tensor chunks into a single tensor along the last dimension
        if the segmentation mode is "weight". Raises a ValueError if the mode is "input".
        """
        if not self.seg_input:
            return torch.cat(output_chunks, dim=-int(not self.seg_input))
        raise ValueError("Spliter: concat_output not work")

    def split_weight(self, weight: torch.Tensor):
        """
        Splits the weight tensor into chunks based on `split_sizes` along the specified dimension
        if the segmentation mode is "weight". Raises a ValueError if the mode is "input".
        """
        if not self.seg_input:
            return weight.split(self.split_sizes, dim=int(self.seg_input))
        raise ValueError("Spliter: split_weight not work")

    def concat_weight(self, weight_chunks: List[torch.Tensor]):
        """
        Concatenates weight tensor chunks into a single tensor along the specified dimension
        if the segmentation mode is "weight". Raises a ValueError if the mode is "input".
        """
        if not self.seg_input:
            return torch.cat(weight_chunks, dim=int(self.seg_input))
        raise ValueError("Spliter: concat_weight not work")

    def split_input(self, input_t: torch.Tensor):
        """
        Splits the input tensor into chunks based on `split_sizes` along the last dimension
        if the segmentation mode is "input". Raises a ValueError if the mode is "weight".
        """
        if self.seg_input:
            return input_t.split(self.split_sizes, dim=-1)
        raise ValueError("Spliter: split_input not work")

    def concat_input(self, input_chunks: List[torch.Tensor]):
        """
        Concatenates a list of input tensor chunks into a single tensor along the last dimension
        if the segmentation mode is "input". Raises a ValueError if the mode is "weight".
        """
        if self.seg_input:
            return torch.concat(input_chunks, dim=-1)
        raise ValueError("Spliter: concat_input not work")
