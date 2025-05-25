import torch
from typing import List, Literal


class BaseSplitter:
    def __init__(self, split_sizes, seg_mode: Literal["input", "weight"] = "weight"):
        self.split_sizes = split_sizes
        self.seg_input = True if seg_mode == "input" else False

    def split_output(self, output: torch.Tensor):
        if not self.seg_input:
            return output.split(self.split_sizes, dim=-int(not self.seg_input))
        else:
            raise ValueError("Spliter: split_output not work")

    def concat_output(self, output_chunks: List[torch.Tensor]):
        if not self.seg_input:
            return torch.cat(output_chunks, dim=-int(not self.seg_input))
        else:
            raise ValueError("Spliter: concat_output not work")

    def split_weight(self, weight: torch.Tensor):
        if not self.seg_input:
            return weight.split(self.split_sizes, dim=int(self.seg_input))
        else:
            raise ValueError("Spliter: split_weight not work")

    def concat_weight(self, weight_chunks: List[torch.Tensor]):
        if not self.seg_input:
            return torch.cat(weight_chunks, dim=int(self.seg_input))
        else:
            raise ValueError("Spliter: concat_weight not work")

    def split_input(self, input: torch.Tensor):
        if self.seg_input:
            return input.split(self.split_sizes, dim=-1)
        else:
            raise ValueError("Spliter: split_input not work")

    def concat_input(self, input_chunks: List[torch.Tensor]):
        if self.seg_input:
            return torch.split(input_chunks, dim=-1)
        else:
            raise ValueError("Spliter: concat_input not work")
