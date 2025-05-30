"""
This module provides functionality for detecting specific patterns
in PyTorch models' computation graphs.
It includes the `SegQuantPatternDetector` class,
which uses symbolic tracing to analyze models and identify
patterns such as linear layers followed by
chunking, splitting, concatenation, stacking, and activation functions.

Classes:
    SegQuantPatternDetector: A class for detecting computation graph patterns in PyTorch models.

Usage:
    The `SegQuantPatternDetector` class can be instantiated with a PyTorch model and example inputs.
    It supports searching for various patterns in the computation graph, such as:
        - Linear layers followed by chunking or splitting.
        - Concatenation or stacking of tensors followed by linear layers.
        - Activation functions followed by linear layers.

Example:
    ```
    model=MyModel()
    detector = SegQuantPatternDetector(
        model,
        example_inputs=(torch.randn(2, 10),),
        search_patterns_lst=[
            "linear_to_chunk",
            "linear_to_split",
            "concat_to_linear",
            "stack_to_linear",
            "activation_to_linear",
        ],
    )
    # Find all patterns in the model
    results = detector.find_all_patterns()
    print(results)
    ```
"""
import inspect
import warnings
from collections import namedtuple
import torch
from torch import nn
from torch import fx
from torch.fx.passes.shape_prop import ShapeProp
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.nn.functional as F


LinearInfo = namedtuple("LinearInfo", ["found", "name", "in_features", "out_features"])
ChunkInfo = namedtuple("ChunkInfo", ["found", "chunks"])
SplitInfo = namedtuple("SplitInfo", ["found", "split_size_or_sections"])
ConcatInfo = namedtuple("ConcatInfo", ["found", "chunksizes"])
StackInfo = namedtuple("StackInfo", ["found", "chunksizes"])
ActInfo = namedtuple("ActInfo", ["found", "name"])


class SegQuantPatternDetector:
    """
    A class to detect specific patterns in a PyTorch model's computation graph.
    This class uses symbolic tracing to analyze the model and identify patterns
    such as linear layers followed by chunking, splitting, concatenation, stacking,
    and activation functions.
    """
    def __init__(
        self,
        model: nn.Module,
        example_inputs: tuple,
        search_patterns_lst=None,
        acts=None,
        act_funcs=None,
    ):
        self.model = model
        self.acts = acts if acts is not None else [nn.SiLU, nn.GELU]
        self.act_funcs = act_funcs if act_funcs is not None else [F.silu, F.gelu]

        sig = inspect.signature(self.model.forward)
        param_names = list(sig.parameters.keys())
        concrete = dict(zip(param_names, example_inputs))
        expand_keys = [
            "block_controlnet_hidden_states",
            "controlnet_block_samples",
            "controlnet_single_block_samples",
        ]

        for key in expand_keys:
            value = concrete.get(key, None)
            if isinstance(value, (list, tuple)):
                concrete.update(
                    {f"{key}_{i+1}": tensor for i, tensor in enumerate(value)}
                )
                del concrete[key]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.traced = fx.symbolic_trace(model, concrete_args=concrete)
            self.module_map = dict(self.traced.named_modules())

            if search_patterns_lst is None:
                self.search_patterns = [
                    "linear_to_chunk",
                    "concat_to_linear",
                    "linear_to_split",
                    "stack_to_linear",
                    "activation_to_linear",
                ]
            else:
                self.search_patterns = search_patterns_lst

            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            ShapeProp(self.traced, fake_mode).propagate(*example_inputs)

    def _is_transparent(self, node):
        if node.op == "call_method" and node.target == "to":
            return True
        if node.op == "call_module":
            submod = self.module_map.get(node.target, None)
            if isinstance(submod, torch.nn.Dropout):
                return True

        return False

    def _is_linear(self, node):
        if node.op == "call_module":
            linear_mod = self.module_map.get(node.target, None)
            if isinstance(linear_mod, nn.Linear):
                return LinearInfo(
                    True, node.target, linear_mod.in_features, linear_mod.out_features
                )
        return LinearInfo(False, None, None, None)

    @staticmethod
    def _is_chunk(node):
        if (node.op == "call_method" and node.target == "chunk") or (
            node.op == "call_function" and node.target is torch.chunk
        ):
            chunks = node.args[1] if len(node.args) > 1 else node.kwargs.get("chunks")
            dim = node.kwargs.get("dim", 0)
            if dim == 1:
                return ChunkInfo(True, chunks)
        return ChunkInfo(False, None)

    @staticmethod
    def _is_split(node):
        if (node.op == "call_method" and node.target == "split") or (
            node.op == "call_function" and node.target is torch.split
        ):
            split_size_or_sections = (
                node.args[1] if len(node.args) > 1 else node.kwargs.get("split_size")
            )
            dim = node.kwargs.get("dim", 0)
            if dim == 1:
                return SplitInfo(True, split_size_or_sections)
        return SplitInfo(False, None)

    @staticmethod
    def _is_concat(node):
        if node.op == "call_function" and node.target is torch.cat:
            tensor_list = node.args[0]
            dim = node.kwargs.get("dim", 0)
            if dim == 1:
                if isinstance(tensor_list, (list, tuple)):
                    chunksizes = []
                    for tensor in tensor_list:
                        shape = tensor.meta["tensor_meta"].shape
                        chunksizes.append(shape[1])
                    return ConcatInfo(True, chunksizes)
        return ConcatInfo(False, None)

    @staticmethod
    def _is_stack(node):
        # only work for 1d-shape tensors to stack with dim = 1
        if node.op == "call_function" and node.target is torch.stack:
            tensor_list = node.args[0]
            dim = node.kwargs.get("dim", 0)
            if dim == 1:
                if isinstance(tensor_list, (list, tuple)):
                    chunksizes = []
                    for tensor in tensor_list:
                        shape = tensor.meta.get("tensor_meta", {}).get("shape", None)
                        if shape is not None and len(shape) == 1:
                            chunksizes.append(shape[0])
                    if len(chunksizes) == len(tensor_list):
                        return StackInfo(True, chunksizes)
        return StackInfo(False, None)

    def _is_activation(self, node):
        if node.op == "call_module":
            act_mod = self.module_map.get(node.target, None)
            if any(isinstance(act_mod, act) for act in self.acts):
                return ActInfo(True, node.target)
        elif node.op == "call_function" and node.target in self.act_funcs:
            return ActInfo(True, node.target)
        return ActInfo(False, None)

    def _find_pattern(self, node, pattern_type):
        if pattern_type == "linear_to_chunk":
            chunk_info = self._is_chunk(node)
            if chunk_info.found:
                input_node = node.args[0]
                linear_info = self._is_linear(input_node)
                if linear_info.found:
                    chunks = chunk_info.chunks
                    out_features = linear_info.out_features

                    chunk_size = out_features // chunks
                    remainder = out_features % chunks
                    chunksizes = [
                        chunk_size + (1 if i < remainder else 0) for i in range(chunks)
                    ]

                    return {
                        "seg_mode": "weight",
                        "linear_name": linear_info.name,
                        "linear_in": linear_info.in_features,
                        "linear_out": linear_info.out_features,
                        "chunksizes": chunksizes,
                    }

        elif pattern_type == "linear_to_split":
            split_info = self._is_split(node)
            if split_info.found:
                input_node = node.args[0]
                linear_info = self._is_linear(input_node)
                if linear_info.found:
                    out_features = linear_info.out_features
                    chunk_size = split_info.split_size_or_sections
                    num_full_chunks = out_features // chunk_size
                    remainder = out_features % chunk_size

                    chunksizes = [chunk_size] * num_full_chunks
                    if remainder > 0:
                        chunksizes.append(remainder)

                    return {
                        "seg_mode": "weight",
                        "linear_name": linear_info.name,
                        "linear_in": linear_info.in_features,
                        "linear_out": linear_info.out_features,
                        "chunksizes": chunksizes,
                    }

        elif pattern_type == "concat_to_linear":
            linear_info = self._is_linear(node)
            if linear_info.found:
                input_node = node.args[0]
                while isinstance(input_node, torch.fx.Node):
                    concat_info = self._is_concat(input_node)
                    if concat_info.found:
                        return {
                            "seg_mode": "input",
                            "linear_name": linear_info.name,
                            "linear_in": linear_info.in_features,
                            "linear_out": linear_info.out_features,
                            "chunksizes": concat_info.chunksizes,
                        }

                    if self._is_transparent(input_node):
                        input_node = input_node.args[0]
                        continue
                    break

        elif pattern_type == "stack_to_linear":
            linear_info = self._is_linear(node)
            if linear_info.found:
                input_node = node.args[0]
                while isinstance(input_node, torch.fx.Node):
                    stack_info = self._is_stack(input_node)
                    if stack_info.found:
                        return {
                            "seg_mode": "input",
                            "linear_name": linear_info.name,
                            "linear_in": linear_info.in_features,
                            "linear_out": linear_info.out_features,
                            "stack_size": stack_info.chunksizes,
                        }

                    if self._is_transparent(input_node):
                        input_node = input_node.args[0]
                        continue
                    break

        elif pattern_type == "activation_to_linear":
            linear_info = self._is_linear(node)
            if linear_info.found:
                input_node = node.args[0]
                while isinstance(input_node, torch.fx.Node):
                    act_info = self._is_activation(input_node)
                    if act_info.found:
                        return linear_info.name

                    if self._is_transparent(input_node):
                        input_node = input_node.args[0]
                        continue
                    break

        return None

    def find_all_patterns(self):
        """
        Find all patterns in the traced graph based on the specified search patterns.
        Returns:
            dict: A dictionary where keys are pattern names
            and values are lists of matched patterns.
        """
        patterns = {pattern: [] for pattern in self.search_patterns}

        for node in self.traced.graph.nodes:
            for pattern in self.search_patterns:
                matched_pattern = self._find_pattern(node, pattern)
                if matched_pattern:
                    patterns[pattern].append(matched_pattern)

        return patterns


if __name__ == "__main__":
    from typing import Optional
    from backend.torch.layers.activations import GELU

    class MyModel2(nn.Module):
        def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            mult: int = 4,
            dropout: float = 0.0,
            inner_dim=None,
            bias: bool = True,
        ):
            super().__init__()
            if inner_dim is None:
                inner_dim = int(dim * mult)
            dim_out = dim_out if dim_out is not None else dim

            act_fn = GELU(dim, inner_dim, bias=bias)

            self.net = nn.ModuleList([])
            # project in
            self.net.append(act_fn)
            # project dropout
            self.net.append(nn.Dropout(dropout))
            # project out
            self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            for module in self.net:
                hidden_states = module(hidden_states)
            return hidden_states

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 10)
            self.act = nn.SiLU()

        def forward(self, x):
            out = self.linear1(x)
            # a, b, c = out.chunk(3, dim=1)
            # a, b = out.split(10, dim=1)
            # out = torch.cat([a, b, c], dim=1)
            # out = torch.stack([a, b, c], dim=1)
            # c, d = out.split(2, dim=1)
            # out = torch.stack([c, d], dim=0)
            out = self.act(out)
            out = self.linear2(out)
            return out

    search_patterns = [
        "linear_to_chunk",
        "linear_to_split",
        "concat_to_linear",
        "stack_to_linear",
        "activation_to_linear",
    ]
    detector = SegQuantPatternDetector(
        MyModel2(dim=10),
        example_inputs=(torch.randn(2, 10),),
        search_patterns_lst=search_patterns,
    )
    results = detector.find_all_patterns()

    from pprint import pprint

    pprint(results)
