import torch
import torch.nn as nn
import torch.fx as fx
import inspect
from collections import namedtuple
from torch.fx.passes.shape_prop import ShapeProp
from torch._subclasses.fake_tensor import FakeTensorMode

LinearInfo = namedtuple("LinearInfo", ["found", "name", "in_features", "out_features"])
ChunkInfo = namedtuple("ChunkInfo", ["found", "chunks"])
SplitInfo = namedtuple("SplitInfo", ["found", "split_size_or_sections"])
ConcatInfo = namedtuple("ConcatInfo", ["found", "chunksizes"])
StackInfo = namedtuple("StackInfo", ["found", "chunksizes"])

class SegQuantPatternDetector:
    def __init__(self, model: nn.Module, example_inputs: tuple, search_patterns=None):
        self.model = model

        sig = inspect.signature(self.model.forward)
        param_names = list(sig.parameters.keys())
        concrete = {
            name: val for name, val in zip(param_names, example_inputs)
        }
        expand_keys = [
            'block_controlnet_hidden_states',
            'controlnet_block_samples',
            'controlnet_single_block_samples',
        ]

        for key in expand_keys:
            value = concrete.get(key, None)
            if isinstance(value, (list, tuple)):
                concrete.update({
                    f'{key}_{i+1}': tensor
                    for i, tensor in enumerate(value)
                })
                del concrete[key]

        self.traced = fx.symbolic_trace(model, concrete_args=concrete)
        self.module_map = dict(self.traced.named_modules())
        
        if search_patterns is None:
            self.search_patterns = [
                'linear_to_chunk', 'concat_to_linear', 'linear_to_split', 'stack_to_linear'
            ]
        else:
            self.search_patterns = search_patterns
        
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        ShapeProp(self.traced, fake_mode).propagate(*example_inputs)

    def _is_linear(self, node):
        if node.op == 'call_module':
            linear_mod = self.module_map.get(node.target, None)
            if isinstance(linear_mod, nn.Linear):
                return LinearInfo(True, node.target, linear_mod.in_features, linear_mod.out_features)
        return LinearInfo(False, None, None, None)

    def _is_chunk(self, node):
        if (node.op == "call_method" and node.target == 'chunk') or \
           (node.op == 'call_function' and node.target is torch.chunk):
            chunks = node.args[1] if len(node.args) > 1 else node.kwargs.get("chunks")
            dim = node.kwargs.get("dim", 0)
            if dim == 1:
                return ChunkInfo(True, chunks)
        return ChunkInfo(False, None)

    def _is_split(self, node):
        if (node.op == "call_method" and node.target == 'split') or \
           (node.op == 'call_function' and node.target is torch.split):
            split_size_or_sections = node.args[1] if len(node.args) > 1 else node.kwargs.get("split_size")
            dim = node.kwargs.get("dim", 0)
            if dim == 1:
                return SplitInfo(True, split_size_or_sections)
        return SplitInfo(False, None)

    def _is_concat(self, node):
        if node.op == "call_function" and node.target is torch.cat:
            tensor_list = node.args[0]
            dim = node.kwargs.get("dim", 0)
            if dim == 1:
                if isinstance(tensor_list, (list, tuple)):
                    chunksizes = []
                    for tensor in tensor_list:
                        shape = tensor.meta['tensor_meta'].shape
                        chunksizes.append(shape[1])
                    return ConcatInfo(True, chunksizes)
        return ConcatInfo(False, None)

    def _is_stack(self, node):
        # only work for 1d-shape tensors to stack with dim = 1
        if node.op == "call_function" and node.target is torch.stack:
            tensor_list = node.args[0]
            dim = node.kwargs.get("dim", 0)
            if dim == 1:
                if isinstance(tensor_list, (list, tuple)):
                    chunksizes = []
                    for tensor in tensor_list:
                        shape = tensor.meta.get('tensor_meta', {}).get('shape', None)
                        if shape is not None and len(shape) == 1:
                            chunksizes.append(shape[0])
                    if len(chunksizes) == len(tensor_list):
                        return StackInfo(True, chunksizes)
        return StackInfo(False, None)

    def _find_pattern(self, node, pattern_type):
        if pattern_type == 'linear_to_chunk':
            chunk_info = self._is_chunk(node)
            if chunk_info.found: 
                input_node = node.args[0]
                linear_info = self._is_linear(input_node)
                if linear_info.found:
                    chunks = chunk_info.chunks
                    out_features = linear_info.out_features

                    chunk_size = out_features // chunks
                    remainder = out_features % chunks
                    chunksizes = [chunk_size + (1 if i < remainder else 0) for i in range(chunks)]

                    return {
                        "seg_mode": "weight",
                        "linear_name": linear_info.name,
                        "linear_in": linear_info.in_features,
                        "linear_out": linear_info.out_features,
                        "chunksizes": chunksizes,
                    }

        elif pattern_type == 'linear_to_split':
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

        elif pattern_type == 'concat_to_linear':
            linear_info = self._is_linear(node)
            if linear_info.found:
                input_node = node.args[0]
                if isinstance(input_node, torch.fx.Node):
                    concat_info = self._is_concat(input_node)
                    if concat_info.found:
                        return {
                            "seg_mode": "input",
                            "linear_name": linear_info.name,
                            "linear_in": linear_info.in_features,
                            "linear_out": linear_info.out_features,
                            "chunksizes": concat_info.chunksizes,
                        }

        elif pattern_type == 'stack_to_linear':
            linear_info = self._is_linear(node)
            if linear_info.found:
                input_node = node.args[0]
                if isinstance(input_node, torch.fx.Node):
                    stack_info = self._is_stack(input_node)
                    if stack_info.found:
                        return {
                            "seg_mode": "input",
                            "linear_name": linear_info.name,
                            "linear_in": linear_info.in_features,
                            "linear_out": linear_info.out_features,
                            "stack_size": stack_info.chunksizes,
                        }

        return None


    def find_all_patterns(self):
        patterns = {pattern: [] for pattern in self.search_patterns}

        for node in self.traced.graph.nodes:
            for pattern in self.search_patterns:
                matched_pattern = self._find_pattern(node, pattern)
                if matched_pattern:
                    patterns[pattern].append(matched_pattern)

        return patterns



if __name__ == '__main__':
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 10)

        def forward(self, x):
            out = self.linear1(x)
            a, b, c = out.chunk(3, dim=1)
            # a, b = out.split(10, dim=1)
            out = torch.cat([a, b, c], dim=1)
            # out = torch.stack([a, b, c], dim=1)
            # c, d = out.split(2, dim=1)
            # out = torch.stack([c, d], dim=0)
            out = self.linear2(out)
            return out

    search_patterns = ['linear_to_chunk', 'linear_to_split', 'concat_to_linear', 'stack_to_linear']
    # search_patterns = ['linear_to_split', 'linear_to_chunk']
    # search_patterns = ['concat_to_linear', 'stack_to_linear']
    detector = SegQuantPatternDetector(MyModel(), example_inputs=(torch.randn(2, 10),), search_patterns=search_patterns)
    results = detector.find_all_patterns()

    from pprint import pprint
    pprint(results)
