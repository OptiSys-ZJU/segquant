import torch
import torch.nn as nn
import torch.fx as fx
from collections import namedtuple

LinearInfo = namedtuple("LinearInfo", ["found", "module", "in_features", "out_features"])
ChunkInfo = namedtuple("ChunkInfo", ["found", "chunk_size"])
SplitInfo = namedtuple("SplitInfo", ["found", "split_size"])
ConcatInfo = namedtuple("ConcatInfo", ["found", "concat_size"])
StackInfo = namedtuple("StackInfo", ["found", "stack_size"])

class SegQuantPatternDetector:
    def __init__(self, graph_module):
        self.module = graph_module
        self.graph = graph_module.graph
        self.module_map = dict(graph_module.named_modules())

    def _is_linear(self, node):
        if node.op == 'call_module':
            linear_mod = self.module_map.get(node.target, None)
            if isinstance(linear_mod, nn.Linear):
                return LinearInfo(True, linear_mod, linear_mod.in_features, linear_mod.out_features)
        return LinearInfo(False, None, None, None)

    def _is_chunk(self, node):
        if (node.op == "call_method" and node.target == 'chunk') or \
           (node.op == 'call_function' and node.target is torch.chunk):
            chunks = node.args[1] if len(node.args) > 1 else node.kwargs.get("chunks")
            dim = node.kwargs.get("dim", 0)
            if dim == 1:
                return ChunkInfo(True, chunks, dim)
        return ChunkInfo(False, None, None)

    def _is_split(self, node):
        if (node.op == "call_method" and node.target == 'split') or \
           (node.op == 'call_function' and node.target is torch.split):
            split_size_or_sections = node.args[1] if len(node.args) > 1 else node.kwargs.get("split_size")
            dim = node.kwargs.get("dim", 0)
            if dim == 1:
                return SplitInfo(True, split_size_or_sections)
        return SplitInfo(False, None, None)

    def _is_concat(self, node):
        if node.op == "call_function" and node.target is torch.cat:
            tensor_list = node.args[0]
            num_tensors = len(tensor_list) if isinstance(tensor_list, (list, tuple)) else None
            dim = node.kwargs.get("dim", 0)
            if dim == 1:
                return ConcatInfo(True, num_tensors, dim)
        return ConcatInfo(False, None, None)

    def _is_stack(self, node):
        if node.op == "call_function" and node.target is torch.stack:
            tensor_list = node.args[0]
            num_tensors = len(tensor_list) if isinstance(tensor_list, (list, tuple)) else None
            dim = node.kwargs.get("dim", 0)
            if dim == 1:
                return StackInfo(True, num_tensors, dim)
        return StackInfo(False, None, None)

    def __init__(self, model: nn.Module, search_patterns=None):
        self.model = model
        self.traced = fx.symbolic_trace(model)
        self.module_map = dict(self.traced.named_modules())
        
        if search_patterns is None:
            self.search_patterns = [
                'linear_to_chunk', 'concat_to_linear', 'linear_to_split', 'stack_to_linear'
            ]
        else:
            self.search_patterns = search_patterns

    def _find_pattern(self, node, pattern_type):
        if pattern_type == 'linear_to_chunk':
            chunk_info = self._is_chunk(node)
            if chunk_info.found: 
                input_node = node.args[0]
                linear_info = self._is_linear(input_node)
                if linear_info.found:
                    return {
                        "type": "linear→chunk",
                        "linear_in": linear_info.in_features,
                        "linear_out": linear_info.out_features,
                        "chunk_size": chunk_info.chunk_size,
                        "chunk_dim": chunk_info.dim,
                    }

        elif pattern_type == 'linear_to_split':
            split_info = self._is_split(node)
            if split_info.found:
                input_node = node.args[0]
                linear_info = self._is_linear(input_node)
                if linear_info.found:
                    return {
                        "type": "linear→split",
                        "linear_in": linear_info.in_features,
                        "linear_out": linear_info.out_features,
                        "split_size": split_info.split_size,
                        "split_dim": split_info.dim,
                    }

        elif pattern_type == 'concat_to_linear':
            linear_info = self._is_linear(node)
            if linear_info.found:
                input_node = node.args[0]
                if isinstance(input_node, torch.fx.Node):
                    concat_info = self._is_concat(input_node)
                    if concat_info.found:
                        return {
                            "type": "concat→linear",
                            "linear_in": linear_info.in_features,
                            "linear_out": linear_info.out_features,
                            "concat_size": concat_info.concat_size,
                            "concat_dim": concat_info.dim,
                        }

        elif pattern_type == 'stack_to_linear':
            linear_info = self._is_linear(node)
            if linear_info.found:
                input_node = node.args[0]
                if isinstance(input_node, torch.fx.Node):
                    stack_info = self._is_stack(input_node)
                    if stack_info.found:
                        return {
                            "type": "stack→linear",
                            "linear_in": linear_info.in_features,
                            "linear_out": linear_info.out_features,
                            "stack_size": stack_info.stack_size,
                            "stack_dim": stack_info.dim,
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
            self.linear2 = nn.Linear(40, 10)

        def forward(self, x):
            out = self.linear1(x)
            a, b = out.chunk(6, dim=1)
            out = torch.cat([a, b], dim=1)
            # c, d = out.split(2, dim=1)
            # out = torch.stack([c, d], dim=0)
            out = self.linear2(out)
            return out

    search_patterns = ['linear_to_chunk', 'linear_to_split', 'concat_to_linear', 'stack_to_linear']
    # search_patterns = ['linear_to_chunk']
    detector = SegQuantPatternDetector(MyModel(), search_patterns=search_patterns)
    results = detector.find_all_patterns()

    from pprint import pprint
    pprint(results)
