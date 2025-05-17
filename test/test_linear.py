from typing import Tuple
import torch
import torch.nn as nn
import copy
import modelopt.torch.quantization as mtq
from segquant.config import DType, SegPattern
from segquant.torch.quantization import quantize

class RandomTensorDataset:
    def __init__(self, num_batches=6, seed=42):
        self.num_batches = num_batches
        self.seed = seed
        self._generate_batches()

    def _generate_batches(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        self.batches = [(torch.rand(2, 10, generator=generator), torch.rand(2, 10, generator=generator)) for _ in range(self.num_batches)]

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return self.batches[idx]

dataset = RandomTensorDataset(num_batches=5, seed=42)

def create_seg_data_loader(dataset):
    class DataLoader:
        def __iter__(self):
            for batch in dataset:
                yield [batch]
    return DataLoader()
seg_data_loader = create_seg_data_loader(dataset)

def create_modelopt_data_loader(dataset):
    class DataLoader:
        def __iter__(self):
            for batch in dataset:
                yield batch
    return DataLoader()
modelopt_data_loader = create_modelopt_data_loader(dataset)
def forward_loop(model):
    for batch in modelopt_data_loader:
        model(*batch)

class TestModel(nn.Module):
    def __init__(self, embedding_dim: int=10, bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,x: torch.Tensor,emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:        
        linear_input = self.silu(emb)
        linear_out = self.linear(linear_input)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = linear_out.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

def test_default_int8():
    test_model = TestModel()
    ######################################
    CFG = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 8, "axis": None},
            "*input_quantizer": {"num_bits": 8, "axis": None},
            "linear.weight_quantizer": {
                "num_bits": 8, 
                "block_sizes": {0: 10}, 
                "enable": True
            }
        },
        "algorithm": "max",
    }
    modelopt_model = mtq.quantize(copy.deepcopy(test_model), CFG, forward_loop)
    mtq.print_quant_summary(modelopt_model)
    ######################################
    config = {
        "default": {
            "enable": True,
            "dtype": DType.INT8,
            "seglinear": True,
            'search_patterns': SegPattern.all(),
            "input_axis": None,
            "weight_axis": None,
        },
    }
    segquant_model = quantize(copy.deepcopy(test_model), seg_data_loader, (torch.rand(2, 10), torch.rand(2, 10)), config, True)
    ######################################
    x_generator = torch.Generator()
    x_generator.manual_seed(1234)
    x = torch.rand(2, 10, generator=x_generator)
    emb = torch.rand(2, 10, generator=x_generator)
    res = test_model.forward(x, emb)
    print('origin:', res)
    res = modelopt_model.forward(x)
    print('modelopt:', res)
    res = segquant_model.forward(x)
    print('segquant:', res)

def test_smooth_int8():
    test_model = TestModel()
    ######################################
    CFG = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 8, "axis": None},
            "*input_quantizer": {"num_bits": 8, "axis": -1},
            "linear.weight_quantizer": {
                "num_bits": 8, 
                "block_sizes": {0: 10}, 
                "enable": True
            }
        },
        "algorithm": {
            "method": "smoothquant",
            "alpha": 0.5
        },
    }
    modelopt_model = mtq.quantize(copy.deepcopy(test_model), CFG, forward_loop)
    mtq.print_quant_summary(modelopt_model)
    ######################################
    config = {
        "default": {
            "enable": True,
            "dtype": DType.INT8SMOOTH,
            "seglinear": True,
            'search_patterns': SegPattern.seg(),
            "input_axis": None,
            "weight_axis": None,
            "alpha": 0.5,
        },
    }
    segquant_model = quantize(copy.deepcopy(test_model), seg_data_loader, config, True, example=(torch.rand(2, 10), torch.rand(2, 10)))
    ######################################
    x_generator = torch.Generator()
    x_generator.manual_seed(1234)
    x = torch.rand(2, 10, generator=x_generator)
    emb = torch.rand(2, 10, generator=x_generator)
    res = test_model.forward(x, emb)
    print('origin:', res[0])
    res = modelopt_model.forward(x, emb)
    print('modelopt:', res[0])
    res = segquant_model.forward(x, emb)
    print('segquant:', res[0])

if __name__ == '__main__':
    test_smooth_int8()