from typing import Tuple
import torch
import torch.nn as nn
import copy
from segquant.config import Calibrate, DType, Optimum, SegPattern
from segquant.torch.quantization import quantize


embedding_dim = 1536

class RandomTensorDataset:
    def __init__(self, num_batches=6, seed=42):
        self.num_batches = num_batches
        self.seed = seed
        self._generate_batches()

    def _generate_batches(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        self.batches = [
            (
                torch.rand(2, embedding_dim, generator=generator),
                torch.rand(2, embedding_dim, generator=generator),
            )
            for _ in range(self.num_batches)
        ]

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
    def __init__(self, embedding_dim: int = 16, bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        linear_input = self.silu(emb)
        linear_out = self.linear(linear_input)

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = linear_out.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

def test_search():
    test_model = TestModel(embedding_dim)
    ######################################
    config = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": SegPattern.all(),
            "real_quant": False,
            "opt": {
                "type": Optimum.SVD,
                "alpha": 0.5,
            },
            "calib": {
                "type": Calibrate.AMAX,
            },
            "input_quant": {
                "type": DType.INT8,
                "axis": None,
            },
            "weight_quant": {
                "type": DType.INT8,
                "axis": None,
            },
        },
    }
    nosearch_model = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        config,
        True,
        example=(torch.rand(2, embedding_dim), torch.rand(2, embedding_dim)),
    )
    ######################################
    search_config = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": SegPattern.all(),
            "real_quant": False,
            "opt": {
                "type": Optimum.SVD,
                "alpha": 0.5,
                "search_alpha_config": {
                    "enable": True,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                },
            },
            "calib": {
                "type": Calibrate.AMAX,
            },
            "input_quant": {
                "type": DType.INT8,
                "axis": None,
            },
            "weight_quant": {
                "type": DType.INT8,
                "axis": None,
            },
        },
    }
    search_model = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        search_config,
        True,
        example=(torch.rand(2, embedding_dim), torch.rand(2, embedding_dim)),
    )
    ######################################
    x_generator = torch.Generator()
    x_generator.manual_seed(1234)
    x = torch.rand(2, embedding_dim, generator=x_generator)
    emb = torch.rand(2, embedding_dim, generator=x_generator)
    res = test_model.forward(x, emb)
    print("origin:", res)
    a = res[0]

    res = nosearch_model.forward(x, emb)
    print("nosearch:", res)
    b = res[0]

    res = search_model.forward(x, emb)
    print("search:", res)
    c = res[0]

    print('origin-nosearch', torch.norm(a - b).item())
    print('origin-search', torch.norm(a - c).item())

if __name__ == "__main__":
    test_search()
