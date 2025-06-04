from typing import Tuple
import torch
import torch.nn as nn
import copy
import modelopt.torch.quantization as mtq
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


def test_default_fp8():
    test_model = TestModel(embedding_dim)
    ######################################
    CFG = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
            "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        },
        "algorithm": "max",
    }
    modelopt_model = mtq.quantize(copy.deepcopy(test_model), CFG, forward_loop)
    mtq.print_quant_summary(modelopt_model)
    ######################################
    config = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": [],
            # "search_patterns": SegPattern.all(),
            "real_quant": False,
            "opt": {
                "type": Optimum.DEFAULT,
                "alpha": 0.5,
            },
            "calib": {
                "type": Calibrate.AMAX,
            },
            "input_quant": {
                "type": DType.FP8E4M3,
                "axis": None,
            },
            "weight_quant": {
                "type": DType.FP8E4M3,
                "axis": None,
            },
        },
    }
    segquant_model = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        config,
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
    res = modelopt_model.forward(x, emb)
    print("modelopt:", res)
    b = res[0]
    res = segquant_model.forward(x, emb)
    print("segquant:", res)
    c = res[0]

    print('diff1', torch.norm(a - b).item())
    print('diff2', torch.norm(a - c).item())
    print('diff3', torch.norm(b - c).item())

def test_default_fp8_real():
    test_model = TestModel(embedding_dim).to(torch.device("cuda:0"))
    ######################################
    config = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": [],
            # "search_patterns": SegPattern.all(),
            "real_quant": False,
            "opt": {
                "type": Optimum.DEFAULT,
                "alpha": 0.5,
            },
            "calib": {
                "type": Calibrate.AMAX,
            },
            "input_quant": {
                "type": DType.FP8E4M3,
                "axis": None,
            },
            "weight_quant": {
                "type": DType.FP8E4M3,
                "axis": None,
            },
        },
    }
    segquant_model = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        config,
        True,
        example=(torch.rand(2, embedding_dim), torch.rand(2, embedding_dim)),
    )

    ######################################
    config_real = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": [],
            # "search_patterns": SegPattern.all(),
            "real_quant": True,
            "opt": {
                "type": Optimum.DEFAULT,
                "alpha": 0.5,
            },
            "calib": {
                "type": Calibrate.AMAX,
            },
            "input_quant": {
                "type": DType.FP8E4M3,
                "axis": None,
            },
            "weight_quant": {
                "type": DType.FP8E4M3,
                "axis": None,
            },
        },
    }
    segquant_model_real = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        config_real,
        True,
        example=(torch.rand(2, embedding_dim), torch.rand(2, embedding_dim)),
    )
    ######################################
    x_generator = torch.Generator()
    x_generator.manual_seed(1234)
    x = torch.rand(2, embedding_dim, generator=x_generator).to(torch.device("cuda:0"))
    emb = torch.rand(2, embedding_dim, generator=x_generator).to(torch.device("cuda:0"))
    res = test_model.forward(x, emb)
    print("origin:", res)
    a = res[0]
    res = segquant_model.forward(x, emb)
    print("fake:", res)
    b = res[0]
    res = segquant_model_real.forward(x, emb)
    print("real:", res)
    c = res[0]

    print('diff1', torch.norm(a - b).item())
    print('diff2', torch.norm(a - c).item())
    print('diff3', torch.norm(b - c).item())

def test_default_int8():
    test_model =TestModel(embedding_dim)
    ######################################
    CFG = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 8, "axis": None,},
            "*input_quantizer": {"num_bits": 8, "axis": None,},
        },
        "algorithm": "max",
    }
    modelopt_model = mtq.quantize(copy.deepcopy(test_model), CFG, forward_loop)
    mtq.print_quant_summary(modelopt_model)
    ######################################
    config = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": [],
            # "search_patterns": SegPattern.all(),
            "real_quant": False,
            "opt": {
                "type": Optimum.DEFAULT,
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
    segquant_model = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        config,
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
    res = modelopt_model.forward(x, emb)
    print("modelopt:", res)
    b = res[0]
    res = segquant_model.forward(x, emb)
    print("segquant:", res)
    c = res[0]

    print('diff1', torch.norm(a - b).item())
    print('diff2', torch.norm(a - c).item())
    print('diff3', torch.norm(b - c).item())

def test_smooth_int8():
    test_model = TestModel(embedding_dim)
    ######################################
    CFG = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 8, "axis": None},
            "*input_quantizer": {"num_bits": 8, "axis": -1},
        },
        "algorithm": {"method": "smoothquant", "alpha": 0.5},
    }
    modelopt_model = mtq.quantize(copy.deepcopy(test_model), CFG, forward_loop)
    mtq.print_quant_summary(modelopt_model)
    ######################################
    config = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": [],
            # "search_patterns": SegPattern.all(),
            "real_quant": False,
            "opt": {
                "type": Optimum.SMOOTH,
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
    segquant_model = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        config,
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
    res = modelopt_model.forward(x, emb)
    print("modelopt:", res)
    b = res[0]
    res = segquant_model.forward(x, emb)
    print("segquant:", res)
    c = res[0]

    print('diff1', torch.norm(a - b).item())
    print('diff2', torch.norm(a - c).item())
    print('diff3', torch.norm(b - c).item())

def test_smooth_int8_real():
    test_model = TestModel(embedding_dim).to(torch.device("cuda:0"))
    ######################################
    config = {
        "default": {
            "enable": True,
            "seglinear": True,
            # "search_patterns": [],
            "search_patterns": SegPattern.all(),
            "real_quant": False,
            "opt": {
                "type": Optimum.SMOOTH,
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
    segquant_model = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        config,
        True,
        example=(torch.rand(2, embedding_dim), torch.rand(2, embedding_dim)),
    )
    ######################################
    config_real = {
        "default": {
            "enable": True,
            "seglinear": True,
            # "search_patterns": [],
            "search_patterns": SegPattern.all(),
            "real_quant": True,
            "opt": {
                "type": Optimum.SMOOTH,
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
    segquant_model_real = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        config_real,
        True,
        example=(torch.rand(2, embedding_dim), torch.rand(2, embedding_dim)),
    )
    ######################################
    x_generator = torch.Generator()
    x_generator.manual_seed(1234)
    x = torch.rand(2, embedding_dim, generator=x_generator).to(torch.device("cuda:0"))
    emb = torch.rand(2, embedding_dim, generator=x_generator).to(torch.device("cuda:0"))
    res = test_model.forward(x, emb)
    print("origin:", res)
    a = res[0]
    res = segquant_model.forward(x, emb)
    print("fake:", res)
    b = res[0]
    res = segquant_model_real.forward(x, emb)
    print("real:", res)
    c = res[0]

    print('diff1', torch.norm(a - b).item())
    print('diff2', torch.norm(a - c).item())
    print('diff3', torch.norm(b - c).item())

def test_svd_int4():
    test_model = TestModel(embedding_dim)
    ######################################
    CFG = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 4, "axis": None},
            "*input_quantizer": {"num_bits": 4, "axis": -1},
            # "linear.weight_quantizer": {
            #     "num_bits": 8,
            #     "block_sizes": {0: 10},
            #     "enable": True,
            # },
        },
        "algorithm": {"method": "svdquant", "lowrank": 32},
    }
    modelopt_model = mtq.quantize(copy.deepcopy(test_model), CFG, forward_loop)
    mtq.print_quant_summary(modelopt_model)
     ######################################
    config = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": [],
            # "search_patterns": SegPattern.all(),
            "real_quant": False,
            "opt": {
                "type": Optimum.SVD,
                "alpha": 0.5,
                "low_rank": 32,
            },
            "calib": {
                "type": Calibrate.AMAX,
            },
            "input_quant": {
                "type": DType.INT4,
                "axis": None,
            },
            "weight_quant": {
                "type": DType.INT4,
                "axis": None,
            },
        },
    }
    segquant_model = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        config,
        True,
        example=(torch.rand(2, embedding_dim), torch.rand(2, embedding_dim)),
    )
    ######################################
    x_generator = torch.Generator()
    x_generator.manual_seed(1234)
    x = torch.rand(2, embedding_dim, generator=x_generator)
    emb = torch.rand(2, embedding_dim, generator=x_generator)
    res = test_model.forward(x, emb)
    print("origin:", res[0])
    a = res[0]
    res = segquant_model.forward(x, emb)
    print("modelopt:", res[0])
    b = res[0]
    res = segquant_model.forward(x, emb)
    print("segquant:", res[0])
    c = res[0]
    print('diff1', torch.norm(a - b).item())
    print('diff2', torch.norm(a - c).item())
    print('diff3', torch.norm(b - c).item())

def test_svd_int4_real():
    test_model = TestModel(embedding_dim).to(torch.device("cuda:0"))
    ######################################
    config = {
        "default": {
            "enable": True,
            "seglinear": True,
            # "search_patterns": [],
            "search_patterns": SegPattern.all(),
            "real_quant": False,
            "opt": {
                "type": Optimum.SVD,
                "alpha": 0.5,
                "low_rank": 32,
            },
            "calib": {
                "type": Calibrate.AMAX,
            },
            "input_quant": {
                "type": DType.INT4,
                "axis": None,
            },
            "weight_quant": {
                "type": DType.INT4,
                "axis": None,
            },
        },
    }
    segquant_model = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        config,
        True,
        example=(torch.rand(2, embedding_dim), torch.rand(2, embedding_dim)),
    )
    ######################################
    config_real = {
        "default": {
            "enable": True,
            "seglinear": True,
            # "search_patterns": [],
            "search_patterns": SegPattern.all(),
            "real_quant": True,
            "opt": {
                "type": Optimum.SVD,
                "alpha": 0.5,
                "low_rank": 32,
            },
            "calib": {
                "type": Calibrate.AMAX,
            },
            "input_quant": {
                "type": DType.INT4,
                "axis": None,
            },
            "weight_quant": {
                "type": DType.INT4,
                "axis": None,
            },
        },
    }
    segquant_model_real = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        config_real,
        True,
        example=(torch.rand(2, embedding_dim), torch.rand(2, embedding_dim)),
    )
    ######################################
    x_generator = torch.Generator()
    x_generator.manual_seed(1234)
    x = torch.rand(2, embedding_dim, generator=x_generator).to(torch.device("cuda:0"))
    emb = torch.rand(2, embedding_dim, generator=x_generator).to(torch.device("cuda:0"))
    res = test_model.forward(x, emb)
    print("origin:", res)
    a = res[0]
    res = segquant_model.forward(x, emb)
    print("fake:", res)
    b = res[0]
    res = segquant_model_real.forward(x, emb)
    print("real:", res)
    c = res[0]

    print('diff1', torch.norm(a - b).item())
    print('diff2', torch.norm(a - c).item())
    print('diff3', torch.norm(b - c).item())

if __name__ == "__main__":
    test_svd_int4_real()
