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
                "alpha": 0.2,
                "search": True,
                "step": 0.1,
                "end": 0.8
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

    print('origin-modelopt', torch.norm(a - b).item())
    print('origin-segquant', torch.norm(a - c).item())

def test_smooth_int8_real():
    test_model = TestModel(embedding_dim).to(torch.device("cuda:0"))
    ######################################
    config = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": [SegPattern.ACTIVATION2LINEAR],
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
                "axis": -1,
            },
            "weight_quant": {
                "type": DType.INT8,
                "axis": 1,
            },
        },
    }
    segquant_model = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        config,
        per_layer_mode=False,
        verbose=True,
        example=(torch.rand(2, embedding_dim), torch.rand(2, embedding_dim)),
    )
    ######################################
    config_real = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": [SegPattern.ACTIVATION2LINEAR],
            # "search_patterns": SegPattern.all(),
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
                "axis": -1,
            },
            "weight_quant": {
                "type": DType.INT8,
                "axis": 1,
            },
        },
    }
    segquant_model_real = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        config_real,
        per_layer_mode=False,
        verbose=True,
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
            "*weight_quantizer": {"num_bits": 8, "axis": None},
            "*input_quantizer": {"num_bits": 8, "axis": -1},
            # "linear.weight_quantizer": {
            #     "num_bits": 8,
            #     "block_sizes": {0: 10},
            #     "enable": True,
            # },
        },
        "algorithm": {"method": "svdquant", "lowrank": 16},
    }
    modelopt_model = mtq.quantize(copy.deepcopy(test_model), CFG, forward_loop)
    mtq.print_quant_summary(modelopt_model)
    ######################################
    CFG = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 8, "axis": None},
            "*input_quantizer": {"num_bits": 8, "axis": -1},
            # "linear.weight_quantizer": {
            #     "num_bits": 8,
            #     "block_sizes": {0: 10},
            #     "enable": True,
            # },
        },
        "algorithm": {"method": "smoothquant", "alpha": 0.5},
    }
    modelopt_smooth_model = mtq.quantize(copy.deepcopy(test_model), CFG, forward_loop)
    mtq.print_quant_summary(modelopt_smooth_model)
    ######################################
    smooth_config = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": [],
            # "search_patterns": SegPattern.all(),
            "real_quant": False,
            "opt": {
                "type": Optimum.SMOOTH,
                "alpha": 0.5,
                "low_rank": 16,
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
    smooth_model = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        smooth_config,
        True,
        example=(torch.rand(2, embedding_dim), torch.rand(2, embedding_dim)),
    )
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
                "low_rank": 16,
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
    res = test_model.forward(x.clone(), emb.clone())[0].clone()
    print("origin:", res)
    a = res.clone()
    res = modelopt_model.forward(x.clone(), emb.clone())[0].clone()
    print("modelopt:", res)
    b = res.clone()
    res = segquant_model.forward(x.clone(), emb.clone())[0].clone()
    print("segquant:", res)
    c = res.clone()

    res = modelopt_smooth_model.forward(x.clone(), emb.clone())[0].clone()
    print("segquant-smooth:", res)
    d = res.clone()
    res = smooth_model.forward(x.clone(), emb.clone())[0].clone()
    print("smooth:", res)
    e = res.clone()
    print('origin-modelopt-svd', torch.norm(a - b).item())
    print('origin-segquant-svd', torch.norm(a - c).item())

    print('origin-modelopt-smooth', torch.norm(a - d).item())
    print('origin-segquant-smooth', torch.norm(a - e).item())


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

def test_gptq():
    test_model = TestModel(embedding_dim).to(torch.device("cuda:0"))
    ######################################
    config = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": [],
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
    config_gptq = {
        "default": {
            "enable": True,
            "seglinear": True,
            "search_patterns": [],
            "real_quant": False,
            "opt": {
                "type": Optimum.DEFAULT,
                "alpha": 0.5,
            },
            "calib": {
                "type": Calibrate.GPTQ,
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
    segquant_model_gptq = quantize(
        copy.deepcopy(test_model),
        seg_data_loader,
        config_gptq,
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
    print("amax:", res)
    b = res[0]
    res = segquant_model_gptq.forward(x, emb)
    print("gptq:", res)
    c = res[0]

    print('diff1', torch.norm(a - b).item())
    print('diff2', torch.norm(a - c).item())
    print('diff3', torch.norm(b - c).item())


def test_mix_real():
    def step_mix(atype, wtype, axis=False, dual=False):
        test_model = TestModel(embedding_dim).to(torch.device("cuda:0"))

        # ==============================
        # Baseline Quant (fake)
        # ==============================
        config = {
            "default": {
                "enable": True,
                "seglinear": True,
                "search_patterns": [SegPattern.ACTIVATION2LINEAR] if dual else [],
                "real_quant": False,
                "opt": {
                    "type": Optimum.SMOOTH,
                    "alpha": 0.5,
                },
                "calib": {
                    "type": Calibrate.AMAX,
                },
                "input_quant": {
                    "type": atype,
                    "axis": -1 if axis else None,
                },
                "weight_quant": {
                    "type": wtype,
                    "axis": 1 if axis else None,
                },
            },
        }

        # Fake quantization model
        segquant_model = quantize(
            copy.deepcopy(test_model),
            seg_data_loader,
            config,
            per_layer_mode=False,
            verbose=False,
            example=(torch.rand(2, embedding_dim), torch.rand(2, embedding_dim)),
        )

        # ==============================
        # Real Quant (int4/fp8 kernel path)
        # ==============================
        config["default"]["real_quant"] = True

        segquant_model_real = quantize(
            copy.deepcopy(test_model),
            seg_data_loader,
            config,
            per_layer_mode=False,
            verbose=False,
            example=(torch.rand(2, embedding_dim), torch.rand(2, embedding_dim)),
        )

        # ==============================
        # Run and Compare
        # ==============================
        x_generator = torch.Generator().manual_seed(1234)
        x = torch.rand(2, embedding_dim, generator=x_generator).to(torch.device("cuda:0"))
        emb = torch.rand(2, embedding_dim, generator=x_generator).to(torch.device("cuda:0"))

        a = test_model(x, emb)[0]            # FP baseline
        b = segquant_model(x, emb)[0]        # Fake quant output
        c = segquant_model_real(x, emb)[0]   # Real quant output

        # ==============================
        # Print result
        # ==============================
        print(f"\n[ {atype} * {wtype} ] axis: {axis}, dual: {dual}")
        print("→ diff1 (fp - fake):", torch.norm(a - b).item())
        print("→ diff2 (fp - real):", torch.norm(a - c).item())
        print("→ diff3 (fake - real):", torch.norm(b - c).item())

    # ==============================
    # All Config Combinations
    # ==============================
    configs = [
        (DType.INT8, DType.INT8),
        (DType.INT4, DType.INT4),
        (DType.FP8E4M3, DType.FP8E4M3),
        (DType.INT8, DType.INT4),
        (DType.FP16, DType.INT8),
        (DType.FP16, DType.INT4),
    ]

    step_mix(DType.FP16, DType.INT4, False, False)

    # for atype, wtype in configs:
    #     for axis in [False, True]:
    #         for dual in [False, True]:
    #             step_mix(atype, wtype, axis, dual)


if __name__ == "__main__":
    test_mix_real()
