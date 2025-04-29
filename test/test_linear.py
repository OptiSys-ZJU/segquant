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
        self.batches = [torch.rand(2, 10, generator=generator) for _ in range(self.num_batches)]

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return self.batches[idx]

dataset = RandomTensorDataset(num_batches=5, seed=42)

def create_seg_data_loader(dataset):
    class DataLoader:
        def __iter__(self):
            for batch in dataset:
                yield [(batch,)]
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
        model(batch)

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=10, out_features=5, bias=False)
        self.linear2 = nn.Linear(in_features=5, out_features=5, bias=False)
    
    def forward(self, x):
        y = self.linear1(x)
        z = self.linear2(y)
        return z

def test_default_int8():
    test_model = TestModel()
    ######################################
    CFG = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 8, "axis": None},
            "*input_quantizer": {"num_bits": 8, "axis": None},
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
    segquant_model = quantize(copy.deepcopy(test_model), seg_data_loader, torch.rand(2, 10), config, True)
    ######################################
    x_generator = torch.Generator()
    x_generator.manual_seed(1234)
    x = torch.rand(2, 10, generator=x_generator)
    res = test_model.forward(x)
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
        },
        "algorithm": "smoothquant",
    }
    modelopt_model = mtq.quantize(copy.deepcopy(test_model), CFG, forward_loop)
    mtq.print_quant_summary(modelopt_model)
    ######################################
    config = {
        "default": {
            "enable": True,
            "dtype": DType.INT8SMOOTH,
            "seglinear": True,
            'search_patterns': SegPattern.all(),
            "input_axis": None,
            "weight_axis": None,
            "alpha": 1.0,
        },
    }
    segquant_model = quantize(copy.deepcopy(test_model), seg_data_loader, config, True, example=torch.rand(2, 10))
    ######################################
    x_generator = torch.Generator()
    x_generator.manual_seed(1234)
    x = torch.rand(2, 10, generator=x_generator)
    res = test_model.forward(x)
    print('origin:', res)
    res = modelopt_model.forward(x)
    print('modelopt:', res)
    res = segquant_model.forward(x)
    print('segquant:', res)

if __name__ == '__main__':
    test_smooth_int8()