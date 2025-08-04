from typing import Tuple
import torch
import torch.nn as nn
import copy
import modelopt.torch.quantization as mtq


in_channels = 4
out_channels = 16
height = 8
width = 8

class RandomTensorDataset:
    def __init__(self, num_batches=6, height=256, width=256, seed=42):
        self.num_batches = num_batches
        self.height = height
        self.width = width
        self.seed = seed
        self._generate_batches()

    def _generate_batches(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        self.batches = [
            (
                torch.rand(2, in_channels, self.height, self.width, generator=generator),
            )
            for _ in range(self.num_batches)
        ]

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return self.batches[idx]


dataset = RandomTensorDataset(num_batches=5, height=height, width=width, seed=42)


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
    def __init__(self, in_channels: int = 3, out_channels: int = 16, bias=True):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            bias=bias,
        )

    def forward(
        self, x: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv(x)

test_model = TestModel(in_channels=in_channels, out_channels=out_channels)

def test_conv_fp8():
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
    
    ######################################
    x_generator = torch.Generator()
    x_generator.manual_seed(1234)
    x = torch.rand(2, in_channels, height, width, generator=x_generator)
    res = test_model.forward(x)
    a = res[0]
    res = modelopt_model.forward(x)
    b = res[0]

    print('diff1', torch.norm(a - b).item())

def test_conv_int8():
    ######################################
    CFG = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 8, "axis": 1,},
            "*input_quantizer": {"num_bits": 8, "axis": None,},
        },
        "algorithm": "max",
    }
    modelopt_model = mtq.quantize(copy.deepcopy(test_model), CFG, forward_loop)
    mtq.print_quant_summary(modelopt_model)
    ######################################
    print(torch.amax(test_model.conv.weight.data, dim=(1, 2, 3)))
    
    ######################################
    x_generator = torch.Generator()
    x_generator.manual_seed(1234)
    x = torch.rand(2, in_channels, height, width, generator=x_generator)
    res = test_model.forward(x)
    a = res[0]
    res = modelopt_model.forward(x)
    b = res[0]

    print('diff1', torch.norm(a - b).item())


if __name__ == "__main__":
    # print("Testing FP8 Convolution")
    # test_conv_fp8()
    # print("----------------------")

    print("Testing INT8 Convolution")
    test_conv_int8()
