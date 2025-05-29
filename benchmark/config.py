from segquant.config import DType, Optimum, SegPattern
from enum import Enum
import torch


class CalibrationConfig:
    MAX_TIMESTEP = 50
    SAMPLE_SIZE = 256
    TIMESTEP_PER_SAMPLE = 50
    CONTROLNET_CONDITIONING_SCALE = 0
    GUIDANCE_SCALE = 7
    SHUFFLE = True

    @classmethod
    def to_dict(cls):
        return {
            "max_timestep": cls.MAX_TIMESTEP,
            "sample_size": cls.SAMPLE_SIZE,
            "timestep_per_sample": cls.TIMESTEP_PER_SAMPLE,
            "controlnet_conditioning_scale": cls.CONTROLNET_CONDITIONING_SCALE,
            "guidance_scale": cls.GUIDANCE_SCALE,
            "shuffle": cls.SHUFFLE,
        }

class QuantizationConfigs:
    INT8SMOOTH_BASE_CONFIG = {
        "enable": True,
        "input_dtype": DType.INT8,
        "weight_dtype": DType.INT8,
        "opt": Optimum.SMOOTH,
        "input_axis": None,
        "weight_axis": None,
        "alpha": 0.5,
        "dit": True,
        "controlnet": False,
    }

    FP8_BASE_CONFIG = {
        "enable": True,
        "input_dtype": DType.FP8E4M3,
        "weight_dtype": DType.FP8E5M2,
        "opt": Optimum.DEFAULT,
        "input_axis": None,
        "weight_axis": None,
        "alpha": 0.5,
        "dit": True,
        "controlnet": False,
    }

    INT8SMOOTH_DEFAULT_CONFIG = {"default": {**INT8SMOOTH_BASE_CONFIG, "seglinear": False, "search_patterns": []}, "affine": False}

    INT8SMOOTH_AFFINE_CONFIG = {"default": {**INT8SMOOTH_BASE_CONFIG, "seglinear": False, "search_patterns": []}, "affine": True}

    INT8SMOOTH_SEG_CONFIG = {"default": {**INT8SMOOTH_BASE_CONFIG, "seglinear": True, "search_patterns": SegPattern.seg()}, "affine": False}
    
    INT8SMOOTH_DUAL_CONFIG = {"default": {**INT8SMOOTH_BASE_CONFIG, "seglinear": True, "search_patterns": [SegPattern.ACTIVATION2LINEAR]}, "affine": False}

    INT8SMOOTH_SEG_DUAL_CONFIG = {"default": {**INT8SMOOTH_BASE_CONFIG, "seglinear": True, "search_patterns": SegPattern.all()}, "affine": False}

    INT8SMOOTH_SEG_DUAL_AFFINE_CONFIG = {"default": {**INT8SMOOTH_BASE_CONFIG, "seglinear": True, "search_patterns": SegPattern.all()}, "affine": True}

    FP8_DEFAULT_CONFIG = {"default": {**FP8_BASE_CONFIG, "seglinear": False, "search_patterns": []}, "affine": False}
    
    FP8_AFFINE_CONFIG = {"default": {**FP8_BASE_CONFIG, "seglinear": False, "search_patterns": []}, "affine": True}
    
    FP8_SEG_CONFIG = {"default": {**FP8_BASE_CONFIG, "seglinear": True, "search_patterns": SegPattern.seg()}, "affine": False}
    
    FP8_DUAL_CONFIG = {"default": {**FP8_BASE_CONFIG, "seglinear": True, "search_patterns": [SegPattern.ACTIVATION2LINEAR]}, "affine": False}
    
    FP8_SEG_DUAL_CONFIG = {"default": {**FP8_BASE_CONFIG, "seglinear": True, "search_patterns": SegPattern.all()}, "affine": False}

    FP8_SEG_DUAL_AFFINE_CONFIG = {"default": {**FP8_BASE_CONFIG, "seglinear": True, "search_patterns": SegPattern.all()}, "affine": True}

    METHOD_CONFIG = {
        "int8smooth": INT8SMOOTH_DEFAULT_CONFIG,
        "int8smooth_seg": INT8SMOOTH_SEG_CONFIG,
        "int8smooth_dual": INT8SMOOTH_DUAL_CONFIG,
        "int8smooth_affine": INT8SMOOTH_AFFINE_CONFIG,
        "int8smooth_seg_dual": INT8SMOOTH_SEG_DUAL_CONFIG,
        "int8smooth_seg_dual_affine": INT8SMOOTH_SEG_DUAL_AFFINE_CONFIG,
        "fp8": FP8_DEFAULT_CONFIG,
        "fp8_seg": FP8_SEG_CONFIG,
        "fp8_dual": FP8_DUAL_CONFIG,
        "fp8_affine": FP8_AFFINE_CONFIG,
        "fp8_seg_dual": FP8_SEG_DUAL_CONFIG,
        "fp8_seg_dual_affine": FP8_SEG_DUAL_AFFINE_CONFIG,
    }

    @classmethod
    def get_config(cls, quant_method):
        if quant_method not in cls.METHOD_CONFIG:
            raise ValueError(f"Unknown quantization method: {quant_method}. Available methods: {list(cls.METHOD_CONFIG.keys())}")
        return cls.METHOD_CONFIG[quant_method]

class AffineConfig:
    SOLVER = {
        "type": "mserel",
        "blocksize": 8,
        "alpha": 0.5,
        "lambda1": 0.1,
        "lambda2": 0.1,
        "sample_mode": "interpolate",
        "percentile": 100,
        "greater": True,
        "scale": 1,
        "verbose": True,
    }
    
    STEPPER = {
        "type": "blockwise",
        "max_timestep": 50,
        "sample_size": 1,
        "recurrent": True,
        "noise_target": "uncond",
        "enable_latent_affine": False,
        "enable_timesteps": None,
    }
    
    EXTRA_ARGS = {
        "controlnet_conditioning_scale": 0,
        "guidance_scale": 7,
    }
    
    @classmethod
    def to_dict(cls):
        return {
            "solver": cls.SOLVER,
            "stepper": cls.STEPPER,
            "extra_args": cls.EXTRA_ARGS,
        }

class BenchmarkConfig:
    DATASET_PATH = {
        "COCO": "../dataset/controlnet_datasets/coco_canny",
        "MJHQ": "../dataset/MJGQ-30K",
        "DCI": "../dataset/densely_captioned_images",
    }
    
    def __init__(self, 
        quant_method, 
        dataset_type, 
        data_cache_size=1024, 
        generate_real_pics=False,
        latents_path="../latents.pt",
        controlnet_conditioning_scale=0,
        guidance_scale=7,
        test_benchmark_size=2,
        benchmark_size=5000,
        max_timestep=50,
        gpu_id=0,
    ):
        self.quant_method = quant_method
        self.dataset_type = dataset_type
        self.dataset_path = self.DATASET_PATH[dataset_type]
        self.data_cache_size = data_cache_size
        self.latents_path = latents_path
        self.generate_real_pics = generate_real_pics
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.guidance_scale = guidance_scale
        self.test_benchmark_size = test_benchmark_size
        self.benchmark_size = benchmark_size
        self.max_timestep = max_timestep
        self.dataset = self.get_dataset()
        self.res_dir = self.get_res_dir()
        self.latents = self.get_latents()
        self.gpu_id = gpu_id
    
    def get_res_dir(self):
        return f"../segquant/benchmark_record/{self.dataset_type}/run_{self.quant_method}_module"

    def get_dataset(self):  
        if self.dataset_type == "COCO":
            from dataset.coco.coco_dataset import COCODataset
            dataset = COCODataset(self.dataset_path, self.data_cache_size)
        elif self.dataset_type == "MJHQ":
            from dataset.mjhq.mjhq_dataset import MJHQDataset
            dataset = MJHQDataset(self.dataset_path, self.data_cache_size)
        elif self.dataset_type == "DCI":
            from dataset.dci.dci_dataset import DCIDataset
            dataset = DCIDataset(self.dataset_path, self.data_cache_size)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        return dataset
        
    def get_latents(self):
        latents = torch.load(self.latents_path)
        return latents