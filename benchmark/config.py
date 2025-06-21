from segquant.config import DType, Optimum, SegPattern, Calibrate
from enum import Enum
import torch
import os

# Available choices for CLI arguments
QUANT_METHOD_CHOICES = ["int8smooth", "int8smooth_seg", "int8smooth_dual", 
                        "int8smooth_affine", "int8smooth_seg_dual", "int8smooth_seg_dual_affine", 
                        "fp8", "fp8_seg", "fp8_dual", "fp8_affine", 
                        "fp8_seg_dual", "fp8_seg_dual_affine"]

DATASET_TYPE_CHOICES = ["COCO", "MJHQ", "DCI"]

MODEL_TYPE_CHOICES = ["sd3", "flux", "sdxl"]

LAYER_TYPE_CHOICES = {
    "sd3": ["dit", "controlnet"],
    "flux": ["dit", "controlnet"], 
    "sdxl": ["unet"]
}

# Supported weight/activation bit configurations
QUANT_CONFIG_CHOICES = ["W4A4", "W4A8", "W8A4", "W8A8", "W4A16", "W8A16"]

class CalibrationConfig:
    MAX_TIMESTEP = 50
    SAMPLE_SIZE = 1
    TIMESTEP_PER_SAMPLE = 50
    CONTROLNET_CONDITIONING_SCALE = 0
    GUIDANCE_SCALE = 7
    SHUFFLE = False

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

class ModelConfigs:
    """Model-specific quantization configurations"""
    
    @staticmethod
    def parse_quant_config(quant_config_str):
        """Parse weight/activation bit configuration string like 'W4A4'"""
        if not quant_config_str:
            return None, None
        
        import re
        match = re.match(r'W(\d+)A(\d+)', quant_config_str)
        if not match:
            raise ValueError(f"Invalid quant config format: {quant_config_str}. Expected format: W<bits>A<bits> (e.g., W4A4)")
        
        weight_bits = int(match.group(1))
        activation_bits = int(match.group(2))
        return weight_bits, activation_bits
    
    @staticmethod
    def get_dtype_from_bits(bits):
        """Convert bit width to corresponding DType"""
        if bits == 4:
            return DType.INT4
        elif bits == 8:
            return DType.INT8
        elif bits == 16:
            return DType.FP16
        else:
            raise ValueError(f"Unsupported bit width: {bits}")
    
    @staticmethod
    def create_quant_config(model_type, quant_config_str=None):
        """Create quantization config based on model type and bit configuration"""
        # Get base config for the model type
        if model_type == "flux":
            base_config = ModelConfigs.FLUX_BASE_CONFIG.copy()
        elif model_type == "sd3":
            base_config = ModelConfigs.SD3_BASE_CONFIG.copy()
        elif model_type == "sdxl":
            base_config = ModelConfigs.SDXL_BASE_CONFIG.copy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # If no quant config specified, return baseline (original config)
        if not quant_config_str:
            return {"default": base_config}
        
        # Parse bit configuration
        weight_bits, activation_bits = ModelConfigs.parse_quant_config(quant_config_str)
        
        # Update quantization types
        base_config["input_quant"]["type"] = ModelConfigs.get_dtype_from_bits(activation_bits)
        base_config["weight_quant"]["type"] = ModelConfigs.get_dtype_from_bits(weight_bits)
        
        # For 4-bit weights, enable real quantization
        if weight_bits == 4:
            base_config["real_quant"] = True
        
        return {"default": base_config}
    
    # FLUX configurations
    FLUX_BASE_CONFIG = {
        "enable": True,
        "seglinear": True,
        "search_patterns": [],
        "real_quant": True,
        "opt": {
            "type": Optimum.SVD,
            "alpha": 0.5,
            "low_rank": 64,
            "search_alpha_config": {
                "enable": False,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
            },
            "verbose": True,
        },
        "calib": {
            "type": Calibrate.GPTQ,
            "cpu_storage": False,
            "verbose": False,
        },
        "input_quant": {
            "type": DType.INT4,
            "axis": -1,
            "dynamic": True,
        },
        "weight_quant": {
            "type": DType.INT4,
            "axis": 1,
        },
    }

    # SD3 configurations  
    SD3_BASE_CONFIG = {
        "enable": True,
        "seglinear": True,
        "search_patterns": [],
        "real_quant": False,
        "opt": {
            "type": Optimum.SVD,
            "alpha": 0.5,
            "low_rank": 64,
            "search_alpha_config": {
                "enable": True,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
            },
            "verbose": True,
        },
        "calib": {
            "type": Calibrate.GPTQ,
            "cpu_storage": False,
            "verbose": False,
        },
        "input_quant": {
            "type": DType.INT8,
            "axis": -1,
            "dynamic": True,
        },
        "weight_quant": {
            "type": DType.INT4,
            "axis": 1,
        },
    }

    # SDXL configurations
    SDXL_BASE_CONFIG = {
        "enable": True,
        "seglinear": True,
        "search_patterns": [],
        "real_quant": False,
        "opt": {
            "type": Optimum.SVD,
            "alpha": 0.5,
            "low_rank": 64,
            "search_alpha_config": {
                "enable": True,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
            },
            "verbose": True,
        },
        "calib": {
            "type": Calibrate.GPTQ,
            "cpu_storage": False,
            "verbose": False,
        },
        "input_quant": {
            "type": DType.FP16,
            "axis": None,
        },
        "weight_quant": {
            "type": DType.INT4,
            "axis": 1,
        },
    }

    MODEL_CONFIGS = {
        "flux": {"default": FLUX_BASE_CONFIG},
        "sd3": {"default": SD3_BASE_CONFIG},
        "sdxl": {"default": SDXL_BASE_CONFIG}
    }

    @classmethod
    def get_config(cls, model_type):
        if model_type not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(cls.MODEL_CONFIGS.keys())}")
        return cls.MODEL_CONFIGS[model_type]

class QuantizationConfigs:
    """Legacy quantization configs for SD3 experiments"""
    INT8SMOOTH_BASE_CONFIG = {
        "enable": True,
        "input_dtype": DType.INT8,
        "weight_dtype": DType.INT8,
        "opt": Optimum.SMOOTH,
        "real_quant": False,
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
        "real_quant": False,
        "input_axis": None,
        "weight_axis": None,
        "alpha": 0.5,
        "dit": True,
        "controlnet": False,
    }

    INT8SMOOTH_DEFAULT_CONFIG = {"default": {**INT8SMOOTH_BASE_CONFIG, "seglinear": False, "search_patterns": [],"affine": False}}
    INT8SMOOTH_AFFINE_CONFIG = {"default": {**INT8SMOOTH_BASE_CONFIG, "seglinear": False, "search_patterns": [],"affine": True}}
    INT8SMOOTH_SEG_CONFIG = {"default": {**INT8SMOOTH_BASE_CONFIG, "seglinear": True, "search_patterns": SegPattern.seg(),"affine": False}}
    INT8SMOOTH_DUAL_CONFIG = {"default": {**INT8SMOOTH_BASE_CONFIG, "seglinear": True, "search_patterns": [SegPattern.ACTIVATION2LINEAR],"affine": False}}
    INT8SMOOTH_SEG_DUAL_CONFIG = {"default": {**INT8SMOOTH_BASE_CONFIG, "seglinear": True, "search_patterns": SegPattern.all(),"affine": False}}
    INT8SMOOTH_SEG_DUAL_AFFINE_CONFIG = {"default": {**INT8SMOOTH_BASE_CONFIG, "seglinear": True, "search_patterns": SegPattern.all(),"affine": True}}

    FP8_DEFAULT_CONFIG = {"default": {**FP8_BASE_CONFIG, "seglinear": False, "search_patterns": [],"affine": False}}
    FP8_AFFINE_CONFIG = {"default": {**FP8_BASE_CONFIG, "seglinear": False, "search_patterns": [],"affine": True}}
    FP8_SEG_CONFIG = {"default": {**FP8_BASE_CONFIG, "seglinear": True, "search_patterns": SegPattern.seg(),"affine": False}}
    FP8_DUAL_CONFIG = {"default": {**FP8_BASE_CONFIG, "seglinear": True, "search_patterns": [SegPattern.ACTIVATION2LINEAR],"affine": False}}
    FP8_SEG_DUAL_CONFIG = {"default": {**FP8_BASE_CONFIG, "seglinear": True, "search_patterns": SegPattern.all(),"affine": False}}
    FP8_SEG_DUAL_AFFINE_CONFIG = {"default": {**FP8_BASE_CONFIG, "seglinear": True, "search_patterns": SegPattern.all(),"affine": True}}

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
        "sample_size": 256,
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

class DatasetConfig:
    """Simple dataset configuration"""
    def __init__(self, name, path, cache_size=16, max_timestep=50):
        self.name = name
        self.path = path
        self.cache_size = cache_size
        self.max_timestep = max_timestep

class BenchmarkConfig:
    """Unified benchmark configuration for all model types"""
    
    def __init__(self, 
                 model_type="sd3", 
                 layer_type="dit", 
                 dataset_type="COCO", 
                 num_images=5000, 
                 generate_real_pics=False, 
                 enable_affine=False, 
                 per_layer_mode=False, 
                 quant_config=None,
                 reprocess=False):
        # Validate inputs
        if model_type not in MODEL_TYPE_CHOICES:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        if layer_type not in LAYER_TYPE_CHOICES[model_type]:
            raise ValueError(f"Invalid layer_type for {model_type}: {layer_type}")
        
        if dataset_type not in DATASET_TYPE_CHOICES:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")
        
        if quant_config and quant_config not in QUANT_CONFIG_CHOICES:
            raise ValueError(f"Invalid quant_config: {quant_config}. Choices: {QUANT_CONFIG_CHOICES}")
        
        self.model_type = model_type
        self.layer_type = layer_type
        self.dataset_type = dataset_type
        self.num_images = num_images
        self.generate_real_pics = generate_real_pics
        self.enable_affine = enable_affine
        self.per_layer_mode = per_layer_mode
        self.quant_config = quant_config
        self.reprocess = reprocess
        
        # Legacy properties for compatibility
        self.benchmark_size = num_images
        self.max_timestep = CalibrationConfig.MAX_TIMESTEP
        self.gpu_id = 0
        
        # Generate paths
        # self.result_dir = self._get_result_dir()
        self.dataset = self._get_dataset()
        self.model_quant_config = self._get_model_quant_config()
        self.latents = self._get_latents()
        
        # Legacy property for compatibility
        # self.res_dir = self.result_dir
        
    def _get_result_dir(self):
        """Generate result directory path with quantization info"""
        base_path = f"../segquant/benchmark_record/main_expr/{self.model_type}_{self.layer_type}"
        
        if self.quant_config:
            # Add quantization config to path for clarity
            return f"{base_path}_{self.quant_config}"
        else:
            # Baseline test
            return f"{base_path}_baseline"
    
    def _get_dataset(self):
        """Get dataset configuration"""
        dataset_paths = {
            "COCO": "../dataset/controlnet_datasets/COCO-Caption2017-canny",
            "MJHQ": "../dataset/controlnet_datasets/MJHQ-30K-canny", 
            "DCI": "../dataset/controlnet_datasets/DCI-30K-canny",
        }
        
        dataset_path = dataset_paths[self.dataset_type]
        
        if self.dataset_type == "COCO":
            from dataset.coco.coco_dataset import COCODataset
            dataset = COCODataset(dataset_path, cache_size=16)
        elif self.dataset_type == "MJHQ":
            from dataset.mjhq.mjhq_dataset import MJHQDataset
            dataset = MJHQDataset(dataset_path, cache_size=16)
        elif self.dataset_type == "DCI":
            from dataset.dci.dci_dataset import DCIDataset
            dataset = DCIDataset(dataset_path, cache_size=16)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        return dataset
        
    def _get_model_quant_config(self):
        """Get model quantization configuration"""
        return ModelConfigs.create_quant_config(self.model_type, self.quant_config)
    
    def _get_latents(self):
        """Generate appropriate latents for each model type"""
        import torch
        from backend.torch.utils import randn_tensor
        
        # check if latent.pt exitst
        latent_path = f"../segquant/benchmark_record/main_expr/{self.model_type}/latent.pt"
        if os.path.exists(latent_path):
            print(f"[INFO] Latent found at {latent_path}, loading latent...")
            return torch.load(latent_path)
        
        # generate latent
        if self.model_type == 'flux':
            latents = randn_tensor(
                (1, 4096, 64,), device=torch.device(f"cuda:{self.gpu_id}"), dtype=torch.float16
            )
        elif self.model_type == 'sd3':
            latents = randn_tensor(
                (1, 16, 128, 128,), device=torch.device(f"cuda:{self.gpu_id}"), dtype=torch.float16
            )
        elif self.model_type == 'sdxl':
            latents = randn_tensor(
                (1, 4, 128, 128,), device=torch.device(f"cuda:{self.gpu_id}"), dtype=torch.float16
            )
        else:
            raise ValueError(f'Unknown model type: {self.model_type}')
        
        # save latent
        os.makedirs(os.path.dirname(latent_path), exist_ok=True)
        torch.save(latents, latent_path)
        print(f"[INFO] Latent stored at {latent_path}")

        return latents
    
    def get_model_quant_path(self):
        """Generate model quantization save path with bit configuration"""
        base_path = f"../segquant/benchmark_record/main_expr/{self.model_type}/model/{self.layer_type}"
        
        if self.quant_config:
            return f"{base_path}/{self.quant_config}"
        else:
            return f"{base_path}/baseline"
    
    def get_pic_store_path(self):
        """Generate picture storage path with quantization info"""
        base_path = f"../segquant/benchmark_record/main_expr/{self.model_type}/pics/{self.layer_type}"
        
        if self.quant_config:
            return f"{base_path}/{self.quant_config}"
        else:
            return f"{base_path}/baseline"

    def __str__(self):
        """String representation of the configuration"""
        return (f"BenchmarkConfig(\n"
                f"  model_type={self.model_type}\n"
                f"  layer_type={self.layer_type}\n"
                f"  dataset_type={self.dataset_type}\n"
                f"  quant_config={self.quant_config or 'baseline'}\n" 
                f"  num_images={self.num_images}\n"
                f"  generate_real_pics={self.generate_real_pics}\n"
                f"  enable_affine={self.enable_affine}\n"
                f"  per_layer_mode={self.per_layer_mode}\n"
                # f"  result_dir={self.result_dir}\n"
                f"  pic_store_path={self.get_pic_store_path()}\n"
                f"  model_quant_path={self.get_model_quant_path()}\n"
                f")")