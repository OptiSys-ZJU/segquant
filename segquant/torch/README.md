# API Reference

This module provides essential APIs for model quantization, calibration dataset generation, and affiner loading.

---

## quantize

```python
quantize(
    model: nn.Module,
    calib_data_loader: torch.utils.data.DataLoader,
    config: dict = None,
    per_layer_mode: bool = False,
    verbose: bool = False,
    example: Any = None,
) -> nn.Module
```

Quantizes the given PyTorch model using calibration data and optional configuration.

**Arguments**:
- `model` (`nn.Module`): The PyTorch model to quantize.
- `calib_data_loader` (`DataLoader`): DataLoader used for calibration.
- `config` (`dict`, optional): Quantization configuration dictionary.
- `per_layer_mode` (`bool`, optional): Enable per-layer quantization mode.
- `verbose` (`bool`, optional): If `True`, print detailed logs.
- `example` (`Any`, optional): Example input for pattern detection.

**Returns**:
- The quantized model (`nn.Module`).

---

## calibrate

```python
generate_calibrate_set(
    model: nn.Module,
    sampler: Sampler,
    sample_dataloader: torch.utils.data.DataLoader,
    calib_layer: nn.Module,
    dump_path: str = None,
    chunk_size: int = 400,
    compress: bool = True,
    **kwargs
) -> BaseCalibSet
```

Generates a calibration dataset by sampling activations from the model and saves it in chunks.

**Arguments**:
- `model` (`nn.Module`): The model to sample from.
- `sampler` (`Sampler`): Sampler used to extract data.
- `sample_dataloader` (`DataLoader`): Dataloader providing sampling inputs.
- `calib_layer` (`nn.Module`): The layer from which to collect calibration activations.
- `dump_path` (`str`, optional): Directory to store the calibration set. If `None`, data is returned in memory.
- `chunk_size` (`int`, optional): Number of samples per chunk (default: 400).
- `compress` (`bool`, optional): Whether to compress the stored chunks (default: `True`).
- `**kwargs`: Additional arguments passed to the model's forward method.

**Returns**:
- `BaseCalibSet`: A wrapper containing calibration data.

---

```python
load_calibrate_set(
    dump_path: str,
    compress: bool = True,
) -> Optional[BaseCalibSet]
```

Loads a previously saved calibration dataset from disk.

**Arguments**:
- `dump_path` (`str`): Path to the saved calibration folder.
- `compress` (`bool`, optional): Whether the dataset was saved in compressed format.

**Returns**:
- `BaseCalibSet` if found, otherwise `None`.

---

## affiner

```python
load_affiner(
    config: dict,
    dataset: Dataset,
    model_real: nn.Module,
    model_quant: nn.Module,
    latents: Any = None,
    shuffle: bool = True,
    thirdparty_affiner: Any = None,
) -> Any
```

Initializes or loads an affiner to adjust the quantized model based on training dynamics.

**Arguments**:
- `config` (`dict`): Configuration dictionary (stepper, solver, etc.).
- `dataset` (`Dataset`): Training dataset for affiner.
- `model_real` (`nn.Module`): Original full-precision model.
- `model_quant` (`nn.Module`): Quantized model.
- `latents` (`Any`, optional): Latent representations used for alignment.
- `shuffle` (`bool`, optional): Whether to shuffle dataset indices.
- `thirdparty_affiner` (`Any`, optional): External affiner instance.

**Returns**:
- Trained affiner instance (`BlockwiseAffiner` or custom).



# Quantization Configuration

The `quantize` function accepts a `config` dictionary to control how quantization is performed. Below is a breakdown of the structure and available options.

## Example

```python
default_quantize_config = {
    "default": {
        "enable": True,
        "seglinear": True,
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
```

## Configuration Fields

- `enable` (`bool`)  
  Whether to enable quantization for this block or layer.

- `seglinear` (`bool`)  
  Whether to allow segmented transformation of linear layers for finer-grained quantization.

- `search_patterns` (`List[SegPattern]`)  
  A list of transformation patterns used to identify segments in the computation graph. See [SegPattern](#segpattern-values) below.

- `real_quant` (`bool`)  
  If `True`, enables actual quantization execution; otherwise, performs fake-quantization.

- `opt` (`dict`)  
  Optimization configuration:
  - `type` (`Optimum`) — Quantization optimization strategy.
  - `alpha` (`float`, optional) — Used by some optimizers such as `SMOOTH`.
  - `search_alpha_config` (`dict`, optional) - `alpha` Searing Config
    - `enable` (`bool`) - 
    - `min` (`float`) - Min `alpha`
    - `max` (`float`) - Max `alpha`
    - `step` (`float`) - Step `alpha` value

- `calib` (`dict`)  
  Calibration method for determining quantization parameters:
  - `type` (`Calibrate`) — Calibration method (e.g., `AMAX`, `GPTQ`).

- `input_quant` / `weight_quant` (`dict`)  
  Quantization settings for input and weights:
  - `type` (`DType`) — Data type used in quantization (see below).
  - `axis` (`int` or `None`) — Channel-wise quantization axis; `None` means per-tensor.

---

### `DType` Values

Supported quantization data types:

- `INT4` – 4-bit integer
- `INT6` – 6-bit integer
- `INT8` – 8-bit integer
- `INT16` – 16-bit integer
- `FP8E4M3` – 8-bit float (E4M3)
- `FP8E5M2` – 8-bit float (E5M2)
- `FP16` – 16-bit float

---

### `Optimum` Values

Quantization optimization strategies:

- `DEFAULT` – No extra optimization
- `SMOOTH` – SmoothQuant (`alpha`)
- `SVD` – SVDQuant (`low_rank`)
- `SMOOTH` and `SVD` support `search_alpha_config`
---

### `Calibrate` Values

Calibration algorithms for scale computation:

- `AMAX` – Use absolute max of activation or weight
- `GPTQ` – Use GPTQ

---

### `SegPattern` Values

Graph detecting patterns used for layer segmentation:

- `LINEAR2CHUNK` – Split large linear into chunked segments
- `LINEAR2SPLIT` – Split linear input into sub-parts
- `CONCAT2LINEAR` – Merge concat outputs into linear
- `STACK2LINEAR` – Merge stacked inputs into linear
- `ACTIVATION2LINEAR` – Insert activation layer before linear

Use `SegPattern.seg()` to get segment-only patterns, or `SegPattern.all()` to use all available patterns.

