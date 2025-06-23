# ControlNet Dataset Preprocessing Tool

This script processes a dataset (e.g. COCO-Caption2017) to generate ControlNet-compatible training data, such as Canny edge maps or depth maps.

---

## Usage

```bash
python -m {dataset_name}.preprocess [OPTIONS]
```

For example:

```bash
python -m coco.preprocess \
  --output_dir ./controlnet_data \
  --cn_type canny \
  --sample_size 10000 \
  --enable_blur \
  --dataset COCO-Caption2017 \
  --split val
```

---

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | `str` | `../dataset/controlnet_datasets` | Directory where the processed data will be saved. |
| `--cn_type` | `str` | `canny` | Type of control map to generate. Options: `canny`, `depth`. |
| `--sample_size` | `int` | `5000` | Maximum number of samples to process. |
| `--enable_blur` | flag | `False` | Enable Gaussian blur preprocessing for Canny edge detection. |
| `--blur_kernel_size` | `int` | `3` | Kernel size used for Gaussian blur (must be odd). |
| `--dataset` | `str` | `COCO-Caption2017` | Name of the dataset to use. |
| `--split` | `str` | `val` | Dataset split to process (`train`, `val`, etc.). |
| `--enable_no_prompt` | flag | `False` | If set, removes prompts from the output. |
| `--random_sample` | flag | `False` | If set, randomly samples from the dataset instead of sequential order. |

---

## Notes

- **Canny mode** uses OpenCV edge detection; enabling `--enable_blur` can improve edge clarity.
- This tool is often used to generate paired image/control map datasets for ControlNet training or finetuning.

---

## Dependencies

Make sure to install any required packages before running the script:

```bash
pip install opencv-python tqdm
```

---

## Output Structure

The script will generate a directory with the following structure:

```
output_dir/
├── images/
│   ├── 000001.jpg
│   └── ...
├── controls/
│   ├── 000001.png  # e.g., Canny edge or depth map
│   └── ...
└── meta.json       # Optional metadata
```
