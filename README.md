# SegQuant
A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models

## Environment
Our project has been tested with Python 3.10 (specifically version 3.10.12) and CUDA 12.5. We highly recommend using a virtual environment, such as Anaconda3, to manage and install the required dependencies.

## Install
Before installation, make sure all required Python dependencies are available. You can install them using:
```bash
pip install -r requirements.txt
```
Then, install the `segquant` package using editable mode (recommended for development):
```bash
pip install -e .
```
This installs the package in-place, so changes to the source code will be reflected immediately without reinstallation.

Alternatively, you can build and install the package as a standard Python package:
```bash
python -m build
pip install dist/segquant-*.whl
```
> Note: This project is organized using pyproject.toml, and requires Python ≥ 3.10. You should also ensure build tools such as setuptools, wheel, and build are installed.

### Setting `CUTLASS_PATH` Environment Variable

This project may depend on the **CUTLASS** library. You'll need to set the `CUTLASS_PATH` environment variable to point to its installation directory. If it's not set, the project will default to `/usr/local/cutlass`.

You can set this temporarily in your current terminal session:

```bash
export CUTLASS_PATH=/path/to/your/cutlass
```

## Usage
### Quantization

To quantize a diffusion model, follow these steps:

1. **Generate a calibration dataset**  
Use `generate_calibrate_set` to sample data from the model. This dataset will be used to calibrate and optimize the quantized model.
```python
generate_calibrate_set(
    model,               # The original model to sample from
    sampler,             # A data sampler (e.g., NormalSampler, UniformSampler)
    sample_dataloader,   # Dataloader for the calibration images or text prompts
    calib_layer,         # The target layer to extract calibration features
    dump_path="calib_data", # Optional: where to store calibration data
)
```
This function supports chunked saving and optional compression, making it scalable for large datasets. If no `dump_path` is specified, the data is returned in memory.

2. **Quantize the model**  
Once calibration data is prepared, call `quantize()`:
```python
quantized_model = quantize(
    model,                  # The original full-precision model
    calib_data_loader,      # The loader of calibration features
    config=quant_config,    # Optional config dict for quantization parameters
    verbose=True,           # Print debug info if needed
    example=input_sample    # Optional example input for operator tracing
)
```
The result is a quantized model that can be evaluated or deployed.

### Affiner for Diffusion Model
To further improve the quality of quantized diffusion models, an affiner can be trained to compensate for errors introduced by quantization.

Use `process_affiner()` as follows:
```python
affiner = process_affiner(
    config=affiner_cfg,       # Dict containing optimizer & solver settings
    dataset=calib_dataset,    # Dataset used to compute affine corrections
    model_real=fp_model,      # Ground-truth full-precision model
    model_quant=quant_model,  # Already quantized model
    latents=optional_latents, # Optional: precomputed latents
    shuffle=True              # Whether to shuffle training data
)
```
This step is optional but highly recommended for high-fidelity tasks such as image generation. You can also plug in a third-party affiner module via the `thirdparty_affiner` argument.

Note: In our implementation of the diffusion model's `forward` function, we support a `stepper` argument. You can pass the trained `affiner` to this argument to seamlessly perform error reconstruction during the generation process.

For detailed usage and parameter descriptions of these APIs, please refer to the [full documentation](segquant/torch/README.md).

## Features
### Automated Pattern Searching
During quantization, our framework constructs semantic structures from the model to automatically select appropriate quantization configurations for linear layers. These include patterns suitable for techniques like `Chunk-Linear` in segmentation and `Activation` structures for `DualScale` quantization. We also support graph-based semantic pattern detection, enabling integration with other optimization strategies.

### Easy Integration with Other Frameworks
We provide CUDA kernels that implement key optimization strategies. These kernels are designed to be easily reused in other quantization or model inference frameworks. For integration examples, please refer to [this](segquant/src/README.md).

## Contributing
### Model
In the `backend` directory, we use two popular text-to-image diffusion models ([Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)<sup>[1]</sup> and [FLUX-1.0-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)<sup>[2]</sup>) as examples. These implementations are adapted from the [Diffusers library](https://github.com/huggingface/diffusers). Our framework supports any PyTorch-based models, and the quantization and optimization are not limited to diffusion models.

### Dataset
In the `dataset` directory, we provide several commonly used detection datasets, including MS-COCO<sup>[3]</sup>, Densely Captioned Images<sup>[4]</sup> and MJHQ-30K<sup>[5]</sup> in the diffusion model domain. To support quantization testing with ControlNet<sup>[6]</sup>, we have created preprocessing scripts that generate ControlNet (Canny<sup>[7]</sup>) input images based on these open-source datasets. These tools make it easy to convert and adapt datasets for various experimental needs.

For detailed instructions on dataset usage, please refer to [this](dataset/README.md).

### Quantization Implement and CUDA Kernel
To align with mainstream quantization frameworks such as [ModelOPT](https://github.com/NVIDIA/TensorRT-Model-Optimizer), we reference parts of their quantization implementation—particularly components related to FP8 quantization.

On the kernel side, we build upon [CUTLASS](https://github.com/NVIDIA/cutlass) for custom CUDA kernel development. Our framework also integrates unique features such as `Seglinear` and `DualScale` quantization, offering improved performance and flexibility.

In terms of quantization optimization algorithms, we draw inspiration from SmoothQuant<sup>[8]</sup> (INT8-based) and SVDQuant<sup>[9]</sup> (INT4-based) implementations. These references help demonstrate the orthogonality of our quantization approach with respect to mainstream methods, showcasing its general applicability and compatibility.

### Other Methods for Diffusion Models
For experimental purposes, we have also implemented several related optimization methods from recent papers, including PTQ4DM<sup>[10]</sup>, Q-Diffusion<sup>[11]</sup>, PTQD<sup>[12]</sup>, and TAC-Diffusion<sup>[13]</sup>.

## References
[1] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. 2024. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the 41st International Conference on Machine Learning (ICML'24), Vol. 235. JMLR.org, Article 503, 12606–12633.

[2] Black Forest Labs. 2024. Flux.1. Retrieved May 5, 2025 from https://blackforestlabs.ai/

[3] Lin, TY. et al. (2014). Microsoft COCO: Common Objects in Context. In: Fleet, D., Pajdla, T., Schiele, B., Tuytelaars, T. (eds) Computer Vision – ECCV 2014. ECCV 2014. Lecture Notes in Computer Science, vol 8693. Springer, Cham. https://doi.org/10.1007/978-3-319-10602-1_48

[4] J. Urbanek et al., "A Picture is Worth More Than 77 Text Tokens: Evaluating CLIP-Style Models on Dense Captions," in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 2024, pp. 26690-26699, doi: 10.1109/CVPR52733.2024.02521.

[5] Li, Daiqing, et al. "Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation." arXiv preprint arXiv:2402.17245 (2024).

[6] L. Zhang, A. Rao and M. Agrawala, "Adding Conditional Control to Text-to-Image Diffusion Models," 2023 IEEE/CVF International Conference on Computer Vision (ICCV), Paris, France, 2023, pp. 3813-3824, doi: 10.1109/ICCV51070.2023.00355.

[7] J. Canny, "A Computational Approach to Edge Detection," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI-8, no. 6, pp. 679-698, Nov. 1986, doi: 10.1109/TPAMI.1986.4767851.

[8] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. 2023. SmoothQuant: accurate and efficient post-training quantization for large language models. In Proceedings of the 40th International Conference on Machine Learning (ICML'23), Vol. 202. JMLR.org, Article 1585, 38087–38099.

[9] Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. 2025. SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models. In Proceedings of the Thirteenth International Conference on Learning Representations (ICLR 2025).

[10] Y. Shang, Z. Yuan, B. Xie, B. Wu and Y. Yan, "Post-Training Quantization on Diffusion Models," 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, BC, Canada, 2023, pp. 1972-1981, doi: 10.1109/CVPR52729.2023.00196.

[11] X. Li et al., "Q-Diffusion: Quantizing Diffusion Models," 2023 IEEE/CVF International Conference on Computer Vision (ICCV), Paris, France, 2023, pp. 17489-17499, doi: 10.1109/ICCV51070.2023.01608.

[12] Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. 2023. PTQD: accurate post-training quantization for diffusion models. In Proceedings of the 37th International Conference on Neural Information Processing Systems (NIPS '23). Curran Associates Inc., Red Hook, NY, USA, Article 580, 13237–13249.

[13] Yuzhe Yao, Feng Tian, Jun Chen, Haonan Lin, Guang Dai, Yong Liu, and Jingdong Wang. 2024. Timestep-Aware Correction for Quantized Diffusion Models. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part LXVI. Springer-Verlag, Berlin, Heidelberg, 215–232. https://doi.org/10.1007/978-3-031-72848-8_13