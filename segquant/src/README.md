# Kernels
## Supported Quantization Type
| Activation Type | Weight Type | Real Quant | Architecture |
|:---------------:|:-----------:|:----------:|:------------:|
| INT8            | INT8        | ✔          |     SM80     |
| INT8            | INT4        | ✔          |     SM80     |
| INT4            | INT4        | ✔          |     SM80     |
| FP8E4M3         | FP8E4M3     | ✔          |     SM89     |
| FP16            | INT8        | ✔          |     SM80     |
| FP16            | INT4        | ❌         |     SM80     |

## Supported Quantization Config
| Config | Supported |
|:---------------|:-----------:|
| Dynamic(Weight)     | ✔          |
| Per-Tensor          | ✔          |
| Per-Token(Act.)     | ✔          |
| Per-Channel(Weight) | ✔          |
| Group-Wise(Weight) | ❌          |

## Supported Quantization Algorithm
| Algorithm | Supported |
|:---------------|:-----------:|
| AdaRound              | ❌          |
| BRECQ                 | ❌          |
| GPTQ          | ✔          |
| SmoothQuant            | ✔          |
| AWQ                    | ❌          |
| SVDQuant               | ✔          |