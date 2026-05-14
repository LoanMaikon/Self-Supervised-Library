<div align="center">

# Self-supervised Library

This is the repository containing the source code and documentation for my personal project **Self-supervised Library**.

A modular PyTorch-based library for experimenting with and benchmarking self-supervised learning methods on computer vision tasks.

</div>

---

## 1. Overview

The following self-supervised methods are currently available:

- SimCLR
- BYOL
- SwAV
- MAE
- I-JEPA
- iBOT
- DINOv1

This library aims to provide:

- Clean implementations of popular self-supervised methods
- Easy integration with custom datasets
- Support for multiple GPU
- Support for multiple configuration setups

---

## 2. Pretraining

Here you can run any method using a config file found at `configs/pretraining`. For instance, the following command runs SimCLR in a single GPU <cuda:0> and the output is set to `../simclr`

```bash
nohup torchrun --nproc_per_node=1 main.py --config configs/pretraining/pretraining_simclr.yaml --devices cuda:0 --output_folder ../simclr &
```

## 3. Linear Evaluation and Fine-tuning

You can easily run a linear evaluation or fine-tuning using a config file found at `configs/evaluate`. For instance, the following command runs a SimCLR linear evaluation in a single GPU <cuda:0> using the encoder trained previously. The output is set to `../simclr/linear_evaluation`

```bash
nohup torchrun --nproc_per_node=1 main.py --config configs/evaluate/evaluate_simclr.yaml --devices cuda:0 --output_folder ../simclr/linear_evaluation --evaluate_weights ../simclr/models/encoder.pyh &
```

## 4. Results

We run some experiments trying to match the official results.

| Method | Epochs | Linear Eval Top-1 Accuracy (%) |
|:------:|:------:|:------------------------------:|
| SimCLR | 100    | 59.37                          |
| BYOL   | 100    | 61.42                          |
| SwAV   | 200    | 70.90                          |

The executions of the methods and some extra results for different hyperparameter setups are found [here](https://huggingface.co/buckets/LoanMaikon/Self-Supervised-Library).
