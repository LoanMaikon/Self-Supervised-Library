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
- Barlow Twins
- VICReg

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

## 4. Continue Training

For those who have a poor card (like me), you can stop the executions any time and continue from the last checkpoint by attaching `--continue_training` to the commands, as follows:

```bash
nohup torchrun --nproc_per_node=1 main.py --config configs/pretraining/pretraining_simclr.yaml --devices cuda:0 --output_folder ../simclr --continue_training &
```

```bash
nohup torchrun --nproc_per_node=1 main.py --config configs/evaluate/evaluate_simclr.yaml --devices cuda:0 --output_folder ../simclr/linear_evaluation --continue_training &
```

Note that for Linear Evaluation and Fine-tuning, when `--continue_training` is passed, there is no need for passing `--evaluate_weights` again.

## 5. Results

We run some experiments trying to match the official results.

| Method | Model     | Epochs | Batch Size | Linear Evaluation Top-1 Accuracy (%) |
|--------|-----------|--------|------------|--------------------------------------|
| SimCLR | ResNet-50 | 100    | 512        | 59.37                                |
| BYOL   | ResNet-50 | 100    | 512        | 61.42                                |
| SwAV   | ResNet-50 | 200    | 256        | 70.90                                |

The full training runs and additional experiments with different hyperparameter configurations can be found [here](https://huggingface.co/buckets/LoanMaikon/Self-Supervised-Library).

These results include training curves and analysis for learning rate, weight decay, exponential moving average (EMA), training loss, linear evaluation accuracy, and other relevant metrics.

Our goal is to provide useful empirical insights that help the community better understand self-supervised learning methods and their often challenging hyperparameter tuning process.
