<div align="center">

# Self-supervised Library

This is the repository containing the source code and documentation for my personal project **Self-supervised Library**.

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
- DINOv2
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

---

## 3. Linear Evaluation and Fine-tuning

You can easily run a linear evaluation or fine-tuning using a config file found at `configs/evaluate`. For instance, the following command runs a SimCLR linear evaluation in a single GPU <cuda:0> using the encoder trained previously. The output is set to `../simclr/linear_evaluation`

```bash
nohup torchrun --nproc_per_node=1 main.py --config configs/evaluate/evaluate_simclr.yaml --devices cuda:0 --output_folder ../simclr/linear_evaluation --evaluate_weights ../simclr/models/encoder.pyh &
```

---

## 4. Continue Training

For those who have a poor card (like me), you can stop the executions any time and continue from the last checkpoint by attaching `--continue_training` to the commands, as follows:

```bash
nohup torchrun --nproc_per_node=1 main.py --config configs/pretraining/pretraining_simclr.yaml --devices cuda:0 --output_folder ../simclr --continue_training &
```

---

## 5. Evaluate Original Weights

This library also allows you to evaluate the original weights provided by the authors. For instance, the following command runs a SwAV linear evaluation in a single GPU <cuda:0> using the original encoder weights. The output is set to `../swav_original`.

```bash
nohup torchrun --nproc_per_node=1 main.py --config configs/evaluate/evaluate_swav.yaml --devices cuda:0 --output_folder ../swav_original --evaluate_weights /original/swav/model/ &
```

This also allows you to experiment original models with custom datasets.

---

## 6. Results

We conducted several experiments to reproduce the official results and evaluate different hyperparameter settings.

### SimCLR

| Model | Epochs (Warmup) | Top-1 Acc. | Batch Size | LR (init / max / final) | Weight Decay | Temperature | Notes |
|:---:|:---:|:---:|:---:|:---|:---:|:---:|:---|
| ResNet-50 | 100 (10) | 59.37 | 512 | 1e-5 / 0.6 / 0 | 1e-6 | 0.1 | - |

### BYOL

| Model | Epochs (Warmup) | Top-1 Acc. | Batch Size | LR (init / max / final) | Weight Decay | EMA (init / final) | Notes |
|:---:|:---:|:---:|:---:|:---|:---:|:---:|:---|
| ResNet-50 | 100 (10) | 61.42 | 512 | 1e-4 / 0.4 / 0 | 1e-6 | 0.9995 / 1.0 | Scheduler configured for 1000 epochs,<br>but training stopped at 100 epochs. |
| ResNet-50 | 100 (10) | 57.93 | 512 | 1e-4 / 0.4 / 0 | 1e-6 | 0.996 / 1.0 | - |

### SwAV

| Model | Epochs (Warmup) | Top-1 Acc. | Batch Size | Global / Local Views | LR (init / max / final) | Weight Decay | Temperature | Notes |
|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---:|:---|
| ResNet-50 | 200 (0) | 70.90 | 256 | 2 / 4 | 1e-5 / 0.6 / 6e-4 | 1e-6 | 0.1 | Queue length of 3840 starting at epoch 15.<br>Prototype freezing for 5005 iterations. |

### DINO

| Model | Epochs (Warmup) | Top-1 Acc. | Batch Size | Global / Local Views | LR (init / max / final) | Weight Decay (init / final) | EMA (init / final) | Teacher Temp. (init / max / final) (Warmup) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---:|:---|:---|
| ViT-S | 100 (0) | 59.52 | 256 | 2 / 0 | 5e-5 / 5e-4 / 1e-6 | 0.04 / 0.4 | 0.9995 / 1.0 | 0.04 / 0.07 / 0.07 (30) | - |
| ViT-S | 100 (0) | 67.18 | 256 | 2 / 2 | 5e-5 / 5e-4 / 1e-6 | 0.04 / 0.4 | 0.9995 / 1.0 | 0.04 / 0.04 / 0.04 (0) | - |

### iBOT

| Model | Epochs (Warmup) | Top-1 Acc. | Batch Size | Global / Local Views | LR (init / max / final) | Weight Decay (init / final) | EMA (init / final) | Teacher Temp. CLS (init / max / final) (Warmup) | Teacher Temp. Patch (init / max / final) (Warmup) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---:|:---|:---|:---|
| ViT-S | 100 (10) | 71.94 | 256 | 2 / 2 | 5e-5 / 5e-4 / 1e-6 | 0.04 / 0.4 | 0.996 / 1.0 | 0.04 / 0.07 / 0.07 (30) | 0.04 / 0.07 / 0.07 (30) | - |


The full training runs and additional experiments with different hyperparameter configurations can be found [here](https://huggingface.co/buckets/LoanMaikon/Self-Supervised-Library).

These results include training curves and analysis for learning rate, weight decay, exponential moving average (EMA), training loss, linear evaluation accuracy, and other relevant metrics.

Our goal is to provide useful empirical insights that help the community better understand self-supervised learning methods and their often challenging hyperparameter tuning process.
