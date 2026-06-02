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

| Method | Model     |  Epochs  | Batch Size | Top-1 Accuracy | Notes                                                                                                                                                                                     |
| :----- | :-------- | :------: | :--------: | :------------: | :------------------------------------------------------- |
| SimCLR | ResNet-50 | 100 | 512 | 59.37     | LR: 1e-5 / 0.6 / 0; Weight Decay: 1e-6; Temperature: 0.1; warmup: 10 |
| BYOL | ResNet-50 | 100 | 512 | 61.42 | LR: 1e-4 / 0.4 / 0; Weight Decay: 1e-6; EMA: 0.9995 / 1.0; Scheduler configured for 1000 epochs, but training stopped at 100 epochs; warmup: 10|
| BYOL | ResNet-50 | 100 | 512 | 57.93 | LR: 1e-4 / 0.4 / 0; Weight Decay: 1e-6; EMA: 0.996 / 1.0; warmup: 10 |
| SwAV | ResNet-50 | 200 | 256 | 70.90 | Global / Local Views: 2 / 4; LR: 1e-5 / 0.6 / 6e-4; Weight Decay: 1e-6; Temperature: 0.1; Queue length of 3840 starting at epoch 15; Prototype freezing for 5005 iterations.; warmup: 0 |
| DINO | ViT-S | 100 | 256 | 59.52 | Global / Local Views: 2 / 0; LR: 5e-5 / 5e-4 / 1e-6; Weight Decay: 0.04 / 0.4; EMA: 0.9995 / 1.0; Teacher Temp.: 0.04 / 0.07 / 0.07 (30); warmup: 0 |
| DINO | ViT-S | 100 | 256 | 67.18 | Global / Local Views: 2 / 2; LR: 5e-5 / 5e-4 / 1e-6; Weight Decay: 0.04 / 0.4; EMA: 0.9995 / 1.0; Teacher Temp.: 0.04 / 0.04 / 0.04 (0); warmup: 0 |
| iBOT | ViT-S | 100 | 256 | 71.94 | Global / Local Views: 2 / 2; LR: 5e-5 / 5e-4 / 1e-6; Weight Decay: 0.04 / 0.4; EMA: 0.996 / 1.0; Teacher Temp. CLS: 0.04 / 0.07 / 0.07 (30); Teacher Temp. Patch: 0.04 / 0.07 / 0.07 (30); warmup: 10 |
| I-JEPA | ViT-B | 100 | 256 | 47.49 | Warmup: 15; EMA: 0.996 / 1.0; lr: 1e-4 / 1e-3 / 1e-6; wd: 0.04 / 0.4 |
| I-JEPA | ViT-B | 100 | 256 | 50.41 | Warmup: 15; EMA: 0.996 / 1.0; lr: 1e-4 / 5e-3 / 1e-6; scheduler set for 600 epochs, stopped at 100 epochs; wd: 0.04 / 0.4 |
| I-JEPA | ViT-B | 100 | 256 | 58.22 | Warmup: 15; EMA: 0.9995 / 1.0; lr: 1e-4 / 1e-3 / 1e-6; scheduler set for 600 epochs, stopped at 100 epochs; wd: 0.04 / 0.4 |
| MAE | ViT-B | 400 | 256 | 50.31 | Scheduler set to 800 epochs, stopped at 400 epochs; lr: 1e-5 / 3e-5 / 0; wd: 0.5; warmup: 40 |
| MAE | ViT-B | 800 | 256 | 54.27 | lr: 1e-5 / 3e-5 / 0; wd: 0.5; warmup: 40 |
| Barlow Twins | ResNet-50 | 100 | 256 | 56.12 | lr_weights: 1e-4 / 0.4 / 0; lr_biases: 1e-6 / 0.0096 / 0; warmup: 10; schedulers set to 1000 epochs, stopped at 100 |
| VICReg | ResNet-50 | 100 | 256 | 63.84 | lr: 1e-4 / 0.4 / 0.002; warmup: 10; schedulers set to 1000 epochs, stopped at 100 |

The full training runs and additional experiments with different hyperparameter configurations can be found [here](https://huggingface.co/buckets/LoanMaikon/Self-Supervised-Library).

These results include training curves and analysis for learning rate, weight decay, exponential moving average (EMA), training loss, linear evaluation accuracy, and other relevant metrics.

Our goal is to provide useful empirical insights that help the community better understand self-supervised learning methods and their often challenging hyperparameter tuning process.
