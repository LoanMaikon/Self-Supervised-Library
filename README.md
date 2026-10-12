<div align="center">

# Self-Supervised Library

15 self-supervised methods · Pretraining · Linear evaluation · Fine-tuning

[Quick start](#quick-start) · [Methods](#available-methods) · [Training & evaluation](#training--evaluation) · [Results](#results) · [Licensing](#licensing) · [Experiment archive](https://huggingface.co/buckets/LoanMaikon/Self-Supervised-Library)

</div>

---

## Overview

Self-Supervised Library is a personal research project that brings together self-supervised learning methods behind a shared training entry point. Use it to pretrain encoders, evaluate learned representations and explore how hyperparameters affect training.

- **One workflow for 15 methods:** select a method through its YAML configuration.
- **Pretraining and evaluation:** train an encoder, fit a linear classifier, or fine-tune the encoder and classifier together.
- **Single and multiple GPUs:** run experiments with `torchrun` and distributed data parallel training.
- **Experiment tracking:** save checkpoints, logs, metrics, and training curves; resume from the last saved epoch.

## Available methods

Each method includes a pretraining configuration and an evaluation configuration. Backbones below are the defaults in the supplied pretraining files; some implementations provide additional architectures.

| Method | Default backbone | Pretraining | Evaluation |
| :--- | :--- | :---: | :---: |
| SimCLR | ResNet-50 | [Config](configs/pretraining/pretraining_simclr.yaml) | [Config](configs/evaluate/evaluate_simclr.yaml) |
| BYOL | ResNet-50 | [Config](configs/pretraining/pretraining_byol.yaml) | [Config](configs/evaluate/evaluate_byol.yaml) |
| SwAV | ResNet-50 | [Config](configs/pretraining/pretraining_swav.yaml) | [Config](configs/evaluate/evaluate_swav.yaml) |
| MAE | ViT-B | [Config](configs/pretraining/pretraining_mae.yaml) | [Config](configs/evaluate/evaluate_mae.yaml) |
| I-JEPA | ViT-B | [Config](configs/pretraining/pretraining_ijepa.yaml) | [Config](configs/evaluate/evaluate_ijepa.yaml) |
| iBOT | ViT-S | [Config](configs/pretraining/pretraining_ibot.yaml) | [Config](configs/evaluate/evaluate_ibot.yaml) |
| DINOv1 | ViT-S | [Config](configs/pretraining/pretraining_dinov1.yaml) | [Config](configs/evaluate/evaluate_dinov1.yaml) |
| DINOv2 | ViT-S | [Config](configs/pretraining/pretraining_dinov2.yaml) | [Config](configs/evaluate/evaluate_dinov2.yaml) |
| DINOv3 | ViT-S | [Config](configs/pretraining/pretraining_dinov3.yaml) | [Config](configs/evaluate/evaluate_dinov3.yaml) |
| Barlow Twins | ResNet-50 | [Config](configs/pretraining/pretraining_barlow_twins.yaml) | [Config](configs/evaluate/evaluate_barlow_twins.yaml) |
| VICReg | ResNet-50 | [Config](configs/pretraining/pretraining_vicreg.yaml) | [Config](configs/evaluate/evaluate_vicreg.yaml) |
| SimSiam | ResNet-50 | [Config](configs/pretraining/pretraining_simsiam.yaml) | [Config](configs/evaluate/evaluate_simsiam.yaml) |
| MSN | ViT-S | [Config](configs/pretraining/pretraining_msn.yaml) | [Config](configs/evaluate/evaluate_msn.yaml) |
| MoCov1 | ResNet-50 | [Config](configs/pretraining/pretraining_mocov1.yaml) | [Config](configs/evaluate/evaluate_mocov1.yaml) |
| MoCov2 | ResNet-50 | [Config](configs/pretraining/pretraining_mocov2.yaml) | [Config](configs/evaluate/evaluate_mocov2.yaml) |

## Quick start

### Installation

```bash
git clone https://github.com/LoanMaikon/Self-Supervised-Library.git
cd Self-Supervised-Library

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

```bash
python -m pip install torch torchvision numpy scipy matplotlib pyyaml pillow timm
```

**DINOv2 also requires xFormers** for pretraining and evaluation. Follow the [xFormers installation instructions](https://github.com/facebookresearch/xformers#installing-xformers) to select a build compatible with PyTorch and CUDA. Dependency versions are currently not pinned in this repository.

### Prepare ImageNet

Download the ImageNet 2012 classification archives from the [official ImageNet website](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php), signing in to access the images. Place these three files in the same directory:

- `ILSVRC2012_img_train.tar`
- `ILSVRC2012_img_val.tar`
- `ILSVRC2012_devkit_t12.tar.gz` — development kit for Tasks 1 & 2.

Run [scripts/prepare_imagenet.py](scripts/prepare_imagenet.py) to extract them into the layout expected by the library.

```bash
python scripts/prepare_imagenet.py \
  --archives-dir /path/to/downloaded/archives \
  --datasets-dir /path/to/datasets
```

The script creates `/path/to/datasets/imagenet/`, extracts the nested training archives into class folders.

```text
/path/to/datasets/
└── imagenet/
    ├── train/
    │   ├── n01440764/
    │   │   └── n01440764_*.JPEG
    │   └── ...
    ├── val/
    │   ├── ILSVRC2012_val_00000001.JPEG
    │   └── ...
    └── ILSVRC2012_devkit_t12/
        └── data/
            ├── meta.mat
            └── ILSVRC2012_validation_ground_truth.txt
```

The loader uses the official validation images as its `test` split. An optional validation subset can be held out from the training data through `data.separate_val_subset` for hyperparameter tuning.

ImageNet is currently the dataset registered in [src/datasets.py](src/datasets.py). To use a custom dataset, extend that loader and its class-count mapping before selecting the new dataset in a configuration.

### Configure your experiment

Edit the selected files in [configs/pretraining](configs/pretraining) and [configs/evaluate](configs/evaluate). Update the dataset path in **both** configurations and adjust batch size and worker count for your hardware.

## Training & evaluation

Run all commands from the repository root. Use a new `--output_folder` for each experiment; an existing folder requires `--continue_training`.

### Pretrain on one GPU

This example trains SimCLR and writes its outputs to `../runs/simclr`:

```bash
torchrun --standalone --nproc_per_node=1 main.py \
  --config configs/pretraining/pretraining_simclr.yaml \
  --devices cuda:0 \
  --output_folder ../runs/simclr
```

To train another method, replace `--config` with its pretraining file and choose a new output folder.

### Pretrain on multiple GPUs

Launch one process per GPU. For example, to use two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 main.py \
  --config configs/pretraining/pretraining_simclr.yaml \
  --devices cuda:0 cuda:1 \
  --output_folder ../runs/simclr_2gpu
```

For distributed runs, `CUDA_VISIBLE_DEVICES` selects the physical GPUs and `torchrun` assigns each process its local device. The `--devices` argument remains required by the CLI.

### Linear evaluation

The supplied evaluation configurations use `meta.mode: linear_eval`: the encoder is frozen and a linear classifier is trained on its representations.

After pretraining, evaluate the saved SimCLR encoder:

```bash
torchrun --standalone --nproc_per_node=1 main.py \
  --config configs/evaluate/evaluate_simclr.yaml \
  --devices cuda:0 \
  --output_folder ../runs/simclr/linear_evaluation \
  --evaluate_weights ../runs/simclr/models/encoder.pth
```

### Fine-tuning

Set `meta.mode` to `fine_tuning` in your evaluation configuration to train the encoder and classifier together. Keep the other fields in the `meta` section:

```yaml
meta:
  mode: fine_tuning
  framework: simclr
```

Review the optimizer and learning rate for fine-tuning, then run the evaluation command with a new output folder, such as `../runs/simclr/fine_tuning`.

### Resume training

Use the same configuration and output folder, adding `--continue_training`:

```bash
torchrun --standalone --nproc_per_node=1 main.py \
  --config configs/pretraining/pretraining_simclr.yaml \
  --devices cuda:0 \
  --output_folder ../runs/simclr \
  --continue_training
```

Training resumes after the last saved epoch using the model, optimizer, scheduler, and scaler checkpoints. Keep the complete output folder, including `models/` and `last_epoch.json`. To resume evaluation, also retain the `--evaluate_weights` argument and its original checkpoint file.

### Evaluate external checkpoints

To evaluate compatible weights released by a method's authors, download the checkpoint locally, select the matching evaluation configuration, and pass the **checkpoint file** to `--evaluate_weights`:

```bash
torchrun --standalone --nproc_per_node=1 main.py \
  --config configs/evaluate/evaluate_swav.yaml \
  --devices cuda:0 \
  --output_folder ../runs/swav_original \
  --evaluate_weights /path/to/swav_checkpoint.pth
```

The architecture and checkpoint format must match a loader supported by the selected `meta.framework`. Available loaders are defined in [src/methods/evaluation/Evaluation.py](src/methods/evaluation/Evaluation.py) and the corresponding model implementations.

### Command-line arguments

| Argument | Required | Description |
| :--- | :---: | :--- |
| `--config` | Yes | Path to a pretraining or evaluation YAML file. |
| `--devices` | Yes | CUDA device names, e.g. `cuda:0` or `cuda:0 cuda:1`. |
| `--output_folder` | Yes | Directory for the experiment's outputs. |
| `--evaluate_weights` | For evaluation | Path to the encoder checkpoint file. |
| `--continue_training` | No | Resumes an experiment from its saved state. |

## Experiment outputs

Each run saves its configuration alongside model checkpoints and metrics. A typical SimCLR pretraining folder contains:

```text
../runs/simclr/
├── config.yaml
├── log.txt
├── log.csv
├── last_epoch.json
├── training_info.json
├── models/
│   ├── encoder.pth
│   ├── projection_head.pth
│   ├── optimizer.pth
│   ├── lr_scheduler.pth
│   ├── wd_scheduler.pth
│   └── scaler.pth
└── figs/
    ├── loss.png
    ├── learning_rate.png
    └── weight_decay.png
```

Other methods save additional components, such as teacher networks or prediction heads. Evaluation also saves classifier weights, `testing_info.json`, and accuracy curves; runs with a held-out validation subset save validation metrics too.

## Results

The table preserves the results recorded for this project. Top-1 accuracy is shown as a percentage. Architectures, training budgets, and schedules vary across runs; use the experiment notes when comparing values.

| Method | Backbone | Epochs | Batch size | Top-1 (%) |
| :--- | :--- | ---: | ---: | ---: |
| SimCLR | ResNet-50 | 100 | 512 | 59.37 |
| BYOL | ResNet-50 | 100 | 512 | 61.42 |
| BYOL | ResNet-50 | 100 | 512 | 57.93 |
| SwAV | ResNet-50 | 200 | 256 | 70.90 |
| DINOv1 | ViT-S | 100 | 256 | 59.52 |
| DINOv1 | ViT-S | 100 | 256 | 67.18 |
| iBOT | ViT-S | 100 | 256 | 71.94 |
| I-JEPA | ViT-B | 100 | 256 | 47.49 |
| I-JEPA | ViT-B | 100 | 256 | 50.41 |
| I-JEPA | ViT-B | 100 | 256 | 58.22 |
| MAE | ViT-B | 400 | 512 | 50.31 |
| MAE | ViT-B | 800 | 512 | 54.27 |
| Barlow Twins | ResNet-50 | 100 | 512 | 56.12 |
| VICReg | ResNet-50 | 100 | 512 | 63.84 |
| DINOv2 | ViT-S | 100 | 256 | 65.58 |
| DINOv2 | ViT-S | 100 | 256 | 8.00 |
| DINOv2 | ViT-S | 100 | 256 | 65.66 |
| SimSiam | ResNet-50 | 100 | 512 | 68.18 |
| MoCov1 | ResNet-50 | 200 | 256 | 56.22 |
| MoCov2 | ResNet-50 | 200 | 256 | 64.86 |

<details>
<summary><strong>Experiment notes and hyperparameters for all 20 runs</strong></summary>

Run IDs correspond to the table above. LR sequences describe initial / peak / final learning rates; WD and EMA pairs describe initial / final values. Teacher temperature sequences include their warmup duration in parentheses. Warmup values are in epochs unless otherwise stated.

| Run | Hyperparameters and schedule |
| :--- | :--- |
| R01 | LR: 1e-5 / 0.6 / 0; WD: 1e-6; temperature: 0.1; warmup: 10. |
| R02 | LR: 1e-4 / 0.4 / 0; WD: 1e-6; EMA: 0.9995 / 1.0; warmup: 10; scheduler configured for 1000 epochs, training stopped at 100. |
| R03 | LR: 1e-4 / 0.4 / 0; WD: 1e-6; EMA: 0.996 / 1.0; warmup: 10. |
| R04 | Global / local views: 2 / 4; LR: 1e-5 / 0.6 / 6e-4; WD: 1e-6; temperature: 0.1; queue length: 3840, starting at epoch 15; prototypes frozen for 5005 iterations; warmup: 0. |
| R05 | Global / local views: 2 / 0; LR: 5e-5 / 5e-4 / 1e-6; WD: 0.04 / 0.4; EMA: 0.9995 / 1.0; teacher temperature: 0.04 / 0.07 / 0.07 (30); warmup: 0. |
| R06 | Global / local views: 2 / 2; LR: 5e-5 / 5e-4 / 1e-6; WD: 0.04 / 0.4; EMA: 0.9995 / 1.0; teacher temperature: 0.04 / 0.04 / 0.04 (0); warmup: 0. |
| R07 | Global / local views: 2 / 2; LR: 5e-5 / 5e-4 / 1e-6; WD: 0.04 / 0.4; EMA: 0.996 / 1.0; teacher CLS temperature: 0.04 / 0.07 / 0.07 (30); teacher patch temperature: 0.04 / 0.07 / 0.07 (30); warmup: 10. |
| R08 | LR: 1e-4 / 1e-3 / 1e-6; WD: 0.04 / 0.4; EMA: 0.996 / 1.0; warmup: 15. |
| R09 | LR: 1e-4 / 5e-3 / 1e-6; WD: 0.04 / 0.4; EMA: 0.996 / 1.0; warmup: 15; scheduler configured for 600 epochs, training stopped at 100. |
| R10 | LR: 1e-4 / 1e-3 / 1e-6; WD: 0.04 / 0.4; EMA: 0.9995 / 1.0; warmup: 15; scheduler configured for 600 epochs, training stopped at 100. |
| R11 | LR: 1e-5 / 3e-5 / 0; WD: 0.5; warmup: 40; scheduler configured for 800 epochs, training stopped at 400. |
| R12 | LR: 1e-5 / 3e-5 / 0; WD: 0.5; warmup: 40. |
| R13 | LR (weights): 1e-4 / 0.4 / 0; LR (biases): 1e-6 / 0.0096 / 0; warmup: 10; schedulers configured for 1000 epochs, training stopped at 100. |
| R14 | LR: 1e-4 / 0.4 / 0.002; warmup: 10; schedulers configured for 1000 epochs, training stopped at 100. |
| R15 | LR: 1e-5 / 1e-4 / 1e-6; WD: 0.04 / 0.4; EMA: 0.994 / 1.0; global / local views: 2 / 2; no registers; warmup: 10. |
| R16 | LR: 1e-5 / 1e-4 / 1e-6; WD: 0.04 / 0.4; EMA: 0.9995 / 1.0; global / local views: 2 / 2; no registers; warmup: 10. |
| R17 | LR: 1e-5 / 1e-4 / 1e-6; WD: 0.04 / 0.4; EMA: 0.995 / 1.0; global / local views: 2 / 2; no registers; warmup: 10. |
| R18 | LR: 0.1 / 0.1 / 0; WD: 0.0001. |
| R19 | LR: 0.03 / 0.03 / 0; WD: 0.0001. |
| R20 | LR: 0.03 / 0.03 / 0.0003; WD: 0.0001. |

</details>

Full training runs and additional hyperparameter experiments are available in the [Hugging Face experiment archive](https://huggingface.co/buckets/LoanMaikon/Self-Supervised-Library), including training curves, learning rate and weight decay schedules, EMA, losses, and linear evaluation metrics.

## Licensing

The repository incorporates source under **Apache-2.0, MIT, CC BY-NC 4.0, BSD-style terms, and the DINOv3 License Agreement**. [LICENSE](LICENSE) records their scope; [third-party notices](LICENSES/THIRD_PARTY_NOTICES) document the files, functions, sources, and attribution. Full upstream texts and verification metadata are in [LICENSES/](LICENSES).

**The current workflows include noncommercial components:** all 15 methods and evaluation use shared schedulers derived from I-JEPA. Other shared utilities also contain CC BY-NC code. The license of a method's backbone alone does not describe the permissions for the complete workflow.

The backbone in `src/methods/byol/resnet.py` comes from a converter for which no explicit license was found. Its permission status remains unresolved. No new license has been selected for the project's own contributions; this documentation does not grant additional rights or establish complete licensing clearance. Dataset and externally downloaded checkpoint terms are separate.

## Repository structure

```text
Self-Supervised-Library/
├── LICENSE                 # License scope index
├── LICENSES/               # Upstream license texts, notices, and source metadata
├── main.py                 # Training and evaluation entry point
├── configs/
│   ├── pretraining/         # One pretraining configuration per method
│   └── evaluate/            # Linear evaluation and fine-tuning configurations
├── scripts/
│   └── prepare_imagenet.py  # Extract the official ImageNet 2012 archives
└── src/
    ├── Model.py            # Configuration loading and method selection
    ├── datasets.py         # ImageNet loading and class mapping
    ├── schedulers.py       # Learning rate, weight decay, EMA, and temperature schedules
    ├── utils.py            # Logging, plotting, and distributed helpers
    └── methods/            # SSL implementations and shared evaluation pipeline
```

## Contributing

Issues and pull requests are welcome. I hope community enjoy this work.
