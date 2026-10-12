# Licensing and third-party notices

This folder documents the existing licenses of incorporated source, its origins, and the scope of each license. No new license has been selected for the project's own contributions.

| File | Contents |
| :--- | :--- |
| [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) | All 18 original license texts, separated by project, with credits and conditions preserved. |
| [sources.json](sources.json) | Source snapshots, file/function mappings, section locations, and verification checksums. |
| This guide | The existing license scope and complete third-party attribution audit below. |

The incorporated source includes Apache-2.0, MIT, CC BY-NC 4.0, BSD-style terms, and the DINOv3 License Agreement. Each keeps its own scope and conditions.

**The current workflows include noncommercial components:** all 15 methods and evaluation use shared schedulers derived from I-JEPA. The BYOL converter-derived backbone also has an unresolved permission status. The full scope and outstanding questions are preserved below; this guide does not grant new rights or resolve missing permission.

Paths to project source and `LICENSES/` in the preserved audit are relative to the repository root. The two document links above are relative to this folder.

## Existing license scope

<details>
<summary>Full scope index and outstanding questions</summary>

```text
Self-Supervised Library - existing license scope
Audit date: 2026-10-12

This repository contains source from several projects under different
licenses. This file is an index of existing terms, not a new license grant
for the repository as a whole. No new license has been selected for
project-authored code, documentation, configurations, or modifications.
Those contributions must not be assumed to be licensed under the terms
of an upstream project solely because they implement the same method.

Existing third-party notices and license terms continue to apply to their
respective source and adapted portions. The complete upstream license
texts are preserved, without modification, in separate project sections of
LICENSES/THIRD_PARTY_LICENSES.txt. File/function scope, attribution,
changes, sources, and unresolved provenance are
documented in LICENSES/README.md. Verification snapshots and
license checksums are recorded in LICENSES/sources.json.

Principal upstream terms found in incorporated source:

  Apache-2.0:
    SimCLR converter, DINOv1, DINOv2, iBOT, Lightning Bolts, and timm.
    MSN's src/deit.py backbone has an explicit Apache-2.0 file notice;
    MSN's root license and its losses/utilities are CC-BY-NC-4.0.

  MIT:
    Barlow Twins, VICReg, and the verified relicensed MoCo implementation
    used as a reference for MoCov1/MoCov2.

  CC-BY-NC-4.0:
    SwAV, MAE, I-JEPA, SimSiam, MSN losses/utilities, and shared code
    derived from MAE, I-JEPA, and MoCo v3.

  DINOv3 License Agreement:
    Incorporated DINOv3 model, loss, and masking/collation source and
    adaptations. See section [dinov3] in LICENSES/THIRD_PARTY_LICENSES.txt
    for the complete agreement.

  BSD-style / BSD-3-Clause:
    Incorporated PyTorch initialization code and inherited torchvision
    ResNet portions. Imported dependencies also retain their own terms.

Important scope and outstanding issues:

  - All 15 method controllers and the evaluation pipeline use the shared
    I-JEPA-derived schedulers. CC-BY-NC terms apply to the covered source
    and adaptations; the complete current workflow cannot be represented
    as having unrestricted commercial-use permission. An import does not
    automatically relicense the importing file.

  - src/methods/byol/resnet.py identifies ajtejankar/byol-convert as its
    source. No explicit license was found in that upstream snapshot.
    This index does not authorize use or redistribution of portions for
    which permission has not been established. The license of the
    original BYOL research implementation does not resolve this gap.

  - Some files contain both project-specific and third-party portions.
    See the function-level scope in LICENSES/README.md; no
    blanket license is assigned to those mixed files by this index.

  - This source-code audit does not grant rights to ImageNet, separately
    downloaded model weights, or other external assets. Their respective
    terms must be checked separately.

The upstream license texts govern their covered materials. This index
does not replace their conditions, cure missing permission, or certify
that the entire repository is fully licensed for a particular use.
```

</details>

## Third-party attribution audit

<details>
<summary>All 15 methods, shared utilities, copyrights, sources, and adaptations</summary>

```text
Self-Supervised Library - third-party notices and source audit
============================================================
Audit date: 2026-10-12

This document records existing source provenance and license scope. It does
not adopt a new license for project-authored contributions or certify that
all permission questions have been resolved. The scope index is preserved
in this licensing guide. Local paths below are relative to the repository root.

Verification commits below pin the evidence examined; original copied
revisions were generally not recorded in local source. A method name, paper
license, or imported module does not establish the license of every local file.

CURRENT WORKFLOW SCOPE
----------------------
All 15 method controllers and the evaluation pipeline import the shared
I-JEPA-derived cosine schedulers (CC-BY-NC-4.0). LARS is derived from MAE/
MoCo v3 (CC-BY-NC-4.0), and is shared by SimCLR, BYOL, SwAV, Barlow Twins,
VICReg and evaluation. MAE positional helpers are used by MAE and I-JEPA.
Therefore permissive backbone terms alone do not provide unrestricted
commercial-use permission for a complete current workflow. This does not
automatically relicense the independent importing code.

METHOD MAP (direct incorporated source; shared terms also apply)
----------------------------------------------------------------
SimCLR       : Apache-2.0 converter source; BSD torchvision ancestry.
BYOL         : Converter-derived backbone permission unresolved.
SwAV         : CC-BY-NC-4.0 adapted ResNet, Sinkhorn and loss; BSD ancestry.
MAE          : CC-BY-NC-4.0 model and positional embedding source.
I-JEPA       : CC-BY-NC-4.0 model, masking and shared helper source.
iBOT         : Apache-2.0 transformer/head and masking source.
DINOv1       : Apache-2.0 transformer/head source.
DINOv2       : Apache-2.0 transformer, layers, losses and masking source.
DINOv3       : DINOv3 License Agreement for incorporated source/adaptations.
Barlow Twins : MIT incorporated model/projector/loss source.
VICReg       : MIT incorporated model/gather/loss source; BSD ancestry.
SimSiam      : CC-BY-NC-4.0 incorporated builder source.
MSN          : Apache-2.0 explicit backbone header; CC-BY-NC losses/utilities.
MoCov1       : MIT verified relicensed official MoCo builder source.
MoCov2       : MIT verified relicensed official MoCo builder source.

UPSTREAM LICENSE TEXTS AND ATTRIBUTION
-------------------------------------
[dino] https://github.com/facebookresearch/dino
Verification commit: 7c446df5b9f45747937fb0d72314eb9f7b66930a
Terms: Apache-2.0
Notice: Copyright (c) Facebook, Inc. and its affiliates.
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: dino]
Official license: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/LICENSE

[dinov2] https://github.com/facebookresearch/dinov2
Verification commit: 7764ea0f912e53c92e82eb78a2a1631e92725fc8
Terms: Apache-2.0
Notice: Copyright (c) Meta Platforms, Inc. and affiliates.
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: dinov2]
Official license: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE

[ibot] https://github.com/bytedance/ibot
Verification commit: da316d82636a7a7356835ef224b13d5f3ace0489
Terms: Apache-2.0
Notice: Copyright (c) ByteDance, Inc. and its affiliates. All rights reserved.
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: ibot]
Official license: https://github.com/bytedance/ibot/blob/da316d82636a7a7356835ef224b13d5f3ace0489/LICENSE

[mae] https://github.com/facebookresearch/mae
Verification commit: efb2a8062c206524e35e47d04501ed4f544c0ae8
Terms: CC-BY-NC-4.0
Notice: Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: mae]
Official license: https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/LICENSE

[ijepa] https://github.com/facebookresearch/ijepa
Verification commit: 52c1ae95d05f743e000e8f10a1f3a79b10cff048
Terms: CC-BY-NC-4.0
Notice: Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: ijepa]
Official license: https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/LICENSE

[msn] https://github.com/facebookresearch/msn
Verification commit: 4388dc1eadbe3042b85d3296d41b9b207656e043
Terms: CC-BY-NC-4.0 (root; backbone file has Apache-2.0 notice)
Notice: Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: msn]
Official license: https://github.com/facebookresearch/msn/blob/4388dc1eadbe3042b85d3296d41b9b207656e043/LICENSE

[swav] https://github.com/facebookresearch/swav
Verification commit: 06b1b7cbaf6ba2a792300d79c7299db98b93b7f9
Terms: CC-BY-NC-4.0
Notice: Copyright (c) Facebook, Inc. and its affiliates.
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: swav]
Official license: https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE

[simsiam] https://github.com/facebookresearch/simsiam
Verification commit: a7bc1772896d0dad0806c51f0bb6f3b16d290468
Terms: CC-BY-NC-4.0
Notice: Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: simsiam]
Official license: https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/LICENSE

[barlowtwins] https://github.com/facebookresearch/barlowtwins
Verification commit: 8e8d284ca0bc02f88b92328e53f9b901e86b4a3c
Terms: MIT
Notice: Copyright (c) Facebook, Inc. and its affiliates.
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: barlowtwins]
Official license: https://github.com/facebookresearch/barlowtwins/blob/8e8d284ca0bc02f88b92328e53f9b901e86b4a3c/LICENSE

[vicreg] https://github.com/facebookresearch/vicreg
Verification commit: 4e12602fd495af83efd1631fbe82523e6db092e0
Terms: MIT
Notice: Copyright (c) Facebook, Inc. and its affiliates. / Copyright (c) Meta Platforms, Inc. and affiliates. in main_vicreg.py
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: vicreg]
Official license: https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/LICENSE

[moco] https://github.com/facebookresearch/moco
Verification commit: 7397dfe146c7ca6bbb58e9c382498069178ba764
Terms: MIT
Notice: Copyright (c) Meta Platforms, Inc. and affiliates. Existing local Facebook notices are retained.
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: moco]
Official license: https://github.com/facebookresearch/moco/blob/7397dfe146c7ca6bbb58e9c382498069178ba764/LICENSE

[torchvision] https://github.com/pytorch/vision
Verification commit: b68adcf9a9280aef02fc08daed170d74d0892361
Terms: BSD-3-Clause
Notice: Copyright (c) Soumith Chintala 2016,
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: torchvision]
Official license: https://github.com/pytorch/vision/blob/b68adcf9a9280aef02fc08daed170d74d0892361/LICENSE

[dinov3] https://github.com/facebookresearch/dinov3
Verification commit: 6876159a11b4df116f30f667f8c9888617df0751
Terms: DINOv3 License Agreement, last updated August 19, 2025
Notice: Copyright (c) Meta Platforms, Inc. and affiliates.
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: dinov3]
Official license: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md

[simclr_converter] https://github.com/tonylins/simclr-converter
Verification commit: 139d3cb0bd0c64b5ad32aab810e0bd0a0dddaae0
Terms: Apache-2.0
Notice: Copyright [2020] [Ji Lin] (upstream LICENSE appendix)
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: simclr_converter]
Official license: https://github.com/tonylins/simclr-converter/blob/139d3cb0bd0c64b5ad32aab810e0bd0a0dddaae0/LICENSE

[pytorch] https://github.com/pytorch/pytorch
Verification commit: 0398ae1ef13f587e49d96aece24d334d88f562a7
Terms: BSD-style terms; full upstream copyright list retained in license text
Notice: Multiple copyright holders listed in the full PyTorch license.
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: pytorch]
Official license: https://github.com/pytorch/pytorch/blob/0398ae1ef13f587e49d96aece24d334d88f562a7/LICENSE

[timm] https://github.com/huggingface/pytorch-image-models
Verification commit: cfc742eb6a4bc4bb35ef20c9b2d5a2a31f599bba
Terms: Apache-2.0
Notice: Copyright 2019 Ross Wightman (upstream LICENSE appendix)
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: timm]
Official license: https://github.com/huggingface/pytorch-image-models/blob/cfc742eb6a4bc4bb35ef20c9b2d5a2a31f599bba/LICENSE

[bolts] https://github.com/Lightning-Universe/lightning-bolts
Verification commit: 34958a8892d42714edf02ba896fd5861ebb3cb7e
Terms: Apache-2.0
Notice: Copyright 2018-2021 William Falcon (upstream LICENSE appendix)
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: bolts]
Official license: https://github.com/Lightning-Universe/lightning-bolts/blob/34958a8892d42714edf02ba896fd5861ebb3cb7e/LICENSE

[moco_v3] https://github.com/facebookresearch/moco-v3
Verification commit: c349e6e24f40d3fedb22d973f92defa4cedf37a7
Terms: CC-BY-NC-4.0
Notice: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
Unmodified license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: moco_v3]
Official license: https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/LICENSE

[byol_converter] https://github.com/ajtejankar/byol-convert
Verification commit: 7497e7f12016064704c2ad04dc3c90364e80d4a7
Terms: NOASSERTION: no explicit license found
Notice: No explicit copyright/license notice found in the audited converter source.
No license copy exists because no explicit upstream license was found.
Permission must be established with the rights holder, or the affected
implementation must be replaced with appropriately licensed source.

FILE AND FUNCTION SCOPE / LOCAL MODIFICATIONS
---------------------------------------------

01. src/methods/simclr/resnet.py
Source: simclr_converter
Covered portions: ResNet source adapted from the converter; project-specific checkpoint/evaluation helpers are not granted a new license.
Evidence: Explicit local source attribution and matching ResNet structure.
Local adaptations: Local architecture interface, activation checkpointing, weight loading, evaluation helpers and projection-head integration.
Additional notices / source ancestry: torchvision
Upstream source: https://github.com/tonylins/simclr-converter/blob/139d3cb0bd0c64b5ad32aab810e0bd0a0dddaae0/resnet_wider.py

02. src/methods/byol/resnet.py
Source: byol_converter
Covered portions: Converter-derived ResNet and SAME-padding portions: permission unresolved.
Evidence: Explicit local source URL and matching converter code; full audited upstream tree had no LICENSE/COPYING/NOTICE file, and README/source contained no explicit grant.
Local adaptations: Local checkpointing, BYOL model variants and evaluation helpers. No permission is inferred from the BYOL paper, DeepMind repository, or a public GitHub URL.
Additional notices / source ancestry: torchvision
Upstream source: https://github.com/ajtejankar/byol-convert/blob/7497e7f12016064704c2ad04dc3c90364e80d4a7/resnet.py

03. src/methods/dinov1/models.py
Source: dino
Covered portions: Adapted transformer and DINO head portions; existing Apache file notice retained.
Evidence: Declared source and substantial matching blocks.
Local adaptations: Local checkpointing, model construction and evaluation/weight-loading helpers.
Additional notices / source ancestry: timm
Upstream source: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py

04. src/methods/dinov2/models.py
Source: dinov2
Covered portions: Consolidated upstream transformer/layers including channel_adaptive and PatchEmbed; inspected sources have explicit Apache-2.0 headers.
Evidence: Declared source, matching consolidated layers and preserved Meta notice.
Local adaptations: Consolidation, checkpointing and local loading/evaluation interfaces.
Additional notices / source ancestry: dino, timm
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/models/vision_transformer.py
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/layers/mlp.py
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/layers/attention.py
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/layers/block.py
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/layers/layer_scale.py
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/layers/drop_path.py
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/layers/patch_embed.py
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/layers/swiglu_ffn.py
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/layers/dino_head.py

05. src/methods/dinov2/mask_collator.py
Source: dinov2
Covered portions: Adapted masking generator and crop collation.
Evidence: Declared upstream and matching blocks.
Local adaptations: Consolidation and integration with the local dataset/crop interface.
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/data/masking.py
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/data/collate.py

06. src/methods/dinov2/dino_loss.py
Source: dinov2
Covered portions: Adapted DINO loss.
Evidence: Declared source and retained copyright.
Local adaptations: Integration with local training and scheduling.
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/loss/dino_clstoken_loss.py

07. src/methods/dinov2/ibot_loss.py
Source: dinov2
Covered portions: Adapted patch loss.
Evidence: Declared source and retained copyright.
Local adaptations: Integration with local training.
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/loss/ibot_patch_loss.py

08. src/koleo_loss.py
Source: dinov2
Covered portions: KoLeo implementation.
Evidence: Declared source and matching implementation.
Local adaptations: Local shared-module integration; upstream notice restored.
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/loss/koleo_loss.py

09. src/methods/dinov2/DINOv2.py
Source: dinov2
Covered portions: Teacher-output, token-selection and loss-computation portions in train(); surrounding local controller is not assigned a new license.
Evidence: Matching blocks at baseline lines 116-123, 151-163 and 199-213.
Local adaptations: Restructured into the local training/configuration/checkpoint workflow.
Upstream source: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/train/ssl_meta_arch.py

10. src/methods/mae/models.py
Source: mae
Covered portions: Adapted masked autoencoder source.
Evidence: Declared source and substantial matching blocks.
Local adaptations: Model variants, activation checkpointing and local evaluation/weight-loading helpers.
Additional notices / source ancestry: timm
Upstream source: https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py

11. src/pos_embed.py
Source: mae
Covered portions: Copied/adapted positional embedding and interpolation utilities.
Evidence: Declared source and near-verbatim implementation.
Local adaptations: numpy float dtype adaptation and import/module organization. Original TensorFlow, MoCo v3 and DeiT references are retained.
Additional notices / source ancestry: moco_v3
Upstream source: https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py

12. src/methods/ijepa/models.py
Source: ijepa
Covered portions: Adapted encoder/predictor and masking helpers.
Evidence: Declared source and matching model blocks; exact copied revision not recorded.
Local adaptations: Integration with shared positional embeddings, checkpointing and local evaluation interfaces.
Additional notices / source ancestry: mae
Upstream source: https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/models/vision_transformer.py
Upstream source: https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/masks/utils.py

13. src/methods/ijepa/mask_collator.py
Source: ijepa
Covered portions: Adapted MaskCollator.
Evidence: Declared source and matching blocks.
Local adaptations: Integration with the local dataloader and configuration.
Upstream source: https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/masks/multiblock.py

14. src/methods/ibot/models.py
Source: ibot
Covered portions: Adapted iBOT transformer/head source.
Evidence: Declared source, retained ByteDance notice and matching blocks.
Local adaptations: Checkpointing and local weight-loading/evaluation interfaces.
Additional notices / source ancestry: dino, timm
Upstream source: https://github.com/bytedance/ibot/blob/da316d82636a7a7356835ef224b13d5f3ace0489/models/vision_transformer.py

15. src/methods/ibot/mask_collator.py
Source: ibot
Covered portions: Adapted mask-ratio and masking portions.
Evidence: Declared source and matching portions.
Local adaptations: Extraction into a local masking/collation module.
Upstream source: https://github.com/bytedance/ibot/blob/da316d82636a7a7356835ef224b13d5f3ace0489/loader.py

16. src/methods/msn/models.py
Source: msn
Covered portions: Explicit Apache-2.0 file header in both upstream deit.py and local models.py; root LICENSE is CC-BY-NC-4.0. Preserve both and do not infer an Apache license for the entire MSN workflow.
Evidence: Preserved Facebook Apache header and matching transformer source; upstream root/header difference documented.
Local adaptations: Local checkpointing and model/evaluation interfaces.
Additional notices / source ancestry: dino, timm
Upstream source: https://github.com/facebookresearch/msn/blob/4388dc1eadbe3042b85d3296d41b9b207656e043/src/deit.py

17. src/methods/msn/msn_loss.py
Source: msn
Covered portions: MSN loss, Sinkhorn and AllReduce source portions.
Evidence: Declared losses source and additional matching AllReduce utility.
Local adaptations: Integration with local training, schedules and distributed execution.
Upstream source: https://github.com/facebookresearch/msn/blob/4388dc1eadbe3042b85d3296d41b9b207656e043/src/losses.py
Upstream source: https://github.com/facebookresearch/msn/blob/4388dc1eadbe3042b85d3296d41b9b207656e043/src/utils.py

18. src/methods/dinov3/models.py
Source: dinov3
Covered portions: Incorporated DINOv3 model/layer/head source and adaptations; specific additional layer definitions consolidated locally.
Evidence: Explicit source and DINOv3 Agreement header; audited agreement defines software/training code as DINO Materials.
Local adaptations: Consolidation, checkpointing and local construction/weight-loading interfaces. Exact copied revision and full layer-by-layer import history are not recorded.
Upstream source: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/models/vision_transformer.py

19. src/methods/dinov3/mask_collator.py
Source: dinov3
Covered portions: Adapted DINOv3 masking/collation source.
Evidence: Explicit local adaptation statement and verified upstream agreement headers.
Local adaptations: Support for local flat-list crops and backward-compatible collation.
Upstream source: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/data/masking.py
Upstream source: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/data/collate.py

20. src/methods/dinov3/dino_clstoken_loss.py
Source: dinov3
Covered portions: Incorporated DINOv3 loss source and adaptations.
Evidence: Explicit source and preserved DINOv3 Agreement/copyright header.
Local adaptations: Integration into the local training pipeline.
Upstream source: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss/dino_clstoken_loss.py

21. src/methods/dinov3/ibot_patch_loss.py
Source: dinov3
Covered portions: Incorporated DINOv3 loss source and adaptations.
Evidence: Explicit source and preserved DINOv3 Agreement/copyright header.
Local adaptations: Integration into the local training pipeline.
Upstream source: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss/ibot_patch_loss.py

22. src/methods/dinov3/gram_loss.py
Source: dinov3
Covered portions: Incorporated DINOv3 loss source and adaptations.
Evidence: Explicit source and preserved DINOv3 Agreement/copyright header.
Local adaptations: Integration into the local training pipeline.
Upstream source: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss/gram_loss.py

23. src/methods/dinov3/koleo_loss.py
Source: dinov3
Covered portions: Incorporated DINOv3 loss source and adaptations.
Evidence: Explicit source and preserved DINOv3 Agreement/copyright header.
Local adaptations: Integration into the local training pipeline.
Upstream source: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss/koleo_loss.py

24. src/methods/swav/resnet.py
Source: swav
Covered portions: CC-BY-NC-4.0; applies to the upstream-derived portions, not an automatic grant for unrelated project code.
Evidence: Local declared source; 256 of 389 nonblank lines matching upstream, including a contiguous 106-line block.
Local adaptations: Local model/configuration/loading/evaluation or training integration; original notices retained.
Additional notices / source ancestry: torchvision
Upstream source: https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/src/resnet50.py

25. src/sinkhorn.py
Source: swav
Covered portions: CC-BY-NC-4.0; applies to the upstream-derived portions, not an automatic grant for unrelated project code.
Evidence: Local declared source; distributed_sinkhorn adapted with arguments, optional distributed guards, eps clamps.
Local adaptations: Local model/configuration/loading/evaluation or training integration; original notices retained.
Upstream source: https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/main_swav.py

26. src/methods/swav/SwAV.py
Source: swav
Covered portions: CC-BY-NC-4.0 for adapted portions; applies to the upstream-derived portions, not an automatic grant for unrelated project code.
Evidence: _compute_swav_loss carries queue update, assignment computation and swapped prediction loop matching main_swav.py train(); surrounding configuration/logging/checkpoint orchestration is local.
Local adaptations: Local model/configuration/loading/evaluation or training integration; original notices retained.
Upstream source: https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/main_swav.py

27. src/methods/barlow_twins/models.py
Source: barlowtwins
Covered portions: MIT; applies to the upstream-derived portions, not an automatic grant for unrelated project code.
Evidence: Local declared source; projector construction, cross-correlation loss and off_diagonal copied/adapted; local checkpoint/load/eval wrappers.
Local adaptations: Local model/configuration/loading/evaluation or training integration; original notices retained.
Upstream source: https://github.com/facebookresearch/barlowtwins/blob/8e8d284ca0bc02f88b92328e53f9b901e86b4a3c/main.py

28. src/methods/vicreg/models.py
Source: vicreg
Covered portions: MIT; applies to the upstream-derived portions, not an automatic grant for unrelated project code.
Evidence: Local declared source and retained header; 257 of 334 nonblank lines match, including 121 consecutive lines.
Local adaptations: Local model/configuration/loading/evaluation or training integration; original notices retained.
Additional notices / source ancestry: torchvision
Upstream source: https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/resnet.py

29. src/full_gather_layer.py
Source: vicreg
Covered portions: MIT; applies to the upstream-derived portions, not an automatic grant for unrelated project code.
Evidence: Local declared source; FullGatherLayer identical except whitespace. Original file import was in May 2026, after MIT relicensing.
Local adaptations: Local model/configuration/loading/evaluation or training integration; original notices retained.
Upstream source: https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py

30. src/methods/vicreg/VICReg.py
Source: vicreg
Covered portions: MIT for adapted loss portions; applies to the upstream-derived portions, not an automatic grant for unrelated project code.
Evidence: Loss computation closely follows class VICReg forward; local trainer framework elsewhere.
Local adaptations: Local model/configuration/loading/evaluation or training integration; original notices retained.
Upstream source: https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py

31. src/methods/simsiam/models.py
Source: simsiam
Covered portions: CC-BY-NC-4.0; applies to the upstream-derived portions, not an automatic grant for unrelated project code.
Evidence: Local declared source and retained header; 49 of 100 nonblank lines matching. Encoder/projector/predictor and stop-gradient forward adapted; local checkpoint/eval wrappers.
Local adaptations: Local model/configuration/loading/evaluation or training integration; original notices retained.
Upstream source: https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/simsiam/builder.py

32. src/methods/mocov1/models.py, src/methods/mocov2/models.py
Source: moco
Covered portions: MIT (official relicensing 2023-01-18); applies to the upstream-derived portions, not an automatic grant for unrelated project code.
Evidence: Local says Gitee mirror; algorithmic core matches official builder at MIT relicensing snapshot (99 of 235 nonblank lines for MoCov1), with local checkpoint/helpers/eval extensions. Files differ only default mlp False/True.
Local adaptations: Local model/configuration/loading/evaluation or training integration; original notices retained.
Upstream source: https://github.com/facebookresearch/moco/blob/7397dfe146c7ca6bbb58e9c382498069178ba764/moco/builder.py

33. src/schedulers.py
Source: ijepa
Covered portions: WarmupCosineSchedule and CosineWDSchedule derivations; later project-specific schedules/helpers are not automatically relicensed.
Evidence: Initial local commit 56519d4 explicitly attributed I-JEPA; notice later removed. Current implementation retains the adapted cosine schedules.
Local adaptations: Expanded scheduling/state interfaces, parameter-group filters, clamping and additional local schedules.
Upstream source: https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/utils/schedulers.py

34. src/lars.py
Source: mae
Covered portions: MAE/MoCo v3-derived LARS update implementation.
Evidence: Local historical commits 8e22a75 and 8d4d2e1 explicitly attributed MAE; update structure preserved in current code.
Local adaptations: Parameter-group controls, bias/norm decay/adaptation, clipping, epsilon and optimizer-state handling.
Additional notices / source ancestry: moco_v3
Upstream source: https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lars.py

35. src/utils.py
Source: ijepa
Covered portions: repeat_interleave_batch only; exact immediate copying route remains unconfirmed.
Evidence: AST-identical matching function; added in local commit 82a8be46 without a source notice. Conservative attribution, not proof of all-file derivation.
Local adaptations: No substantive difference found for the matching function; unrelated logging/parameter helpers are outside this attribution.
Upstream source: https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/utils/tensors.py

36. src/utils.py
Source: msn
Covered portions: AllReduceSum only; also matches I-JEPA distributed helper.
Evidence: AST-identical helper added in local MSN commit 2dd8d3aa and used by MSN.
Local adaptations: Local class name/integration; retained distributed availability/world-size guards.
Additional notices / source ancestry: ijepa
Upstream source: https://github.com/facebookresearch/msn/blob/4388dc1eadbe3042b85d3296d41b9b207656e043/src/utils.py

37. src/utils.py
Source: pytorch
Covered portions: _no_grad_trunc_normal_ and trunc_normal_ only; possible timm/RW intermediary.
Evidence: Local Cut & paste from PyTorch ... - RW comment; matching initializer. Exact intermediary/version is not established.
Local adaptations: Local helper layout and variant omitting the out-of-range-mean warning; retain PyTorch notices and timm/RW attribution.
Additional notices / source ancestry: timm
Upstream source: https://github.com/pytorch/pytorch/blob/0398ae1ef13f587e49d96aece24d334d88f562a7/torch/nn/init.py

38. src/nt_xent.py
Source: bolts
Covered portions: Adapted nt_xent_loss similarity/positive/negative calculation.
Evidence: Explicit local URL to Lightning Bolts.
Local adaptations: Local normalization, numerical logsumexp calculation and loss-class interface.
Upstream source: https://github.com/Lightning-Universe/lightning-bolts/blob/34958a8892d42714edf02ba896fd5861ebb3cb7e/src/pl_bolts/losses/self_supervised_learning.py

39. src/methods/evaluation/resnet50.py
Source: torchvision
Covered portions: _forward_impl sequence only, conservatively attributed; torchvision model itself is imported rather than vendored.
Evidence: Close matching forward sequence through self.backbone; no explicit copying route in history.
Local adaptations: Activation checkpointing and local evaluation/loading wrapper.
Upstream source: https://github.com/pytorch/vision/blob/b68adcf9a9280aef02fc08daed170d74d0892361/torchvision/models/resnet.py

HISTORY, EXCLUSIONS AND OUTSTANDING QUESTIONS
--------------------------------------------
- MAE, I-JEPA, MSN, SwAV and SimSiam had CC-BY-NC terms before the local
  source additions. DINOv2 changed to Apache-2.0 in August 2023, before
  its local inclusion. Barlow Twins and VICReg were relicensed MIT in
  2021 and 2022, respectively, before their local inclusion.
- Official MoCo commit 7397dfe146c7ca6bbb58e9c382498069178ba764 changed
  LICENSE and builder.py to MIT on 2023-01-18 (PR #135). Its old README
  still mentions CC-BY-NC; the explicit license/source update is recorded.
  Local files cite a Gitee mirror; the verified official corresponding
  MIT source is documented, while the exact mirrored revision is unknown.
- MSN src/deit.py has an explicit Apache-2.0 header, while the upstream
  root LICENSE is CC-BY-NC-4.0. Both notices are preserved; this audit does
  not silently resolve the discrepancy or call the whole MSN workflow Apache.
- DINOv2 has separate Cell-DINO/X-Ray/CLIP terms in its upstream tree.
  Their specialized sources, tokenizers and weights were not found locally.
  Inspected shared transformer/layers, including channel_adaptive and
  PatchEmbed, have Apache-2.0 headers. Unincorporated component licenses
  are not assigned to this code merely because they share an upstream repo.
- PyTorch initializer source and timm/RW ancestry are documented
  conservatively; the exact copied version/intermediary is unknown.
  Likewise, some short matching functions are attributed conservatively
  rather than treated as conclusive proof of a copying route.
- No external source was identified for NanoPark.py or Hochuli.py.
  This is not proof of independent authorship or a license grant.
- Legacy commits may contain replaced source with different provenance
  (for example the initial LARS wrapper); this audit covers the current
  source tree, not licensing clearance for all historical snapshots.

PRESERVATION AND DISTRIBUTION
-----------------------------
Sections in LICENSES/THIRD_PARTY_LICENSES.txt retain full upstream terms
and disclaimers unchanged. The original texts are preserved byte for byte;
sources.json identifies each section and its text checksum and byte range.
Relevant upstream copyright/source notices are retained or restored in
covered files, and adaptations are identified. Apache, MIT and BSD
redistribution conditions must be followed for covered portions. CC BY-NC
requires attribution, identifying changes, license information and its
noncommercial limitation. The DINOv3 Agreement requires its own terms and
agreement copy for covered distributions, plus acknowledgment in research
publications using the materials. Refer to full texts for exact conditions.

This documentation does not imply upstream endorsement. Imports of installed
dependencies are distinguished from copied source. External checkpoints and
ImageNet have their own terms; no rights to those assets are granted here.

Author-contributed code/configurations/documentation still require a separate
licensing decision. Including these notices does not cure the BYOL converter
permission gap or constitute a blanket repository license.
```

</details>
