# MEG-XL

[![arXiv](https://img.shields.io/badge/arXiv-2602.02494-b31b1b.svg)](https://arxiv.org/abs/2602.02494)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/pnpl/MEG-XL)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**MEG-XL** is a pre-trained model for non-invasive electrophysiological brain signals (MEG/EEG). It uses long-context pre-training on MEG to learn contextualised transferable representations, enabling data-efficient fine-tuning for neural decoding. We find that MEG-XL achieves state-of-the-art brain-to-text word decoding accuracy while requiring significantly less downstream data than prior approaches.

**Paper:** [arXiv:2602.02494](https://arxiv.org/abs/2602.02494) | **Model weights:** [HuggingFace](https://huggingface.co/pnpl/MEG-XL)

If you find this work helpful in your research, please cite the paper:
```
@article{jayalath2026megxl,
  title={{MEG-XL}: Data-Efficient Brain-to-Text via Long-Context Pre-Training},
  author={Jayalath, Dulhan and Parker Jones, Oiwi},
  journal={arXiv preprint arXiv:2602.02494},
  year={2026}
}
```

## Table of Contents
- [Requirements](#requirements)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Fine-tuning](#fine-tuning-meg-xl-for-brain-to-text)
- [Linear Probing](#linear-probing-meg-xl-for-brain-to-text)
- [Pre-training](#pre-training-meg-xl)
- [Supported Datasets](#supported-datasets)

## Requirements
- python >= 3.12
- For python packages, see `requirements.txt`

## Setup

### Tokenizer (BioCodec) Setup
The code for the pre-trained tokenizer is currently only available by request from the authors. Until they release it publicly, please:
1. Request access to code and checkpoints for [BioCodec](https://arxiv.org/abs/2510.09095) through the authors' email
2. Place their code in `brainstorm/neuro_tokenizers/biocodec` and the checkpoint in `brainstorm/neuro_tokenizers/biocodec_ckpt.pt`

### MEG-XL Setup
3. (Optional) create and activate a new virtual environment with python >= 3.12, e.g. `conda create -n megxlenv python=3.12.12 && conda activate megxlenv`
4. `pip install -r requirements.txt`
5. (Optional) download pre-trained MEG-XL weights from [HuggingFace](https://huggingface.co/pnpl/MEG-XL)
6. Follow the specific notes below depending on how you wish to use MEG-XL

## Quick Start

```python
import torch
from brainstorm.neuro_tokenizers.biocodec.model import BioCodecModel
from brainstorm.models.criss_cross_transformer import CrissCrossTransformerModule

# Load tokenizer
tokenizer = BioCodecModel._get_optimized_model()
ckpt = torch.load("brainstorm/neuro_tokenizers/biocodec_ckpt.pt", map_location="cuda")
tokenizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()})
tokenizer.eval()

# Load MEG-XL
ckpt = torch.load("path/to/megxl_checkpoint.ckpt", map_location="cuda")
model = CrissCrossTransformerModule(tokenizer=tokenizer, **ckpt["hyper_parameters"])
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval()

# Prepare inputs (shapes for 150s segment at 50Hz with 306 MEG channels)
# meg: [batch, channels, time] - raw MEG signal
# sensor_xyz: [batch, channels, 3] - sensor positions (normalized)
# sensor_abc: [batch, channels, 3] - sensor orientations
# sensor_types: [batch, channels] - 0=gradiometer, 1=magnetometer
# sensor_mask: [batch, channels] - 1=valid sensor, 0=padding

# Forward pass (apply_mask=False for inference)
with torch.no_grad():
    output = model(meg, sensor_xyz, sensor_abc, sensor_types, sensor_mask, apply_mask=False)
    features = output["features"]  # [batch, channels, time_tokens, hidden_dim]
```

## Project Structure

```
MEG-XL/
├── configs/                         # Hydra YAML configs for training and evaluation
│
└── brainstorm/
    ├── train_criss_cross_multi.py                                # Multi-dataset pre-training script
    ├── evaluate_criss_cross_word_classification.py               # Word classification eval with fine-tuning
    ├── evaluate_criss_cross_word_classification_linear_probe.py  # Word classification eval with frozen backbone
    │
    ├── data/
    │   ├── utils.py                     # Sensor position normalization utilities
    │   ├── preprocessing.py             # MEG preprocessing (filtering, resampling, caching)
    │   ├── samplers.py                  # Recording-level shuffle sampler for efficient I/O
    │   ├── lightning_datamodule.py      # PyTorch Lightning DataModule for single dataset
    │   ├── multi_datamodule.py          # DataModule for multi-dataset pre-training
    │   ├── multi_dataset.py             # Wrapper combining multiple MEG datasets
    │   ├── subsampled_dataset.py        # Wrapper for recording subsampling with sampler compat
    │   ├── *_dataset.py                 # Per-corpus dataset implementations
    │   └── *_word_aligned_dataset.py    # Per-corpus word-aligned segment datasets
    │
    ├── models/
    │   ├── criss_cross_transformer.py  # Main model with temporal masking and RVQ prediction
    │   ├── spatial_attention.py        # Gaussian Fourier embeddings for 3D sensor positions
    │   └── attentional/                # Spatial-temporal attention modules
    │
    ├── losses/
    │   └── contrastive.py  # CLIP-style contrastive loss
    │
    └── neuro_tokenizers/
        ├── biocodec_ckpt.pt  # Pre-trained BioCodec checkpoint
        └── biocodec/         # Neural signal tokenizer with RVQ
```

## Fine-tuning MEG-XL for Brain-to-Text

```bash
python -m brainstorm.evaluate_criss_cross_word_classification \
    --config-name=eval_criss_cross_word_classification_{armeni,gwilliams,libribrain} \
    model.criss_cross_checkpoint=/path/to/your/checkpoint.ckpt
```

**Notes:**
- Requires 1 GPU with >= 80GB VRAM (disable activation checkpointing for faster training if more is available)
- Download the dataset and update the path in `configs/eval_criss_cross_word_classification_{armeni,gwilliams,libribrain}.yaml`
- For unsupported datasets, implement a word-aligned data loader following `brainstorm/data/armeni_word_aligned_dataset.py`

## Linear Probing MEG-XL for Brain-to-Text

```bash
python -m brainstorm.evaluate_criss_cross_word_classification_linear_probe \
    --config-name=eval_criss_cross_word_classification_linear_probe_{armeni,gwilliams,libribrain} \
    model.criss_cross_checkpoint=/path/to/your/checkpoint.ckpt
```

**Notes:**
- Requires 1 GPU with >= 40GB VRAM
- See fine-tuning notes above for dataset setup

## Pre-training MEG-XL

```bash
python brainstorm/train_criss_cross_multi.py \
    --config-name=train_criss_cross_multi_50hz_med
```

**Notes:**
- Requires 1 GPU with >= 80GB VRAM (disable activation checkpointing for faster training if more is available)
- Download the pre-training datasets and update paths in `configs/train_criss_cross_multi_50hz_med.yaml`

## Supported Datasets

| Split | Dataset | Link |
|-------|---------|------|
| Pre-training | CamCAN | [mrc-cbu.cam.ac.uk](https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/) |
| Pre-training | MOUS | [data.ru.nl](https://data.ru.nl/collections/di/dccn/DSC_3011020.09_236) |
| Pre-training | SMN4Lang | [OpenNeuro](https://openneuro.org/datasets/ds004078) |
| Fine-tuning | MEG-MASC | [OSF](https://osf.io/ag3kj/) |
| Fine-tuning | Armeni | [data.ru.nl](https://data.ru.nl/collections/di/dccn/DSC_3011085.05_995) |
| Fine-tuning | LibriBrain | [HuggingFace](https://huggingface.co/datasets/pnpl/LibriBrain) |
