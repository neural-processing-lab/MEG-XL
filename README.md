# MEG-<ins>XL</ins>: Data-Efficient Brain-to-Text via Long-Context Pre-Training

This repository contains the code for the paper "MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training."

Paper: [arXiv](https://arxiv.org/abs/2602.02494)
Model weights: [HuggingFace](https://huggingface.co/pnpl/MEG-XL)

If you find this work helpful in your research, please cite the paper:
```
@article{jayalath2026megxl,
  title={{MEG-XL}: Data-Efficient Brain-to-Text via Long-Context Pre-Training},
  author={Jayalath, Dulhan and Parker Jones, Oiwi},
  journal={arXiv preprint arXiv:2602.02494},
  year={2026}
}
```

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

## Fine-tuning MEG-XL for Brain-to-Text
`python -m brainstorm.evaluate_criss_cross_word_classification --config-name=eval_criss_cross_word_classification_{armeni, gwilliams, libribrain} model.criss_cross_checkpoint=/path/to/your/checkpoint.ckpt`

Notes:
1. You will need at least 1 GPU with >= 80GiB of GPU VRAM. If you have more than this, you can turn off activation checkpointing, which is used to save GPU memory, and train faster.
2. If using one of MEG-MASC/Armeni/LibriBrain datasets, you will need to download the dataset from the links at the end of the README and adjust the path in `configs/eval_criss_cross_word_classification_{armeni, gwilliams, libribrain}.yaml` to point to the correct location.
3. If using an unsupported dataset, you will need to implement your own word aligned data loader. Follow the structure of `brainstorm/data/armeni_word_aligned_dataset.py`.

## Linear probing MEG-XL
`python -m brainstorm.evaluate_criss_cross_word_classification --config-name=eval_criss_cross_word_classification_linear_probe_{armeni, gwilliams, libribrain} model.criss_cross_checkpoint=/path/to/your/checkpoint.ckpt`

Notes: see the notes above for fine-tuning MEG-XL. You may use 1 GPU with >= 40GiB of GPU VRAM.

## Pre-training MEG-XL
`python brainstorm/train_criss_cross_multi.py --config-name=train_criss_cross_multi_50hz_med`

Notes:
1. Pre-training requires at least 1 GPU with >= 80GiB of GPU VRAM. If you have more than this, you can turn off activation checkpointing, which is used to save GPU memory, and train faster.
2. You will need to download the pre-training datasets linked at the end of the README file and adjust the paths to point to their location in `configs/train_criss_cross_multi_50hz_med`.

## Supported datasets
For posterity, the datasets used in the paper are as follows:
- Pre-training datasets:
    - [CamCAN](https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/)
    - [MOUS](https://data.ru.nl/collections/di/dccn/DSC_3011020.09_236)
    - [SMN4Lang](https://openneuro.org/datasets/ds004078)
- Fine-tuning datasets:
    - [MEG-MASC](https://osf.io/ag3kj/)
    - [Armeni](https://data.ru.nl/collections/di/dccn/DSC_3011085.05_995)
    - [LibriBrain](https://huggingface.co/datasets/pnpl/LibriBrain)
