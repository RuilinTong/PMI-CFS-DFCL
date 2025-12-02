# Code implementation of CLIP-based continual learning

## Baseline implementation
- This implementation is based on [LAMDA-PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT).
- The implementation of MoE-adapter is referred from [MoE-Adapters4CL](https://github.com/JiazuoYu/MoE-Adapters4CL)

## Data preparation
- This dataset requires all dataset to be located in `/data/Datasets/`.
- To change data location, please refer to `./utils/data.py`.

## CLIP-based CL experiments
- To reimplement CLIP-based continual learning experiments, please run scripts: `bash ./moe_adapter_{dataset}.sh`.
- `{dataset}` can be `cifar` for CIFAR-100, `inr` for ImageNet-R and 'cub' for CUB-200.
- Experient setting can be altered by changin files in `./configs/moe_adapter/`.

## Train model on synthetic new task data
- To reimplement CLIP-based continual learning experiments where training data of last task is synthesized by model inversion, please run scripts: `bash ./moe_openset_{dataset}.sh`.
- `{dataset}` can be `cifar` for CIFAR-100 and `inr` for ImageNet-R.
- Experient setting can be altered by changin files in `./configs/moe_openset/`.
