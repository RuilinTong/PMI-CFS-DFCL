# Code implementation of ResNet-based continual learning

## Baseline implementation
- This implementation is referred from [R-DFCIL](https://github.com/jianzhangcs/R-DFCIL) and [ABD](https://github.com/GT-RIPL/AlwaysBeDreaming-DFCIL).

## ResNet-based CL experiments
- To reimplement ResNet-based continual learning experiments, please run scripts: `bash ./scripts/task_cont_{dataset}_{n}_task.sh`.
- `{dataset}` can be `cifar100` for CIFAR-100 and `tiny` for Tiny-ImageNet.
- `{n}` is number of tasks in continual learning, can be 5, 10 and 20.
