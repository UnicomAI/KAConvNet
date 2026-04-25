# KAConvNet

This repository contains the source code for KAConvNet experiments. Data files, training outputs, checkpoints, logs, and other large runtime artifacts are intentionally excluded from this code-only copy.

## Paper

KAConvNet: Kolmogorov–Arnold convolutional networks for vision recognition.

If you use this code, please cite:

```bibtex
@article{Liu_2026_KAConvNet,
  title = {KAConvNet: Kolmogorov–Arnold convolutional networks for vision recognition},
  author = {Liu, Zhaoxiang and Ma, Zhicheng and Zhao, Kaikai and Wang, Kai and Lian, Shiguo},
  journal = {Image and Vision Computing},
  volume = {170},
  pages = {105983},
  year = {2026},
  doi = {10.1016/j.imavis.2026.105983},
  url = {https://doi.org/10.1016/j.imavis.2026.105983}
}
```

## Code Layout

- `train.py`: ImageNet training entry point.
- `KAConvNet.py`: Main KAConvNet model implementation.
- `KAConvNet_v2.py`: Alternate KAConvNet implementation.
- `ConvNet.py`: Baseline ConvNet implementation.
- `datasets.py`: Dataset construction utilities.
- `engine.py`: Train and evaluation loops.
- `optim_factory.py`: Optimizer and layer decay helpers.
- `utils.py`: Distributed training, logging, checkpoint, and metric utilities.
- `command.txt`: Historical training commands from the handover.

## Dependencies

The code imports these main Python packages:

- `torch`
- `torchvision`
- `timm`
- `tensorboardX`
- `Pillow`
- `numpy`
- `einops`

Exact package versions were not included in the handover and should be recovered from the original server environment when strict reproduction is required.

## Data And Outputs

This code-only copy does not include ImageNet data or training results. Training commands expect ImageNet-style data under:

```text
datasets/ImageNet1K
```

Training outputs are written to `trainresults/` by the historical commands. Keep datasets, checkpoints, logs, and generated training outputs out of git.

## Example Training Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use-env train.py --batch_size 256 --lr 2e-3 --model_ema true --model_ema_eval true --data_path datasets/ImageNet1K --warmup_epochs 5 --epochs 300 --output_dir trainresults/kaconvnet-B-5-01
```

More historical commands are kept in `command.txt`.
