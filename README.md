# RapidLiDAR

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2501.03793-b31b1b.svg)](https://arxiv.org/abs/XXX) -->

Single-pass LiDAR scene completion. Given a partial point cloud `X`, RapidLiDAR predicts the
completed scene `P` end-to-end in a single forward pass, combining:

- **Multi-Scale Feature Extraction** — voxelizes `X` and extracts multi-scale 3D voxel features
  and a dense 2D BEV feature map via a dedicated BEV head with self-attention.
- **Adaptive Initialization Module** — predicts a spatially varying displacement for an expanded
  version of `X` to obtain a coarse initialized scene.
- **Multi-Scale Reconstruction Module** — refines the initialized scene using multi-scale
  deformable attention between per-point features and multi-scale BEV feature maps.
- **Refinement Network** — an optional second-stage network trained on top of the frozen
  completion model that upsamples the scene by a factor of `kappa`.

## Installation

Tested on **CUDA 12.4** with **Python 3.10**, **PyTorch 2.4.1**, **torchvision 0.19.1**, and
**mmcv 2.2.0**. Adjust the PyTorch/mmcv install commands below to match your own CUDA version.

```bash
# 1. Create and activate the conda environment
conda create -n rapidlidar python=3.10 -y
conda activate rapidlidar

# 2. Install PyTorch + torchvision for your CUDA version
#    (pick the right command for your setup at https://pytorch.org/get-started/locally/)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Clone the repo and its ChamferDistancePytorch submodule
git clone --recursive <repo_url> rapidlidar
cd rapidlidar
# if already cloned without --recursive:
git submodule update --init --recursive

# 4. Install the remaining Python dependencies
pip install -r requirements.txt

# 5. Install MultiScaleDeformableAttention (mmcv, built with CUDA ops) via OpenMMLab's mim
#    installer, following the official guide:
#    https://mmcv.readthedocs.io/en/latest/get_started/installation.html
pip install -U openmim
mim install mmcv==2.2.0

# 6. Build the Chamfer distance CUDA extension
pip install -e ChamferDistancePytorch/chamfer3D
```

`CUDA_HOME` must point at a valid CUDA toolkit install (matching the CUDA version PyTorch was
built for) so step 6 can JIT-compile the Chamfer distance extension, e.g.
`export CUDA_HOME=/usr/local/cuda-12.4`.

Once installed, see [Data](#data) to prepare the training pairs and [Training](#training) /
[Checkpoints](#checkpoints) to train or run inference with a trained model.

## Data

For downloading and preparing the raw SemanticKITTI dataset, please follow the instructions in the
[LiDiff repository](https://github.com/PRBonn/LiDiff). You can use the `rapidlidar` conda environment
described in [Installation](#installation) instead of installing the LiDiff dependencies.

Our data loader (`rapidlidar/data/dataset.py`) expects a SemanticKITTI-style directory with
`<sequence>/gt/*.npy` (complete scene) and `<sequence>/input/*.npy` (partial scan) pairs of shape
`(N, 3)`.

To generate the processed `gt`/`input` pairs for faster training, run:

```bash
python tools/generate_gtinput_nomink.py --config_path configs/rapidlidar.yaml --seqs 00 --seqs 08
```

This script is adapted from [LiNeXt](https://github.com/shuaiqianende/LiNeXt-main/blob/main/LiNeXt-main/linext/generate_gtandinput.py) and has been
ported to work without the MinkowskiEngine dependency. It requires the raw SemanticKITTI
`velodyne/*.bin` scans, `calib.txt`, `poses.txt`, and an aggregated `map_clean.npy` per sequence.

Omit `--seqs` to process every sequence listed under `data.train` / `data.validation` in the config.

Finally, please update the paths in `configs/rapidlidar.yaml` / `configs/rapidlidar_refine.yaml` to
point to the downloaded and processed data.


## Training

```bash
# coarse model training
python train.py --config_path configs/rapidlidar.yaml --experiment rapidlidar
# refinement model training
python train_refine.py --config_path configs/rapidlidar_refine.yaml \
    --base_model_checkpoint ckpt_path
```

## Checkpoints

| Model | Checkpoint |
|---|---|
| RapidLiDAR (coarse) | [download](https://drive.google.com/file/d/1GaJFQAN7beHT5KQH9q07A8pPWZErgWA1/view?usp=sharing) |
<!-- | Refinement Network | [TODO: add link] | -->

## Evaluation

Run the completion pipeline on the full SemanticKITTI validation sequence (`08`):

```bash
python tools/completion_pipeline.py -c checkpoints/rapidlidar_vox_0.3_best.pth -p /path/to/sequence/08
```

- `-c`/`--coarse_ckpt`: path to the coarse RapidLiDAR checkpoint.
- `-p`/`--path_scan`: KITTI sequence directory containing `velodyne`, `poses.txt`, `calib.txt`, and `map_clean.npy`.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{Hussian2026TowardsRT,
  title={Towards Real-Time and Adaptable LiDAR Scene Completion},
  author={Azhar Hussian and Martin Vossiek and Vasileios Belagiannis},
  booktitle={European Conference on Computer Vision},
  year={2026}
}
```

## Acknowledgments

This work builds on from the following open-source projects:

- [LiDiff](https://github.com/PRBonn/LiDiff)
- [ScoreLiDAR](https://github.com/happyw1nd/ScoreLiDAR)
- [LiNeXt](https://github.com/shuaiqianende/LiNeXt-main)
