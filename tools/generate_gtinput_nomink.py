import os
import sys

import click
import numpy as np
import torch
import yaml
from natsort import natsorted

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.completion_pipeline import get_ground_truth, load_pcd, load_poses


def efficient_voxel_downsample(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    voxel_indices = torch.floor(points / voxel_size).to(torch.int64)
    _, inverse, counts = torch.unique(voxel_indices, dim=0, return_inverse=True, return_counts=True)
    _, order = torch.sort(inverse)
    repr_ids = order[counts.cumsum(0) - 1]
    return points[repr_ids]


def auto_voxel_downsample(points: torch.Tensor, target: int, init_vs: float = 0.2, tol: int = 1000,
                           vs_min: float = 0.05, vs_max: float = 2.0, max_iters: int = 20):
    lo, hi = vs_min, vs_max
    vs = init_vs
    down = points
    for _ in range(max_iters):
        down = efficient_voxel_downsample(points, vs)
        diff = down.shape[0] - target
        if 0 <= diff < tol:
            return down
        if diff >= tol:
            lo = vs
        else:
            hi = vs
        vs = 0.5 * (lo + hi)
    return down


def to_target_count(points: np.ndarray, target: int, device: torch.device) -> np.ndarray:
    points_t = torch.as_tensor(points, dtype=torch.float32, device=device)
    down = auto_voxel_downsample(points_t, target)

    if down.shape[0] > target:
        idx = torch.randperm(down.shape[0], device=device)[:target]
        down = down[idx]
    elif down.shape[0] < target:
        pad_idx = torch.randint(0, down.shape[0], (target - down.shape[0],), device=device)
        down = torch.cat([down, down[pad_idx]], dim=0)

    return down.cpu().numpy().astype(np.float32)


@click.command()
@click.option('--config_path', type=str, default='configs/rapidlidar.yaml', help='config used to resolve data_dir / num_points')
@click.option('--data_dir', '-d', type=str, default=None, help="root sequences directory (default: config's data_dir)")
@click.option('--seqs', '-s', multiple=True, default=None, help='sequences to process (default: train + validation from config)')
@click.option('--max_range', '-m', type=float, default=50.0, help='max range (in meters) to keep points within')
@click.option('--up_factor', type=int, default=10, help='ratio between the number of gt and input points')
@click.option('--device', type=str, default=None)
@click.option('--overwrite', is_flag=True, default=False, help='overwrite existing gt/input files')
def main(config_path, data_dir, seqs, max_range, up_factor, device, overwrite):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)['data']

    data_dir = os.path.expanduser(data_dir or cfg['data_dir'])
    seqs = list(seqs) if seqs else (cfg['train'] + cfg['validation'])
    num_points = cfg['num_points']
    input_points = max(1, num_points // up_factor)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    for seq in seqs:
        seq_path = os.path.join(data_dir, seq)
        velo_dir = os.path.join(seq_path, 'velodyne')
        if not os.path.isdir(velo_dir):
            print(f"skipping sequence '{seq}': no velodyne directory found at {velo_dir}")
            continue

        poses = load_poses(os.path.join(seq_path, 'calib.txt'), os.path.join(seq_path, 'poses.txt'))
        seq_map = np.load(os.path.join(seq_path, 'map_clean.npy'))

        gt_dir = os.path.join(seq_path, 'gt')
        input_dir = os.path.join(seq_path, 'input')
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(input_dir, exist_ok=True)

        bin_files = [f for f in natsorted(os.listdir(velo_dir)) if f.endswith('.bin')]
        for frame_idx, fname in enumerate(bin_files):
            frame_id = fname.replace('.bin', '')
            gt_path = os.path.join(gt_dir, f'{frame_id}.npy')
            input_path = os.path.join(input_dir, f'{frame_id}.npy')
            if not overwrite and os.path.isfile(gt_path) and os.path.isfile(input_path):
                continue

            scan = load_pcd(os.path.join(velo_dir, fname))
            dist = np.sqrt(np.sum(scan[:, :3] ** 2, axis=-1))
            scan_ranged = scan[(dist < max_range) & (dist > 3.5)][:, :3]

            p_part = to_target_count(scan_ranged, input_points, device)

            pcd_gt = get_ground_truth(poses[frame_idx], scan_ranged.astype(np.float64), seq_map, max_range)
            p_full = to_target_count(np.asarray(pcd_gt.points), num_points, device)

            np.save(input_path, p_part)
            np.save(gt_path, p_full)

            print(f'[{seq}] {frame_idx + 1}/{len(bin_files)}  gt={p_full.shape[0]}  input={p_part.shape[0]}')


if __name__ == '__main__':
    main()
