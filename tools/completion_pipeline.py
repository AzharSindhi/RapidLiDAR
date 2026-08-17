import os
import random
import sys
import time

import click
import numpy as np
import open3d as o3d
import torch
import tqdm
import yaml
from natsort import natsorted

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rapidlidar.models.rapidlidar import RapidLiDAR
from rapidlidar.models.refinement import RefinementNetwork
from rapidlidar.utils.histogram_metrics import compute_hist_metrics
from rapidlidar.utils.metrics import ChamferDistance, CompletionIoU


completion_iou = CompletionIoU(voxel_sizes=[0.5, 0.2, 0.1])
chamfer_distance = ChamferDistance()



DEFAULT_PATH_SCAN = os.path.expanduser("~/Documents/datasets/SemanticKITTI/dataset/sequences/08")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RapidLiDARCompletion:
    def __init__(self, coarse_path, refine_path=None, device=None, up_factor=10, max_range=50.0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.up_factor = up_factor
        self.max_range = max_range

        coarse_ckpt = torch.load(coarse_path, map_location="cpu", weights_only=False)
        coarse_hparams = coarse_ckpt.get("hyper_parameters") or coarse_ckpt.get("hparams")
        self.model = RapidLiDAR(**dict(coarse_hparams))
        self.model.load_state_dict(coarse_ckpt["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"loaded coarse model ({count_parameters(self.model):,} trainable params)")

        self.use_refine = refine_path is not None
        if self.use_refine:
            refine_ckpt = torch.load(refine_path, map_location="cpu", weights_only=False)
            refine_hparams = refine_ckpt.get("hyper_parameters") or refine_ckpt.get("hparams")
            refine_hparams = dict(refine_hparams)
            refine_hparams["base_model_checkpoint"] = coarse_path
            self.model_refine = RefinementNetwork(**refine_hparams)
            self.model_refine.load_state_dict(refine_ckpt["state_dict"])
            self.model_refine = self.model_refine.to(self.device)
            self.model_refine.eval()
            print(f"loaded refine model ({count_parameters(self.model_refine):,} trainable params)")

        self.num_points = self.model.num_points
        self.hparams = {"data": {"max_range": max_range, "num_points": self.num_points}}
        self.cnt = 0

    def preprocess_scan(self, scan):
        dist = np.sqrt(np.sum((scan)**2, -1))
        scan = scan[(dist < self.hparams['data']['max_range']) & (dist > 3.5)][:,:3]

        # use farthest point sampling
        pcd_scan = o3d.geometry.PointCloud()
        pcd_scan.points = o3d.utility.Vector3dVector(scan)
        pcd_scan = pcd_scan.farthest_point_down_sample(int(self.hparams['data']['num_points'] / 10))
        scan = torch.tensor(np.array(pcd_scan.points), dtype=torch.float32, device=self.device)
        full_scan = scan.repeat(10,1)
        scan = scan[None,:,:]
        full_scan = full_scan[None,:,:]

        return scan, full_scan

    def postprocess_scan(self, completed_scan, input_scan):
        if isinstance(completed_scan, torch.Tensor):
            completed_scan = completed_scan.cpu().numpy()
        dist = np.sqrt(np.sum((completed_scan)**2, -1))
        post_scan = completed_scan[dist < self.hparams['data']['max_range']]
        max_z = input_scan[...,2].max().item()
        min_z = (input_scan[...,2].mean() - 2 * input_scan[...,2].std()).item()

        post_scan = post_scan[(post_scan[:,2] < max_z) & (post_scan[:,2] > min_z)]

        return post_scan

    def complete_scan(self, scan):
        x_part, x_full = self.preprocess_scan(scan)

        if self.use_refine:
            coarse_points, refined_points = self.forward_refine(x_part)
            post_scan = self.postprocess_scan(refined_points[0], x_part[0])
        else:
            coarse_points = self.forward_coarse(x_part)
            post_scan = self.postprocess_scan(coarse_points[0], x_part[0])

        self.cnt += 1
        return post_scan, x_part

    @torch.no_grad()
    def forward_refine(self, x_part):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        refined_points, coarse_points = self.model_refine(x_part)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"refine inference time: {time.time() - start_time:.4f}s")
        return coarse_points, refined_points

    @torch.no_grad()
    def forward_coarse(self, x_part):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        output = self.model(x_part, up_factor=self.up_factor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"coarse inference time: {time.time() - start_time:.4f}s")
        return output.points



def load_pcd(pcd_file):
    if pcd_file.endswith('.bin'):
        return np.fromfile(pcd_file, dtype=np.float32).reshape((-1,4))[:,:3]
    elif pcd_file.endswith('.ply') or pcd_file.endswith('.pcd'):
        return np.array(o3d.io.read_point_cloud(pcd_file).points)
    else:
        print(f"Point cloud format '.{pcd_file.split('.')[-1]}' not supported. (supported formats: .bin (kitti format), .ply)")


def parse_calibration(filename):
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib

def load_poses(calib_fname, poses_fname):
    if os.path.exists(calib_fname):
        calibration = parse_calibration(calib_fname)
        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

    poses_file = open(poses_fname)
    poses = []

    for line in poses_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        if os.path.exists(calib_fname):
            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        else:
            poses.append(pose)

    return poses

def get_ground_truth(pose, cur_scan, seq_map, max_range):
    trans = pose[:-1,-1]
    dist_gt = np.sum((seq_map - trans)**2, axis=-1)**.5
    scan_gt = seq_map[dist_gt < max_range]
    scan_gt = np.concatenate((scan_gt, np.ones((len(scan_gt),1))), axis=-1)
    scan_gt = (scan_gt @ np.linalg.inv(pose).T)[:,:3]
    scan_gt = scan_gt[(scan_gt[:,2] > -4.) & (scan_gt[:,2] < 4.4)]
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(scan_gt)

    # filter only over the view point
    cur_pcd = o3d.geometry.PointCloud()
    cur_pcd.points = o3d.utility.Vector3dVector(cur_scan)
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cur_pcd, voxel_size=10.)
    in_viewpoint = viewpoint_grid.check_if_included(pcd_gt.points)
    points_gt = np.array(pcd_gt.points)
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt[in_viewpoint])

    return pcd_gt


@click.command()
@click.option('--coarse_ckpt', '-c', type=str, required=True, help='coarse (RapidLiDAR) checkpoint')
@click.option('--refine_ckpt', '-r', type=str, default=None, help='refinement network checkpoint (optional)')
@click.option('--path_scan', '-p', type=str, default=DEFAULT_PATH_SCAN, help=f"directory of input scans (.bin/.ply/.pcd), or a KITTI sequence directory when --use_eval is set (default: {DEFAULT_PATH_SCAN}'s validation sequence)")
@click.option('--use_eval/--no_use_eval', '-u/-U', default=True, help='evaluate against the KITTI aggregated map instead of just writing completions')
@click.option('--up_factor', type=int, default=10, help='upsampling factor used by the coarse model')
@click.option('--max_range', '-m', type=float, default=50.0, help='max range (in meters) to keep points within')
@click.option('--shuffle', is_flag=True, default=False, help='shuffle the scans before processing')
@click.option('--limit', type=int, default=None, help='limit the number of scans processed (default: all)')
def main(coarse_ckpt, refine_ckpt, path_scan, use_eval, up_factor, max_range, shuffle, limit):

    lidar_completion = RapidLiDARCompletion(
        coarse_ckpt, refine_ckpt, up_factor=up_factor, max_range=max_range,
    )
    if use_eval:
        poses = load_poses(os.path.join(path_scan, 'calib.txt'), os.path.join(path_scan, 'poses.txt'))
        seq_map = np.load(f'{path_scan}/map_clean.npy')

        jsd_3d = []
        jsd_bev = []

        # only .bin files inside velodyne correspond to poses.txt entries; other files (e.g.
        # labels, indices) would otherwise misalign the (pose, scan) correspondence.
        scan_files = natsorted(f for f in os.listdir(f'{path_scan}/velodyne') if f.endswith('.bin'))
        samples = list(zip(poses, scan_files))
        if shuffle:
            random.shuffle(samples)
        if limit is not None:
            samples = samples[:limit]

        for pose, scan_path in tqdm.tqdm(samples):
            pcd_file = os.path.join(path_scan, 'velodyne', scan_path)
            points = load_pcd(pcd_file)
            complete_point, scan = lidar_completion.complete_scan(points)
            complete_scan = o3d.geometry.PointCloud()
            complete_scan.points = o3d.utility.Vector3dVector(complete_point)
            scan_np = scan.squeeze(0).cpu().numpy().astype(np.float64)  # (N, 3)
            pcd_gt = get_ground_truth(pose, scan_np, seq_map, max_range)
            jsd_3d.append(compute_hist_metrics(pcd_gt, complete_scan, bev=False))
            jsd_bev.append(compute_hist_metrics(pcd_gt, complete_scan, bev=True))
            print(f'JSD 3D: {np.array(jsd_3d).mean()}')
            print(f'JSD BEV: {np.array(jsd_bev).mean()}')
            completion_iou.update(pcd_gt, complete_scan)
            thr_ious = completion_iou.compute()
            for v_size in thr_ious.keys():
                print(f'Voxel {v_size}cm IOU: {thr_ious[v_size]}')
            chamfer_distance.update(pcd_gt, complete_scan)
            cd_mean, cd_std = chamfer_distance.compute()
            print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        print('\n\n=================== FINAL RESULTS ===================\n\n')
        print(f'JSD 3D: {np.array(jsd_3d).mean()}')
        print(f'JSD BEV: {np.array(jsd_bev).mean()}')
        thr_ious = completion_iou.compute()
        for v_size in thr_ious.keys():
            print(f'Voxel {v_size}cm IOU: {thr_ious[v_size]}')
        cd_mean, cd_std = chamfer_distance.compute()
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
    else:
        os.makedirs(f'{path_scan}/results', exist_ok=True)
        scan_paths = natsorted(
            f for f in os.listdir(path_scan)
            if f.endswith(".bin") or f.endswith(".ply") or f.endswith(".pcd")
        )
        if shuffle:
            random.shuffle(scan_paths)
        if limit is not None:
            scan_paths = scan_paths[:limit]

        for pcd_path in tqdm.tqdm(scan_paths):
            pcd_file = os.path.join(path_scan, pcd_path)
            points = load_pcd(pcd_file)
            complete_point, scan = lidar_completion.complete_scan(points)

            pcd_complete = o3d.geometry.PointCloud()
            pcd_complete.points = o3d.utility.Vector3dVector(complete_point)
            pcd_complete.estimate_normals()
            o3d.io.write_point_cloud(f'{path_scan}/results/{pcd_path.split(".")[0]}.ply', pcd_complete)

if __name__ == '__main__':
    main()
