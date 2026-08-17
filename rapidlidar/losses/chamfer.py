import torch

from ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist

_compute_cd = chamfer_3DDist()


def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor):
    p1 = p1.float()
    p2 = p2.float()
    d1, d2, _, _ = _compute_cd(p1, p2)
    return d1, d2


def sqrt_chamfer_distance(pred: torch.Tensor, gt: torch.Tensor, sqrt: bool = True) -> torch.Tensor:
    d1, d2 = chamfer_distance(pred, gt)
    if sqrt:
        return (torch.sqrt(d1).mean() + torch.sqrt(d2).mean()) / 2.0
    return (d1.mean() + d2.mean()) / 2.0
