import numpy as np


def rotate_point_cloud(points: np.ndarray) -> np.ndarray:
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval, sinval = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
    return np.dot(points.reshape((-1, 3)), rotation_matrix).reshape(points.shape)


def rotate_perturbation_point_cloud(points: np.ndarray, angle_sigma=0.06, angle_clip=0.18) -> np.ndarray:
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return np.dot(points.reshape((-1, 3)), R).reshape(points.shape)


def random_scale_point_cloud(points: np.ndarray, scale_low=0.95, scale_high=1.05) -> np.ndarray:
    scale = np.random.uniform(scale_low, scale_high)
    return points * scale


def random_flip_point_cloud(points: np.ndarray) -> np.ndarray:
    if np.random.random() > 0.5:
        points = points.copy()
        points[..., 1] = -points[..., 1]
    return points
