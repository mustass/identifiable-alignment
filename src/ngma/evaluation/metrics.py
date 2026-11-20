import numpy as np


def rotation_error(R_pred: np.ndarray, R_star: np.ndarray) -> float:
    """Geodesic distance in SO(3) between two rotation matrices: the angle (radians) of the
    residual rotation R_pred^T R_star."""
    R_res = R_pred.T @ R_star
    cos_angle = (np.trace(R_res) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))
