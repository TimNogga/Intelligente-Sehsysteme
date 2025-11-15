import numpy as np
import cv2

# four bins centered at 0°, 45°, 90°, 135° with +-22.5° half-width
BIN_CENTERS = np.array([0.0, 45.0, 90.0, 135.0], dtype=np.float32)
HALF_WIDTH = 22.5

# fixed BGR colors per bin (distinct and legible)
BIN_COLORS = np.array([
    [  0,   0, 255],  # red
    [255,   0,   0],  # blue
    [  0, 255,   0],  # green
    [255, 255,   0],  # cyan
], dtype=np.uint8)

def orientation_to_bin(theta_deg: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta_deg, dtype=np.float32) % 180.0
    diffs = np.abs(((theta[..., None] - BIN_CENTERS[None, :]) + 90.0) % 180.0 - 90.0)
    return diffs.argmin(axis=-1).astype(np.int32)

def render_orientation_bins(bin_idx: np.ndarray) -> np.ndarray:
    return BIN_COLORS[bin_idx]

def apply_magnitude_brightness(rgb: np.ndarray, mag_u8: np.ndarray) -> np.ndarray:
    s = (mag_u8.astype(np.float32) / 255.0)[..., None]
    out = rgb.astype(np.float32) * s
    return np.clip(out, 0, 255).astype(np.uint8)

def mask_by_threshold(rgb: np.ndarray, mag: np.ndarray, thr: float) -> np.ndarray:
    mask = (mag >= thr).astype(np.uint8)[..., None]
    return rgb * mask
