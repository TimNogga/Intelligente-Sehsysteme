import numpy as np
import cv2
import sys

from pathlib import Path

from io_utils import load_image_grayscale
from sobel_TN import sobel_gradients, magnitude_and_orientation, linear_stretch_to_u8
from viz_TN import orientation_to_bin, render_orientation_bins, apply_magnitude_brightness, mask_by_threshold

MODES = ["magnitude", "orientation", "both"]

def on_change(_=None):
    pass # nothing to do here

def run_gui(img_path: str | Path):
    gray = load_image_grayscale(img_path).astype(np.float32)
    gx, gy = sobel_gradients(gray)
    M, O = magnitude_and_orientation(gx, gy)

    M_u8 = linear_stretch_to_u8(M.copy())
    theta_deg = np.degrees(O)
    bins = orientation_to_bin(theta_deg)
    ori_rgb = render_orientation_bins(bins)

    cv2.namedWindow("Sobel Viewer", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("mode (0:mag, 1:ori, 2:both)", "Sobel Viewer", 0, 2, on_change)
    cv2.createTrackbar("threshold", "Sobel Viewer", 0, 255, on_change)

    while True:
        mode_idx = cv2.getTrackbarPos("mode (0:mag, 1:ori, 2:both)", "Sobel Viewer")
        thr = cv2.getTrackbarPos("threshold", "Sobel Viewer")

        if mode_idx == 0:  
            vis = M_u8
        elif mode_idx == 1:  
            vis = mask_by_threshold(ori_rgb, M_u8, thr)
        else: 
            colored = apply_magnitude_brightness(ori_rgb, M_u8)
            vis = mask_by_threshold(colored, M_u8, thr)

        cv2.imshow("Sobel Viewer", vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')):
            break

    cv2.destroyAllWindows()
    #save a snapshot of the last view
    cv2.imwrite("sobel_gui_output.png", vis)

if __name__ == "__main__":
    run_gui(sys.argv[1] if len(sys.argv) > 1 else "Lena_512x512.png")
