import cv2
import numpy as np
from os.path import *


def median_filter(img: np.ndarray, mask: int = 3) -> np.ndarray:
    padding = mask // 2
    padded_image = np.pad(img, pad_width = padding, mode = 'edge')

    if img.ndim == 3:
        filtered_image = np.zeros_like(img)

        h, w, c = img.shape
        for channels in range(c):
            filtered_image[...,channels] = median_filter(img[...,channels], mask)
        return filtered_image
    else:
        filtered_image = np.zeros_like(img)
        h, w = img.shape
        for y in range(h):
            for x in range(w):
                median_window = padded_image[y:y + mask, x:x + mask]
                filtered_image[y,x] = np.median(median_window)

        

    return filtered_image

def diffusion_filter(
    i: np.ndarray,
    eps0: float = 1.0,
    iters: int = 500,
    lambda_: float = 1) -> np.ndarray:

    for _ in range(iters):
        grad_x = np.roll(i, -1, axis=0) - np.roll(i, 1, axis=0)
        grad_y = np.roll(i, -1, axis=1) - np.roll(i, 1, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        eps_u = eps0 * (lambda_**2) / (grad_mag**2 + lambda_**2)
        jx = -eps_u * grad_x
        jy = -eps_u * grad_y

        jx_grad = np.roll(jx, -1, axis=0) - np.roll(jx, 1, axis=0)
        jy_grad = np.roll(jy, -1, axis=1) - np.roll(jy, 1, axis=1)
        div_j = jx_grad + jy_grad

        i -= div_j  

    return i


def lin_scaling(img_diff: np.ndarray, img_median: np.ndarray) -> np.ndarray:
    difference = img_diff - img_median
    min_diff = np.min(difference)
    max_diff = np.max(difference)
    lin_scaled_image = np.zeros_like(difference, dtype=np.uint8)
    for y in range(difference.shape[0]):
        for x in range(difference.shape[1]):
            lin_scaled_image[y,x] = (difference[y,x] - min_diff)/(max_diff - min_diff) * 255
    return lin_scaled_image


def exercise1(image_folder="."):
    image = cv2.imread(join(image_folder, "Testbild_Lena_512x512.png"), cv2.IMREAD_COLOR)
    image_2 = cv2.imread(join(image_folder, "Testbild_Rauschen_640x480.png"), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("image wasnt found")
        return


    mask = 7
    mask_2 = 7

    out_img = median_filter(image, mask=mask)
    out_img_2 = median_filter(image_2, mask=mask_2)



    cv2.imshow("Ex. 1) Lena", out_img)
    cv2.imshow("Ex. 1) Rauschen", out_img_2)
    cv2.waitKey(0)

    saved_1 = cv2.imwrite("ex1_lena.png", out_img)
    saved_2 = cv2.imwrite("ex1_rauschen.png", out_img_2)
    print("Gespeichert:", saved_1, saved_2)
    cv2.destroyAllWindows()


def exercise2(image_folder="."):
    try:
        image = cv2.imread(join(image_folder, "Testbild_Rauschen_640x480.png"), cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return
    img_arr = np.asarray(image, dtype=np.float32) / 255.0
    out_img = diffusion_filter(i=img_arr, eps0=1.0, iters=500, lambda_=1) #500 is very high was it a typo or was it supposed to show that it is a gaussian filter with high iterations?
    cv2.imshow("Ex. 2) Rauschen", out_img)
    cv2.waitKey(0)
    out_u8 = (out_img * 255).astype(np.uint8) #it was cast to [0,1] but then the pixels are obv all close to black so we need this line. 
    saved_1 = cv2.imwrite("ex2_rauschen_.png", out_u8) 
    print("Gespeichert:", saved_1)
    cv2.destroyAllWindows()


def exercise3(image_folder="."):
    try:
        image_diffusion = cv2.imread(join(image_folder, "Testbild_Rauschen_640x480_Diffusion.png"), cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return
    try:
        image_median = cv2.imread(join(image_folder, "Testbild_Rauschen_640x480_Median.png"), cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return


    out_img = lin_scaling(image_diffusion, image_median)
    cv2.imshow("Ex. 3) Lineare Streckung", out_img)
    cv2.waitKey(0)
    saved_1 = cv2.imwrite("ex3_streckung.png", out_img)
    print("Gespeichert:", saved_1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    source_images = "." #path was wrong

    # ------------------
    # --- EXERCISE 1 ---
    # ------------------

    #exercise1(image_folder=source_images) 

    # ------------------
    # --- EXERCISE 2 ---
    # ------------------

    #exercise2(image_folder=source_images)

    # ------------------
    # --- EXERCISE 3 ---
    # ------------------

    exercise3(image_folder=source_images)
