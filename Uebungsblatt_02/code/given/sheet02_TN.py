import cv2
import numpy as np

from os.path import *

def exercise1(image_folder=".", resize_factor=6):
    image = cv2.imread(join(image_folder, "Weeki_Wachee_spring_10079u.png"))

    image = image.astype(np.float32)
    gamma = float(input("Bitte geben Sie den Wert für gamma ein (z.B. 0.5 oder 2.2): "))
    I_min = np.min(image)
    I_max = np.max(image)
    N_G = I_max + 1
    gamma_corrected = np.round(N_G * ((image - I_min) / (I_max - I_min))**gamma + I_min)
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    image_out = cv2.resize(gamma_corrected, (gamma_corrected.shape[1]//resize_factor, gamma_corrected.shape[0]//resize_factor))
    image_in = cv2.resize(image.astype(np.uint8), (image.shape[1]//resize_factor, image.shape[0]//resize_factor))

    cv2.imshow("Ex. 1)", np.hstack([image_in, image_out]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
def filter_image(image: np.ndarray, kernel: np.ndarray):
    
    filtered_image = image.copy()
    height, width = filtered_image.shape[:2]
    output = np.zeros((height, width), dtype=np.float32)
    filtered_image = None

    kernel = kernel.astype(np.float32)
    if np.sum(kernel) != 0:
        kernel = kernel / np.sum(kernel)
    else:
        kernel = kernel


    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(image.astype(np.float32), ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect') # internet says reflect is king
    

    for y in range(height):
        for x in range(width):
            region = padded[y:y+kh, x:x+kw]
            output[y, x] = np.sum(region * kernel)

    filtered_image = np.clip(output, 0, 255).astype(np.uint8)
    return filtered_image


if __name__ == "__main__":
    
    image_folder = "code/given/"

    # ------------------
    # --- EXERCISE 1 ---
    # ------------------

    #exercise1(image_folder=image_folder)

    # ------------------
    # --- EXERCISE 5 ---
    # ------------------

    input_image = cv2.imread(join(image_folder, "Testbild_Werkzeuge_768x576.png"), cv2.IMREAD_GRAYSCALE)

    kernel_33 = np.ones((3, 3), dtype=np.float32) / 9
    kernel_55 = np.ones((5, 5), dtype=np.float32) / 25

    result_33 = filter_image(image=input_image, kernel=kernel_33)
    result_55 = filter_image(image=input_image, kernel=kernel_55)

    diff_33 = cv2.absdiff(input_image, result_33) # da ich literally die unterschiede nicht gesehen habe hier nochmal die diff geplotted
    diff_55 = cv2.absdiff(input_image, result_55)

    combined_main = np.hstack([input_image, result_33, result_55])
    combined_diff = np.hstack([diff_33, diff_55])

    cv2.imshow("Ex. 5) Original | 3x3 Mittelwert | 5x5 Mittelwert", combined_main)
    cv2.imshow("Ex. 5) Differenzbilder (3x3 | 5x5)", combined_diff)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
