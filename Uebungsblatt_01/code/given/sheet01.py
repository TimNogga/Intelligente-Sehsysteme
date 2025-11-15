import cv2
import numpy as np

from os.path import *

def exercise2a(a: int, k: int, image_folder="."):
    image = cv2.imread(join(image_folder, "Lena_512x512.png"))

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            B, G, R = image[y, x]

            if k == 0:
                R = min(255, R * a)
            elif k == 1:
                G = min(255, G * a)
            elif k == 2:
                B = min(255, B * a)

            image[y, x] = (B, G, R)
    cv2.imshow("Ex. 2a)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    




            
                
            
    

    

def exercise2b(a: int, k: int, image_folder="."):

    image = cv2.imread(join(image_folder, "Lena_512x512.png"))

    B = image[..., 0].astype(np.int32)  # this needs to be int32 cause otherwise numpy minumum thinks that 200 x 2 = 144 and then takes the minmum after that, but we actually want it to take 255 (at least how i would interpret the exercise)
    G = image[..., 1].astype(np.int32)
    R = image[..., 2].astype(np.int32)
    if k == 0:
        R = np.minimum(255, R * a)
    elif k == 1:
        G = np.minimum(255, G * a)
    elif k == 2:
        B = np.minimum(255, B * a)
    image = np.stack([B, G, R], axis=-1).astype(np.uint8)
    
    cv2.imshow("Ex. 2b)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def exercise3(image_folder="."):
    
    image = cv2.imread(join(image_folder, "Lena_512x512.png"))
    mask = cv2.imread(join(image_folder, "Maske_Lena_512x512.png"))
    image = image * (mask //255)



    cv2.imshow("Ex. 3)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # ------------------
    # --- EXERCISE 2 ---
    # ------------------

    a = 2 # scale factor
    k = 1 # channel

    exercise2a(a=a, k=k)
    exercise2b(a=a, k=k)
    

    # ------------------
    # --- EXERCISE 3 ---
    # ------------------

    exercise3()





