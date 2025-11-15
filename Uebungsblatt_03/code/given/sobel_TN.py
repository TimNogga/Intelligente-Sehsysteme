import numpy as np
import cv2

SOBEL_X = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]], dtype=np.float32)

SOBEL_Y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]], dtype=np.float32)

def linear_stretch_to_u8(img: np.ndarray) -> np.ndarray:
    img_min = np.min(img)
    img_max = np.max(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[y, x] = (img[y, x] - img_min) / (img_max - img_min) * 255
    return img.astype(np.uint8)

def sobel_gradients(gray_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    G_x = cv2.filter2D(gray_f32, -1, SOBEL_X)
    G_y = cv2.filter2D(gray_f32, -1, SOBEL_Y)
    return G_x, G_y

def magnitude_and_orientation(gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    M = np.sqrt(gx**2 + gy**2)
   
    O = np.arctan2(gy, gx)

    return M, O



def main():

    lena_gray = cv2.imread("Lena_512x512.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    G_x, G_y = sobel_gradients(lena_gray)
    M, O = magnitude_and_orientation(G_x, G_y)
    M_u8 = linear_stretch_to_u8(M.copy())
    O_u8 = linear_stretch_to_u8(O.copy())

    img2_gray = cv2.imread("Testbild_Werkzeuge_768x576.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    G2_x, G2_y = sobel_gradients(img2_gray)
    M2, O2 = magnitude_and_orientation(G2_x, G2_y)
    M2_u8 = linear_stretch_to_u8(M2.copy())
    O2_u8 = linear_stretch_to_u8(O2.copy())

    h = min(M_u8.shape[0], O_u8.shape[0], M2_u8.shape[0], O2_u8.shape[0])
    w = min(M_u8.shape[1], O_u8.shape[1], M2_u8.shape[1], O2_u8.shape[1])
    M_u8 = cv2.resize(M_u8, (w, h), interpolation=cv2.INTER_AREA)
    O_u8 = cv2.resize(O_u8, (w, h), interpolation=cv2.INTER_AREA)
    M2_u8 = cv2.resize(M2_u8, (w, h), interpolation=cv2.INTER_AREA)
    O2_u8 = cv2.resize(O2_u8, (w, h), interpolation=cv2.INTER_AREA)

    top = cv2.hconcat([M_u8, O_u8])
    bottom = cv2.hconcat([M2_u8, O2_u8])
    grid = cv2.vconcat([top, bottom])
    cv2.imwrite("sobel_output.png", grid)
    cv2.imshow("Magnitude | Orientation (Top: img1, Bottom: img2)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()