import argparse

import cv2
from os.path import *
import numpy as np


def exercise1(image_folder=".", input: str = None):
    image_path = join(image_folder, input)
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Fehler beim Laden des Bildes.")
            return
    except FileNotFoundError:
        print("Fehler beim Laden des Bildes.")
        return
    image_clone = image.copy()
    
    points = []
    def clicked(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x,y))
            cv2.circle(image, (x,y), 3, (255,0,0), -1)
            if len(points) == 2:
                print(f"Start: {points[0]}, Ende: {points[1]}")
            cv2.imshow("Image", image)

    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", clicked)
    cv2.waitKey(0)  
    
    if len(points) == 2:
        image_float = image.astype(np.float64)
        gradx = cv2.Sobel(image_float, cv2.CV_64F, 1, 0, ksize=3)
        grady = cv2.Sobel(image_float, cv2.CV_64F, 0, 1, ksize=3)
        g = np.sqrt(gradx**2 + grady**2)
        
        start = points[0]
        end = points[1]
        
        gmod = 0.5 * (g[start[1], start[0]] + g[end[1], end[0]])
        ck = np.abs(g - gmod)
        
        max_cost = np.sum(ck) + 1
        pk = np.full(image.shape, max_cost)
        pk[start[1], start[0]] = ck[start[1], start[0]]
        active_nodes = [start]
        came_from = {}
        
        def get_neighbors(node):
            x, y = node
            neighbors = []
            for dx,dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                    neighbors.append((nx, ny))
            return neighbors
        
        while active_nodes:
            current = min(active_nodes, key=lambda node: pk[node[1], node[0]])
            active_nodes.remove(current)
            if current == end:
                break
            for neighbor in get_neighbors(current):
                ny, nx = neighbor[1], neighbor[0]
                new_cost = pk[current[1], current[0]] + ck[ny, nx]
                if new_cost < pk[ny, nx]:
                    pk[ny, nx] = new_cost
                    came_from[neighbor] = current
                    if neighbor not in active_nodes:
                        active_nodes.append(neighbor)
        
        if end in came_from:
            path = []
            current = end
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            
            for i in range(len(path)-1):
                cv2.line(image_clone, path[i], path[i+1], (255,0,0), 1)
            cv2.imshow("Path", image_clone)
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()


def exercise3(image_folder="."):
    try:
        image = cv2.imread(join(image_folder, "Testbild_Lena_512x512.png"), cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return

    # Todo Ihre Lösung


def parse_args():
    parser = argparse.ArgumentParser(description="Intelligent Scissors")
    parser.add_argument("--input", type=str, default="Testbild_Gangman_300x200.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    image_folder = "."

    # ------------------
    # --- EXERCISE 1 ---
    # ------------------

    exercise1(image_folder=image_folder, input=args.input)



    # ------------------
    # --- EXERCISE 3 ---
    # ------------------

    exercise3(image_folder=image_folder)



