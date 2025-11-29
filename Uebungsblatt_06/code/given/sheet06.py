import argparse

import cv2
from os.path import *
import numpy as np

def exercise1(image_folder=".", input: str = None):
    image_path = join(image_folder, input)
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return
    except FileNotFoundError:
        return
    
    image_clone = image.copy()
    image_display = image.copy()
    
    points = []
    all_paths = []
    ck = None
    
    def get_neighbors(node):
        x, y = node
        neighbors = []
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                neighbors.append((nx, ny))
        return neighbors
    
    def calculate_cost_map():
        nonlocal ck
        image_float = image.astype(np.float64)
        gradx = cv2.Sobel(image_float, cv2.CV_64F, 1, 0, ksize=3)
        grady = cv2.Sobel(image_float, cv2.CV_64F, 0, 1, ksize=3)
        g = np.sqrt(gradx**2 + grady**2)
        g_max = np.max(g)
        if g_max > 0:
            g_normalized = g / g_max
        else:
            g_normalized = g
        ck = 1.0 - g_normalized + 0.001
    
    def dijkstra(start, end):
        max_cost = np.sum(ck) + 1
        pk = np.full(image.shape, max_cost)
        pk[start[1], start[0]] = ck[start[1], start[0]]
        active_nodes = [start]
        came_from = {}
        
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
            return path
        return None
    
    def update_display():
        display_img = cv2.cvtColor(image_display, cv2.COLOR_GRAY2BGR)
        
        for path in all_paths:
            for i in range(len(path)-1):
                cv2.line(display_img, path[i], path[i+1], (255, 0, 0), 2)
        
        for i, point in enumerate(points):
            color = (0, 255, 0) if i == 0 else (0, 0, 255) if i == len(points)-1 else (255, 0, 0)
            cv2.circle(display_img, point, 4, color, -1)
            cv2.putText(display_img, str(i), (point[0]+5, point[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imshow("Intelligent Scissors", display_img)
    
    def clicked(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            
            if len(points) == 1:
                calculate_cost_map()
            
            elif len(points) >= 2:
                start = points[-2]
                end = points[-1]
                
                path = dijkstra(start, end)
                if path is not None:
                    all_paths.append(path)
            
            update_display()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(points) >= 2:
                cv2.waitKey(1000)
    
    cv2.namedWindow("Intelligent Scissors", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Intelligent Scissors", clicked)
    
    update_display()
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    if len(all_paths) > 0:
        final_result = cv2.cvtColor(image_clone, cv2.COLOR_GRAY2BGR)
        for path in all_paths:
            for i in range(len(path)-1):
                cv2.line(final_result, path[i], path[i+1], (255, 0, 0), 2)
        
        for point in points:
            cv2.circle(final_result, point, 4, (0, 255, 0), -1)
        
        cv2.imshow("Finale Segmentierung", final_result)
        cv2.waitKey(0)
        
        cv2.imwrite("segmentierung_ergebnis.jpg", final_result)
    
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



