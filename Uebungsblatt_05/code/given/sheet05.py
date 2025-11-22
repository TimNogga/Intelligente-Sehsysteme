import cv2
import numpy as np
from os.path import join, dirname

def binarization(gray_u8, thresholds):
    img = np.asarray(gray_u8, dtype=np.uint8)

    def normalize(values):
        if not values:
            return []
        clipped = [min(max(int(v), 0), 255) for v in values]
        return sorted(set(clipped))

    def colorize(thr_list):
        labels = np.digitize(img, thr_list, right=False)
        if len(thr_list) <= 1:
            binary = ((labels > 0) * 255).astype(np.uint8)
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        palette = np.array([
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [128, 128, 128],
            [0, 128, 255],
        ], dtype=np.uint8)
        colors = palette[: len(thr_list) + 1]
        return colors[labels]

    preset_thresholds = normalize(thresholds)#damit die gui nicht erscheint für die aufgabe 2
    if preset_thresholds:
        return colorize(preset_thresholds)

    window = "Interaktive Binarisierung"
    cv2.namedWindow(window)
    cv2.createTrackbar("Anzahl", window, 1, 9, lambda _: None)
    for idx in range(9):
        cv2.createTrackbar(f"T{idx + 1}", window, 128, 255, lambda _: None)

    original = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    view = original.copy()
    last_thresholds: list[int] = []

    while True:
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

        count = cv2.getTrackbarPos("Anzahl", window)
        count = max(0, min(count, 9))
        current = [cv2.getTrackbarPos(f"T{idx + 1}", window) for idx in range(count)]
        current = normalize(current)
        view = original.copy() if not current else colorize(current)
        last_thresholds = current
        cv2.imshow(window, np.hstack((original, view)))

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q')):
            break

    cv2.destroyWindow(window)
    filename = f"Ex1_interactive_{'_'.join(map(str, last_thresholds)) or 'no_thr'}.png"
    output_path = join(dirname(__file__), filename)
    cv2.imwrite(output_path, view)
    print(f"Interaktives Ergebnis gespeichert unter: {output_path}")
    return view


def exercise1(image_folder=".", input: str = None):
    try:
        image = cv2.imread(join(image_folder, input), cv2.IMREAD_GRAYSCALE)

    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return
    if image is None:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return

    print("Öffne interaktive Binarisierung. Verwenden Sie die Regler und schließen Sie mit 'q' oder ESC.")
    binarization(image, thresholds=None)


def connected_component_labeling(source: np.ndarray, neighborhood: int = 8):
    if source.ndim == 3:
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    else:
        gray = source.copy()

    object_mask = gray == 0
    height, width = gray.shape
    labels = np.zeros((height, width), dtype=np.int32) 

    parent = [0]

    def make_set() -> int:
        parent.append(len(parent))
        return len(parent) - 1

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    if neighborhood == 8:
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
    else:
        neighbor_offsets = [(-1, 0), (0, -1)]

    for y in range(height):
        for x in range(width):
            if not object_mask[y, x]:
                continue

            neighbor_labels = []
            for dy, dx in neighbor_offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor_label = labels[ny, nx]
                    if neighbor_label > 0:
                        neighbor_labels.append(neighbor_label)

            if neighbor_labels:
                min_label = min(neighbor_labels)
                labels[y, x] = min_label
                for other_label in neighbor_labels:
                    if other_label != min_label:
                        union(min_label, other_label)
            else:
                new_label = make_set()
                labels[y, x] = new_label

    colorized = np.full((height, width, 3), 255, dtype=np.uint8)
    label_color: dict[int, tuple[int, int, int]] = {}
    palette = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [128, 0, 128],
        [0, 128, 255],
        [128, 128, 0],
        [255, 128, 0],
    ], dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label == 0:
                continue

            root_label = find(label)
            labels[y, x] = root_label

            if root_label not in label_color:
                color_index = (len(label_color)) % len(palette)
                label_color[root_label] = tuple(int(c) for c in palette[color_index])

            colorized[y, x] = label_color[root_label]

    return colorized


def exercise3(image_folder=".", threshold: np.uint8=120):
    try:
        image = cv2.imread(join(image_folder, "Testbild_Werkzeuge_768x576.png"), cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("Fehler beim Laden des Bildes. Bitte Pfad und Dateinamen prüfen.")
        return

    binary_color = binarization(image, thresholds=[threshold])
    binary_gray = cv2.cvtColor(binary_color, cv2.COLOR_BGR2GRAY)
    labeled_image = connected_component_labeling(binary_gray)

    cv2.imshow("Ex. 3) Region Labeling", labeled_image)
    cv2.waitKey(0)
    saved_1 = cv2.imwrite("Ex. 3) Region Labeling.png", labeled_image)
    print("Gespeichert:", saved_1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_folder = dirname(__file__)

    # ------------------
    # --- EXERCISE 1 ---
    # ------------------

    #exercise1(image_folder=image_folder, input="Kopf-MRT-Toenn.png")


    # ------------------
    # --- EXERCISE 3 ---
    # ------------------

    exercise3(image_folder=image_folder, threshold=120)