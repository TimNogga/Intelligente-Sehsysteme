import os

import numpy as np
import cv2
import argparse


def load_image(image_path, grayscale=True):
    """Helper function to load images"""
    if grayscale:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    return image


def binarize_image_with_threshold(image, threshold, binarization_background):
    """Helper function to binarize image"""
    if binarization_background == "white":
        image = image <= threshold
    else:
        image = image >= threshold
    return image


def has_background_neighbor(binarized_image, col, row):
    """Check if any of the neighboring pixels are background"""
    height, width = binarized_image.shape
    for col_offset in [-1, 0, 1]:
        for row_offset in [-1, 0, 1]:
            if col_offset == 0 and row_offset == 0:
                continue
            _col = col + col_offset
            _row = row + row_offset
            if _col < 0 or _row < 0 or _col >= width or _row >= height:
                continue

            if binarized_image[_row, _col] == 0:
                return True
    return False

def calculate_features(binarized_image):
    """Berechne Formmerkmale für gegebenes binarisiertes Bild."""
    area = 0
    contour = 0
    compactness = 0
    moment_col = 0
    moment_row = 0
    moment_combined = 0

    height, width = binarized_image.shape

    mean_col = 0.0
    mean_row = 0.0

    for col in range(width):
        for row in range(height):
            val = binarized_image[row, col]
            if val:
                area += 1
                if has_background_neighbor(binarized_image, col, row):
                    contour += 1
                mean_col += col
                mean_row += row

    compactness = area / contour**2
    mean_col /= area
    mean_row /= area

    for col in range(width):
        for row in range(height):
            val = binarized_image[row, col]
            if val:
                moment_col += (col - mean_col)**2
                moment_row += (row - mean_row)**2
                moment_combined += (col - mean_col) * (row - mean_row)

    return area, contour, compactness, moment_col, moment_row, moment_combined

def process_feature_computation(image_path, binarization_threshold, binarization_background, output_dir=None, write_binarized_image=False):
    print("=== Feature Berechnung ===")
    print(f"Eingabebild: {image_path}")
    print(f"Parameter: Binarisierungs-Schwellwert={binarization_threshold}, Binarisierung invertiert={binarization_background}")

    image_filename = os.path.basename(image_path)
    feat_filename = image_filename.split(".")[0] + ".feat"
    binarized_output_filename = image_filename.split(".")[0] + "_binarized.png"

    # Bild laden
    image = load_image(image_path)
    print(f"Bildgröße: {image.shape}")

    # Binarisiere Bild mit gegebenem threshold
    binarized_image = binarize_image_with_threshold(image, binarization_threshold, binarization_background)

    # Berechne Features
    area, contour, compactness, moment_col, moment_row, moment_combined = calculate_features(binarized_image)
    print(f"Area: {area}, Contour: {contour}, Compactness: {compactness}, Moment Col: {moment_col}, Moment Row: {moment_row}, Moment Combined: {moment_combined}")

    # Schreibe .feat Datei
    if not output_dir:
        output_dir = ""

    with open(os.path.join(output_dir, feat_filename), "w") as output:
        output.write(f"{image_filename.split(".")[0]}\n")
        output.write(f"{area}\n")
        output.write(f"{contour}\n")
        output.write(f"{compactness}\n")
        output.write(f"{moment_col}\n")
        output.write(f"{moment_row}\n")
        output.write(f"{moment_combined}\n")

    if write_binarized_image:
        binarized_image_rgb = cv2.cvtColor(binarized_image.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(os.path.join(output_dir, binarized_output_filename)), cv2.cvtColor(binarized_image_rgb, cv2.COLOR_RGB2BGR))

    return area, contour, compactness, moment_col, moment_row, moment_combined

def process_object_identification(image_path, feature_dir, binarization_threshold, binarization_background):
    """TODO: Implementieren Sie die Objektidentifikation anhand der Formmerkmale von Referenzen im gegebenen Verzeichnis feature_dir."""


def main():
    parser = argparse.ArgumentParser(description='Berechnung Formmerkmale')
    parser.add_argument('--mode', required=True, choices=['features', 'identification'])
    parser.add_argument('--image', required=True, help='Pfad zum Eingabebild')
    parser.add_argument('--feat-dir', required=False, help='Pfad zum Verzeichnis mit .feat Dateien')

    parser.add_argument('--binarization-threshold', type=int, default=115,
                        help='Schwellwert für Binarisierung')
    parser.add_argument('--binarization-background', choices=["white", "black"], default="white",
                        help='Hintergrund für Binarisierung')
    parser.add_argument('--output_dir', required=False, default="", help='Ausgabepfad')
    parser.add_argument('--write-binarized', action="store_true", default=False,
                        help='Binarisiertes Bild ausgeben')

    args = parser.parse_args()

    try:
        if args.mode == 'features':
            process_feature_computation(
                args.image,
                args.binarization_threshold,
                args.binarization_background,
                args.output_dir,
                args.write_binarized
            )
        elif args.mode == 'identification':
            process_object_identification(
                args.image,
                args.feat_dir,
                args.binarization_threshold,
                args.binarization_background,
            )

    except Exception as e:
        print(f"Fehler: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nProgramm erfolgreich beendet!")
    return 0


if __name__ == "__main__":
    exit(main())