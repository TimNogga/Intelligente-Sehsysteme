import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt


def sobel_edge_detection(image, threshold=50):
    """
    Sobel Edge Detection mit Schwellwert

    Parameters:
    - image: Input grayscale image (numpy array)
    - threshold: Threshold for edge magnitude

    Returns:
    - edge_image: Binary edge image (numpy array)

    """

    edge_image = np.zeros_like(image)
    # TODO: Sobel-Operator anwenden und mit threshold filtern

    return edge_image


def hough_transform_lines(edge_image, rho_min=-64, rho_max=64, theta_resolution=180):
    """
    Hough Transform für Geraden

    Parameters:
    - edge_image: Binary edge image
    - rho_min: Minimum rho value
    - rho_max: Maximum rho value
    - theta_resolution: Number of theta values (0-179 degrees)

    Returns:
    - accumulator: Hough accumulator matrix
    - rho_values: Array of rho values
    - theta_values: Array of theta values (in degrees)

    """
    height, width = edge_image.shape

    # TODO: Hough Transform für Gerade implementieren

    accumulator = np.zeros((129, theta_resolution))  # Dummy für Struktur
    rho_values = np.arange(rho_min, rho_max + 1)
    theta_values = np.arange(0, theta_resolution)

    print(f"Hough Transform - rho: [{rho_min}, {rho_max}], theta: [0, {theta_resolution - 1}]")

    return accumulator, rho_values, theta_values


def visualize_hough_space(accumulator, rho_values, theta_values):
    """
    Visualisierung des Hough-Raums mit Normalisierung

    Args:
        accumulator: Hough accumulator matrix
        rho_values: Array of rho values
        theta_values: Array of theta values

    Returns:
        visualization: Normalized visualization image (0-255)
    """
    # TODO: Akkumulator normalisieren (über gesamten Intensitätsbereich strecken)

    visualization = np.zeros_like(accumulator, dtype=np.uint8)

    return visualization


def save_hough_visualization(visualization, rho_values, theta_values, output_path):
    """
    Speichere Hough-Raum Visualisierung mit Achsenbeschriftung

    Args:
        visualization: Normalized hough space image
        rho_values: Array of rho values
        theta_values: Array of theta values
        output_path: Output file path
    """
    # Matplotlib Plot erstellen
    plt.figure(figsize=(12, 8))
    plt.imshow(visualization, cmap='hot', aspect='auto',
               extent=[theta_values[0], theta_values[-1], rho_values[-1], rho_values[0]])
    plt.xlabel('Theta (Grad)', fontsize=12)
    plt.ylabel('Rho', fontsize=12)
    plt.title('Hough-Raum (Normalisiert)', fontsize=14)
    plt.colorbar(label='Normalisierte Intensität')

    # Achsen-Ticks anpassen
    plt.xticks(np.arange(0, len(theta_values), 30))
    plt.yticks(np.arange(rho_values[0], rho_values[-1] + 1, 20))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Zusätzlich als einfaches Grauwertbild speichern
    simple_output = output_path.replace('.png', '_simple.png')
    cv2.imwrite(simple_output, visualization)

    print(f"Hough-Raum gespeichert: {output_path}")
    print(f"Einfaches Grauwertbild gespeichert: {simple_output}")


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


def process_hough_transform(image_path, threshold, rho_min, rho_max, theta_resolution):
    """Hough-Transformation Pipeline"""
    print("=== Hough-Transformation für Geraden ===")
    print(f"Eingabebild: {image_path}")
    print(f"Parameter: Schwellwert={threshold}, rho=[{rho_min}, {rho_max}], theta_res={theta_resolution}")

    # Bild laden
    image = load_image(image_path, grayscale=True)
    print(f"Bildgröße: {image.shape}")

    # Schritt 1: Kantendetektion mit Sobel
    print("\n--- Kantendetektion (Sobel) ---")
    edge_image = sobel_edge_detection(image, threshold)

    edge_count = np.sum(edge_image > 0)
    print(f"Anzahl Kantenpixel: {edge_count}")

    # Schritt 2: Hough-Transformation
    print("\n--- Hough-Transformation ---")
    accumulator, rho_values, theta_values = hough_transform_lines(
        edge_image, rho_min, rho_max, theta_resolution)

    max_votes = np.max(accumulator)
    print(f"Maximale Anzahl Votes: {max_votes}")

    # Schritt 3: Hough-Raum visualisieren
    print("\n--- Visualisierung ---")
    hough_vis = visualize_hough_space(accumulator, rho_values, theta_values)

    # Ergebnisse speichern
    base_name = image_path.split('/')[-1].split('.')[0]

    # Kantenbild speichern
    edge_output = f"result_edges_{base_name}.png"
    cv2.imwrite(edge_output, edge_image)
    print(f"Kantenbild gespeichert: {edge_output}")

    # Hough-Raum speichern
    hough_output = f"result_hough_{base_name}.png"
    save_hough_visualization(hough_vis, rho_values, theta_values, hough_output)

    return edge_image, accumulator, hough_vis


def main():
    parser = argparse.ArgumentParser(description='Hough-Transformation für Geraden')
    parser.add_argument('--image', required=True, help='Pfad zum Eingabebild')
    parser.add_argument('--threshold', type=int, default=50,
                        help='Schwellwert für Sobel-Kantendetektion')
    parser.add_argument('--rho-min', type=int, default=-64,
                        help='Minimaler rho-Wert')
    parser.add_argument('--rho-max', type=int, default=64,
                        help='Maximaler rho-Wert')
    parser.add_argument('--theta-resolution', type=int, default=180,
                        help='Anzahl Theta-Werte (0 bis theta-resolution-1 Grad)')

    args = parser.parse_args()

    try:
        process_hough_transform(
            args.image,
            args.threshold,
            args.rho_min,
            args.rho_max,
            args.theta_resolution
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