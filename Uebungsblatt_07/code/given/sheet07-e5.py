#!/usr/bin/env python3

import argparse
import os
import numpy as np
from PIL import Image


def load_image(filepath):
    """Lädt ein Bild und konvertiert es zu Graustufen."""
    try:
        img = Image.open(filepath).convert('L')
        return np.array(img, dtype=np.float64)
    except FileNotFoundError:
        print(f"Fehler: Datei {filepath} nicht gefunden.")
        return None


def save_image(image_array, filepath):
    """Speichert ein NumPy Array als Bild."""
    # Normalisierung auf 0-255 Bereich
    normalized = np.clip(image_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(normalized)
    img.save(filepath)


def save_magnitude_spectrum(fft_result, filepath):
    """Speichert das Magnitudenspektrum der FFT als Bild."""
    magnitude = np.abs(fft_result)
    # Logarithmische Skalierung für bessere Visualisierung
    log_magnitude = np.log(1 + magnitude)

    # Normalisierung für Bilddarstellung
    normalized = (log_magnitude / np.max(log_magnitude) * 255).astype(np.uint8)
    img = Image.fromarray(normalized)
    img.save(filepath)


def dft_2d(image):
    """
    Implementieren Sie hier die 2D Diskrete Fourier-Transformation.

    Args:
        image: 2D NumPy Array mit Pixelwerten

    Returns:
        2D NumPy Array mit komplexen Fourier-Koeffizienten
    """
    M, N = image.shape
    dft_result = np.zeros((M, N), dtype=np.complex128)

    # TODO DFT implementieren

    return dft_result


def idft_2d(fft_result):
    """
    Implementieren Sie hier die inverse 2D Diskrete Fourier-Transformation.

    Args:
        fft_result: 2D NumPy Array mit komplexen Fourier-Koeffizienten

    Returns:
        2D NumPy Array mit rekonstruierten Pixelwerten

    """
    M, N = fft_result.shape
    reconstructed = np.zeros((M, N), dtype=np.complex128)

    # TODO inverse DFT implementieren

    return np.real(reconstructed)


def process_image(filepath, output_dir, apply_inverse=False):
    """Verarbeitet ein einzelnes Bild."""
    print(f"Verarbeite Bild: {filepath}")

    # Bild laden
    image = load_image(filepath)
    if image is None:
        return

    # Dateiname extrahieren (ohne Erweiterung)
    basename = os.path.splitext(os.path.basename(filepath))[0]

    try:
        # Fourier-Transformation anwenden
        print("  Führe 2D-DFT durch...")
        dft_result = dft_2d(image)

        # Magnitudenspektrum speichern
        spectrum_path = os.path.join(output_dir, f"{basename}_spectrum.png")
        save_magnitude_spectrum(dft_result, spectrum_path)
        print(f"  Magnitudenspektrum gespeichert: {spectrum_path}")

        # Inverse Transformation (optional)
        if apply_inverse:
            print("  Führe inverse 2D-DFT durch...")
            reconstructed = idft_2d(dft_result)

            # Rekonstruiertes Bild speichern
            reconstructed_path = os.path.join(output_dir, f"{basename}_reconstructed.png")
            save_image(reconstructed, reconstructed_path)
            print(f"  Rekonstruiertes Bild gespeichert: {reconstructed_path}")

            # Fehleranalyse
            mse = np.mean((image - reconstructed) ** 2)
            print(f"  MSE zwischen Original und Rekonstruktion: {mse:.6f}")

    except NotImplementedError as e:
        print(f"  Fehler: {e}")


def main():
    """Hauptfunktion mit CLI-Interface."""
    parser = argparse.ArgumentParser(
        description="2D Fourier-Transformation für Bilder",
    )

    parser.add_argument(
        '--filepaths',
        nargs='*',
        default=['Kreis.png', 'Sechseck.png'],
        help='Pfade zu den zu verarbeitenden Bilddateien (Default: Kreis.png, Sechseck.png)'
    )

    parser.add_argument(
        '--inverse',
        action='store_true',
        help='Wendet auch die inverse Fourier-Transformation an'
    )

    parser.add_argument(
        '--outdir',
        default='out',
        help='Ausgabeordner für Ergebnisse (Default: out)'
    )

    args = parser.parse_args()

    # Ausgabeordner erstellen
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Ausgabeordner: {args.outdir}")

    # Bilder verarbeiten
    for filepath in args.filepaths:
        process_image(filepath, args.outdir, args.inverse)

    print("Verarbeitung abgeschlossen!")


if __name__ == "__main__":
    main()