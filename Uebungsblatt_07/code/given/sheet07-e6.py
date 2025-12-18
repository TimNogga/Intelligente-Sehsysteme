#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import sys


def load_image(filepath):
    """Lädt ein Bild und konvertiert es zu Graustufen."""
    try:
        img = Image.open(filepath).convert('L')
        return np.array(img, dtype=np.float64)
    except FileNotFoundError:
        print(f"Fehler: Datei {filepath} nicht gefunden.")
        return None


def reduce_operation(image, sigma):
    """
    Reduce-Operation für die Pyramide

    Args:
        image (np.ndarray): Eingangsbild
        sigma (float): Standardabweichung für Gauss-Filter

    Returns:
        np.ndarray: Reduziertes Bild (halb so groß)

    """
    reduced = None

    # TODO: Reduce Operation implementieren

    return reduced


def expand_operation(image, sigma):
    """
    Expand-Operation für die Pyramide

    Args:
        image (np.ndarray): Eingangsbild (kleine Auflösung)
        sigma (float): Standardabweichung für Gauss-Filter

    Returns:
        np.ndarray: Expandiertes Bild (doppelt so groß)

    """
    # 1. Upsampling: Erstelle ein doppelt so großes Bild
    h, w = image.shape
    expanded = np.zeros((2 * h, 2 * w))

    # TODO: Expand Operation implementieren

    return expanded


def build_pyramid(image, sigma, levels=4):
    """
    Erstellt die Gauss-Pyramide

    Args:
        image (np.ndarray): Eingangsbild
        sigma (float): Standardabweichung für Gauss-Filter
        levels (int): Anzahl der Pyramid-Stufen

    Returns:
        list: Liste der Pyramid-Stufen
    """
    pyramid = [image.copy()]

    current_image = image.copy()

    # TODO: Pyramide aufbauen: weitere level der Pyramide in liste "pyramid" einfügen

    return pyramid


def visualize_pyramid(pyramid, sigma, output_dir="out"):
    """
    Visualisiert die Gauss-Pyramide

    Args:
        pyramid (list): Liste der Pyramid-Stufen
        sigma (float): Verwendete Standardabweichung
    """
    n_levels = len(pyramid)
    fig, axes = plt.subplots(1, n_levels, figsize=(15, 4))

    if n_levels == 1:
        axes = [axes]

    for i, image in enumerate(pyramid):
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Stufe {i + 1}\n{image.shape}')
        axes[i].axis('off')

    plt.suptitle(f'Gauss-Pyramide (σ = {sigma})')
    plt.tight_layout()

    # Speichere als Datei
    output_file = os.path.join(output_dir, f'gauss_pyramide_sigma_{sigma:.1f}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()  # Schließe Figure um Speicher zu sparen
    print(f"Pyramide gespeichert: {output_file}")


def test_expand_operation(pyramid, sigma, level=1, output_dir="out"):
    """
    Testet die Expand-Operation auf der angegebenen Stufe

    Args:
        pyramid (list): Die Gauss-Pyramide
        sigma (float): Standardabweichung für Gauss-Filter
        level (int): Pyramiden-Stufe für den Expand-Test
    """
    if level >= len(pyramid):
        print(f"Fehler: Stufe {level} existiert nicht!")
        return

    print(f"Teste Expand-Operation auf Stufe {level}")
    original = pyramid[level]
    expanded = expand_operation(original, sigma)

    # Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original (Stufe {level + 1})\n{original.shape}')
    axes[0].axis('off')

    axes[1].imshow(expanded, cmap='gray')
    axes[1].set_title(f'Nach Expand\n{expanded.shape}')
    axes[1].axis('off')

    plt.suptitle('Expand-Operation Test')
    plt.tight_layout()

    # Speichere als Datei
    output_file = os.path.join(output_dir, f'expand_test_stufe_{level + 1}_sigma_{sigma:.1f}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()  # Schließe Figure um Speicher zu sparen
    print(f"Expand-Test gespeichert: {output_file}")


def process_image(filepath, sigma, levels, output_dir):
    """
    Verarbeitet ein einzelnes Bild

    Args:
        filepath (str): Pfad zur Bilddatei
        sigma (float): Standardabweichung für Gauss-Filter
        levels (int): Anzahl Pyramid-Stufen
    """
    print(f"\n=== Verarbeite: {filepath} ===")
    print(f"Sigma: {sigma}")

    # Lade und konvertiere Bild
    image = load_image(filepath)
    print(f"Original-Bildgröße: {image.shape}")

    # Erstelle Pyramide
    pyramid = build_pyramid(image, sigma, levels)

    # Visualisiere Pyramide
    visualize_pyramid(pyramid, sigma, output_dir=output_dir)

    # Teste Expand-Operation auf Stufe 2
    test_expand_operation(pyramid, sigma, level=1, output_dir=output_dir)


def main():
    """Hauptfunktion mit CLI-Interface."""
    parser = argparse.ArgumentParser(description='Gauss-Pyramide')
    parser.add_argument('filepaths', nargs='*',
                        help='Pfade zu Bilddateien (optional)')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Standardabweichung für Gauss-Filter (default: 2.0)')
    parser.add_argument('--levels', type=int, default=4,
                        help='Anzahl Pyramid-Stufen (default: 4)')
    parser.add_argument('--output', type=str, default='out',
                        help='Ausgabeverzeichnis (default: output)')

    args = parser.parse_args()

    # Ausgabeordner erstellen
    os.makedirs(args.output, exist_ok=True)
    print(f"Ausgabeordner: {args.output}")

    # Bestimme Bilddateien
    if args.filepaths:
        filepaths = args.filepaths
    else:
        default_file = "Testbild_Lena_512x512.png"
        if os.path.exists(default_file):
            filepaths = [default_file]
        else:
            print(f"Warnung: Standard-Testbild '{default_file}' nicht gefunden!")
            print("Bitte Bilddatei als Parameter angeben.")
            sys.exit(1)

    # Verarbeite jedes Bild
    for filepath in filepaths:
        process_image(filepath, args.sigma, args.levels, args.output)


if __name__ == "__main__":
    main()