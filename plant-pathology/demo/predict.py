"""
Demo: klasyfikacja zdjęć liści jabłoni (multi-label).

Uruchomienie z folderu plant-pathology/:
    python demo/predict.py                       # wszystkie zdjęcia z demo/input_images/
    python demo/predict.py --image liść.jpg      # pojedyncze zdjęcie
"""
import argparse
import sys
from pathlib import Path

# Dodaj plant-pathology/ do ścieżki, żeby działał import `src`
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import torch
from PIL import Image

from src import config as C
from src.checkpoint import load_model_for_inference
from src.transforms import get_val_transforms

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = DEMO_DIR / 'input_images'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def predict_image(model, transform, labels, thresholds, path):
    """Zwraca listę (etykieta, prawdopodobieństwo) i flagę, czy użyto top-1 awaryjnie."""
    image = Image.open(path).convert('RGB')
    x = transform(image).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].tolist()

    probs_by_label = dict(zip(labels, probs))
    above = [(label, p) for label, p in probs_by_label.items() if p > thresholds[label]]

    if above:
        above.sort(key=lambda item: item[1], reverse=True)
        return above, False

    # Żadna klasa nie przekracza progu -> pokaż top-1 z adnotacją
    top1 = max(probs_by_label.items(), key=lambda item: item[1])
    return [top1], True


def main():
    parser = argparse.ArgumentParser(description='Predykcja chorób liści jabłoni')
    parser.add_argument('--image', type=str, default=None,
                        help='ścieżka do pojedynczego zdjęcia (domyślnie: cały folder input_images)')
    parser.add_argument('--model', type=str, default=str(C.DEMO_MODEL_PATH),
                        help='ścieżka do checkpointu (domyślnie: models/plant_cnn_demo.pth)')
    args = parser.parse_args()

    # Progi są czytane z checkpointu (klucz 'thresholds'), nie ze stałego 0.5
    model, labels, thresholds = load_model_for_inference(args.model, device='cpu')
    transform = get_val_transforms()

    if args.image:
        paths = [Path(args.image)]
    else:
        DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        paths = sorted(p for p in DEFAULT_INPUT_DIR.iterdir()
                       if p.suffix.lower() in IMAGE_EXTENSIONS)

    if not paths:
        print(f'Brak zdjęć do przetworzenia. Wrzuć pliki do: {DEFAULT_INPUT_DIR}')
        return

    for path in paths:
        results, used_top1 = predict_image(model, transform, labels, thresholds, path)
        print(f'\n{path.name}')
        if used_top1:
            print('  Żadna klasa nie przekracza progu. Najbardziej prawdopodobna (top-1):')
        for label, prob in results:
            print(f'  {label:<20} p={prob:.3f}  (próg {thresholds[label]:.2f})')


if __name__ == '__main__':
    main()