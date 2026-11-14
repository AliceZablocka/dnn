# Demo – klasyfikacja chorób liści jabłoni

## Struktura

- `input_images/` – tu wrzuć zdjęcia liści (jpg, jpeg, png)
- `predict.py` – skrypt predykcji
- `../../../../README.md` – ten plik

## Wymagania

Musi istnieć plik `models/plant_cnn_demo.pth` (patrz sekcja poniżej).
Instalacja zależności:

    pip install -r requirements.txt

## Uruchomienie

Z folderu `..`:

    python demo/predict.py                       # wszystkie zdjęcia z demo/input_images/
    python demo/predict.py --image liść.jpg      # pojedyncze zdjęcie

Program wypisuje wszystkie klasy powyżej progu razem z prawdopodobieństwami.
Jeśli żadna klasa nie przekracza progu, wypisuje klasę top-1.

## Jak powstaje model demo

Po treningu w notebooku, w folderze `..`:

    from src.checkpoint import add_thresholds_to_checkpoint
    add_thresholds_to_checkpoint('baseline_outputs/checkpoints/best_model.pth', thresholds)

Gdzie `thresholds` to słownik `{etykieta: próg}` z optymalizacji. Funkcja zapisuje progi
do checkpointu i eksportuje lekki model do `models/plant_cnn_demo.pth`.