## 1. Co zostało zrobione

### Gałąź `demo_setup`
- Naprawiono `src/config.py` (wcześniej pusty) – ścieżki, etykiety, hiperparametry.
- Dodano `src/transforms.py` – `get_train_transforms()` (z augmentacją) i `get_val_transforms()` (bez losowości, używane w walidacji, testach i w demo).
- Dodano demo: `demo/predict.py` i `demo/README.md`.

### Gałąź `log_process
- **`src/logger.py`** – logowanie treningu:
  - tekstowy log `train_log.txt` (dopisywany i zapisywany po każdej linii, więc przeżywa awarię sesji),
  - zdarzenia TensorBoard (`events.*`),
  - `hparams.json` z hiperparametrami na starcie runu,
  - `final_metrics.json` po zakończeniu treningu.
  Każdy run ma własny folder `runs/<YYYYMMDD_HHMMSS>/`.
- **Notebook** `notebooks/cnn-plant-pathology.ipynb` – hiperparametry brane są z `Config.to_hparams()`
  (jedno źródło prawdy), a trening wywołuje `setup_run(...)` i loguje każdy batch i każdą epokę.
- **`sanity_check.py`** – test „overfit na jednym batchu”, który sprawdza, czy trening w ogóle działa.
  Zapisuje log do `logs/sanity_run.txt`. Jest też plik `logs/sanity_run.txt` z przebiegu w repo.
- **`.gitignore`** – dodano wpis, żeby nie commitować plików `events.out.tfevents.*` z `runs/`.

---

## 2. Jak uruchomić demo

Demo potrzebuje pliku `models/plant_cnn_demo.pth`. Powstaje on w notebooku (komórka z `add_thresholds_to_checkpoint`),
po treningu. Bez niego skrypt się nie uruchomi.

Wymagania:

```bash
pip install -r requirements.txt
```

Uruchamiaj z folderu `plant-pathology/`:

```bash
python demo/predict.py                       # wszystkie zdjęcia z demo/input_images/
python demo/predict.py --image liść.jpg      # pojedyncze zdjęcie
python demo/predict.py --model <ścieżka.pth> # inny checkpoint
```

- Wrzuć zdjęcia (`.jpg`, `.jpeg`, `.png`) do `demo/input_images/`. Najlepiej zdjęcia z `test_images/`,
  bo model ich nie widział w treningu.
- Progi są czytane z checkpointu (klucz `thresholds`), a nie są stałe 0.5.
- Dla każdego zdjęcia wypisywane są klasy powyżej progu razem z prawdopodobieństwem.
  Jeśli żadna klasa nie przekracza progu, wypisywana jest klasa top-1 z adnotacją.

Przykład wyjścia:

```
85f8cb619c66b863.jpg
  rust                 p=0.303  (próg 0.15)

ad8770db05586b59.jpg
  complex              p=0.537  (próg 0.15)

c7b03e718489f3ca.jpg
  frog_eye_leaf_spot   p=0.530  (próg 0.15)
```

## 3. Jak uruchomić trening i gdzie są logi

### Trening (notebook)
1. Ustaw liczbę epok w `src/config.py` → `DEFAULT_HPARAMS['num_epochs']`.
2. Otwórz `notebooks/cnn-plant-pathology.ipynb` i uruchom komórki po kolei (Run All po restarcie kernela).
3. Każdy run tworzy `runs/<timestamp>/`.

Wznawianie przerwanego treningu: w komórce treningu ustaw `RESUME` na ścieżkę do checkpointu
oraz `RUN_DIR` na folder poprzedniego runu. Wtedy logi są dopisywane do tego samego pliku.

### Co powstaje w `runs/<timestamp>/`

| Plik | Zawartość |
|---|---|
| `train_log.txt` | czytelny log tekstowy (patrz sekcja 4) |
| `hparams.json` | hiperparametry runu |
| `final_metrics.json` | najlepsze wyniki po zakończeniu (np. `hparam/best_val_f1`) |
| `events.out.tfevents.*` | dane dla TensorBoard |

### Sanity check
```bash
python sanity_check.py
```
Log trafia do `logs/sanity_run.txt`. Zobacz sekcję 4.

### TensorBoard
Z folderu `plant-pathology/`:

```bash
tensorboard --logdir runs
```

Otwórz w przeglądarce adres, który wypisze program (zwykle `http://localhost:6006`).
Zakładki i tagi:
- `loss/train`, `loss/val` – strata po epokach,
- `acc/train`, `acc/val` – dokładność (exact match),
- `f1/train`, `f1/val` – macro F1,
- `lr` – learning rate,
- `batch/train_loss` – strata co batch,
- zakładka **HParams** – hiperparametry razem z wynikami.

---

## 4. Jak czytać logi

### Format linii
```
<data i czas> | <poziom> | <treść>
```

### `train_log.txt` (trening)

Najważniejsze typy linii:

- `HPARAMS {...}` – hiperparametry runu, na początku pliku.
- `Run dir: ...` – folder runu.
- `epoch E | batch B/N | loss L` – strata co 50 batchy (`batch_log_every=50`). Numeracja batchów od 1.
- `EPOCH E | train loss ... acc ... f1 ... | val loss ... acc ... f1 ... | lr ...` – podsumowanie epoki.
  Epoka liczona od 1. Kluczowe liczby to **val loss** i **val f1**.
- `FINAL {...}` – końcowe metryki (to samo, co w `final_metrics.json`).

Jak to interpretować:
- **Train loss maleje, val loss maleje** – model się uczy.
- **Train loss maleje, val loss rośnie** – przeuczenie (overfitting). Patrz na `val f1` i na `best_epoch`.
- **lr spada o 10×** – `ReduceLROnPlateau` uznał, że val loss stanął w miejscu (cierpliwość: `patience` epok).
- **val f1 rośnie** – poprawa według macro F1, czyli metryki wybieranej do zapisu najlepszego modelu.

Szybkie sprawdzenie w terminalu:

```bash
# Podsumowania epok
grep "EPOCH" runs/<timestamp>/train_log.txt

# Ostatni wynik
tail -n 3 runs/<timestamp>/train_log.txt
```

### `logs/sanity_run.txt` (sanity check)

Test przechodzi, jeśli na końcu jest `WYNIK: PASS`. Kryterium: po treningu na jednym batchu
`loss < 0.05` i `exact_match == 1.0`.

Przykład z repo:

```
iter    1 | loss 0.68659 | macro_F1 0.2153 | exact_match 0.0000   # start, model jeszcze nic nie wie
iter   30 | loss 0.00562 | macro_F1 1.0000 | exact_match 1.0000   # już zapamiętał batch
FINAL (eval) | loss 0.00024 | macro_F1 1.0000 | exact_match 1.0000
WYNIK: PASS - model potrafi się przeuczyć, trening działa
```
---

## 5. Pliki w repo 

```
plant-pathology/
├── demo/                 predict.py, README.md, input_images/ (lokalnie)
├── logs/sanity_run.txt   log z sanity checka
├── notebooks/            cnn-plant-pathology.ipynb
├── src/
│   ├── config.py         ścieżki, etykiety, hiperparametry
│   ├── transforms.py     transformacje train/val
│   ├── dataset.py        PlantDataset, wagi klas
│   ├── model.py          CustomPlantCNN
│   ├── checkpoint.py     zapis/odczyt checkpointów, eksport modelu do demo
│   └── logger.py         logowanie: train_log.txt, TensorBoard, hparams
├── sanity_check.py
└── requirements.txt
```

Dane przechowywane lokalnie: `data/`, `runs/`, `baseline_outputs/`, `models/`.