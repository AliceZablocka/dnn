"""Konfiguracja projektu: ścieżki, etykiety, hiperparametry."""
import os
from pathlib import Path

import torch

# Główny folder repo (plant-pathology/), niezależnie od tego, skąd uruchamiasz kod
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dane: domyślnie <repo>/data, albo folder ze zmiennej środowiskowej PLANT_DATA_DIR
DATA_DIR = Path(os.environ.get('PLANT_DATA_DIR', PROJECT_DIR / 'data'))
TRAIN_CSV = DATA_DIR / 'train.csv'
IMAGES_DIR = DATA_DIR / 'train_images'      # zmień na 'train_data', jeśli tak nazywa się u was folder
TEST_IMAGES_DIR = DATA_DIR / 'test_images'

LABELS = ['complex', 'frog_eye_leaf_spot', 'healthy', 'powdery_mildew', 'rust', 'scab']
NUM_CLASSES = len(LABELS)

INPUT_HEIGHT = 224
INPUT_WIDTH = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DEFAULT_HPARAMS = {
    'batch_size': 32,
    'num_workers': 2,       # ustaw 0, jeśli DataLoader się wiesza
    'num_epochs': 30,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'patience': 3,
    'dropout': 0.5,
    'seed': 42,
}

# Wyniki
OUTPUT_DIR = PROJECT_DIR / 'baseline_outputs'
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
METRICS_DIR = OUTPUT_DIR / 'metrics'
PLOT_DIR = OUTPUT_DIR / 'plots'
LOG_DIR = PROJECT_DIR / 'logs'
RUNS_DIR = PROJECT_DIR / 'runs'

# Lekki model do demo (wagi + etykiety + progi, bez stanu optymalizatora)
MODELS_DIR = PROJECT_DIR / 'models'
DEMO_MODEL_PATH = MODELS_DIR / 'plant_cnn_demo.pth'