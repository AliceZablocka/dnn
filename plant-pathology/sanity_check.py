"""
Sanity check: overfit na JEDNYM batchu.
Oczekiwane: loss -> ~0, exact match -> 1.0. Log: logs/sanity_run.txt

Uruchomienie z głównego folderu repo:
    python sanity_check.py
"""
import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src import config as C
from src.dataset import PlantDataset, compute_class_weights
from src.model import CustomPlantCNN
from src.transforms import get_val_transforms

# Logger osoby A, a dopóki go nie ma - prosty logger zastępczy
try:
    from src.logger import get_logger
except ImportError:
    def get_logger(name, log_file):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        for h in (logging.FileHandler(log_file, mode='w', encoding='utf-8'), logging.StreamHandler()):
            h.setFormatter(fmt)
            logger.addHandler(h)
        return logger


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=str(C.TRAIN_CSV))
    p.add_argument('--images_dir', default=str(C.IMAGES_DIR))
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--log_every', type=int, default=10)
    args = p.parse_args()

    seed = C.DEFAULT_HPARAMS['seed']
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    logger = get_logger('sanity', C.LOG_DIR / 'sanity_run.txt')
    hparams = {**vars(args), 'weight_decay': 0.0, 'dropout': C.DEFAULT_HPARAMS['dropout'],
               'device': C.DEVICE, 'started': datetime.now().isoformat(timespec='seconds')}
    logger.info('=== SANITY CHECK: overfit na jednym batchu ===')
    logger.info('HPARAMS ' + json.dumps(hparams))

    df = pd.read_csv(args.csv)
    df['image'] = df['image'].apply(lambda name: str(Path(args.images_dir) / name))

    # Bez augmentacji: w każdej iteracji dokładnie te same piksele
    loader = DataLoader(PlantDataset(df, get_val_transforms()), batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    images, labels = next(iter(loader))
    images, labels = images.to(C.DEVICE), labels.to(C.DEVICE)
    logger.info(f'Batch: {tuple(images.shape)}, pozytywy na klasę: {labels.sum(0).tolist()}')

    model = CustomPlantCNN(num_classes=C.NUM_CLASSES, dropout=hparams['dropout']).to(C.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_class_weights(df).to(C.DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    model.train()
    for it in range(1, args.iters + 1):
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it == 1 or it % args.log_every == 0:
            preds = (torch.sigmoid(logits) > 0.5).float()
            f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
            exact = (preds == labels).all(1).float().mean().item()
            logger.info(f'iter {it:4d} | loss {loss.item():.5f} | macro_F1 {f1:.4f} | exact_match {exact:.4f}')

    # Końcowa ocena w trybie eval (bez dropoutu)
    model.eval()
    with torch.no_grad():
        logits = model(images)
        final_loss = criterion(logits, labels).item()
        preds = (torch.sigmoid(logits) > 0.5).float()
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
    exact = (preds == labels).all(1).float().mean().item()
    passed = final_loss < 0.05 and exact == 1.0
    logger.info(f'FINAL (eval) | loss {final_loss:.5f} | macro_F1 {f1:.4f} | exact_match {exact:.4f}')
    logger.info('WYNIK: ' + ('PASS - model potrafi się przeuczyć, trening działa'
                              if passed else 'FAIL - sprawdź loss/LR/etykiety'))


if __name__ == '__main__':
    main()