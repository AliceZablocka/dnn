"""Checkpointy: zapis co epokę, wznawianie treningu, eksport lekkiego modelu do demo."""
from pathlib import Path
from typing import Dict, Optional

import torch

from . import config as C
from .model import CustomPlantCNN


def save_checkpoint(path, model, optimizer, scheduler, epoch: int,
                    metrics: Dict, history: Dict, hparams: Dict,
                    best_val_f1: float, thresholds: Optional[Dict] = None):
    """Pełny checkpoint treningowy (z optymalizatorem i schedulerem)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        **metrics,              # val_loss / val_acc / val_f1 na wierzchu - starsze komórki notebooka z tego czytają
        'metrics': metrics,
        'history': history,
        'hparams': hparams,
        'best_val_f1': float(best_val_f1),
        'labels': list(C.LABELS),
        'thresholds': thresholds,
    }, path)


def load_checkpoint(path, device=C.DEVICE):
    # weights_only=False: to nasze własne pliki, zawierają słowniki i listy
    return torch.load(path, map_location=device, weights_only=False)


def resume_from_checkpoint(path, model, optimizer=None, scheduler=None, device=C.DEVICE):
    """Wczytuje stan treningu. Zwraca (start_epoch, history, best_val_f1)."""
    ckpt = load_checkpoint(path, device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and ckpt.get('optimizer_state_dict'):
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    print(f"Wznowiono z {path} (ukończona epoka {ckpt['epoch']})")
    return ckpt['epoch'], ckpt['history'], ckpt.get('best_val_f1', float('-inf'))


def export_for_inference(src_path=C.CHECKPOINT_DIR / 'best_model.pth', dst_path=C.DEMO_MODEL_PATH):
    """Zapisuje lekką wersję modelu (bez optymalizatora) dla demo - tę commitujemy na GitHuba."""
    ckpt = load_checkpoint(src_path, 'cpu')
    light = {k: ckpt.get(k) for k in ('epoch', 'model_state_dict', 'metrics', 'hparams', 'labels', 'thresholds')}
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(light, dst_path)
    print(f'Model do demo zapisany: {dst_path}')


def add_thresholds_to_checkpoint(path, thresholds: Dict, export: bool = True):
    """Dopisuje progi z optymalizacji do checkpointu i (domyślnie) eksportuje model do demo."""
    ckpt = load_checkpoint(path, 'cpu')
    ckpt['thresholds'] = {k: float(v) for k, v in thresholds.items()}
    torch.save(ckpt, path)
    print(f'Progi zapisane w {path}')
    if export:
        export_for_inference(path, C.DEMO_MODEL_PATH)


def load_model_for_inference(path=C.DEMO_MODEL_PATH, device=C.DEVICE):
    """Dla demo: zwraca (model w trybie eval, labels, thresholds)."""
    ckpt = load_checkpoint(path, device)
    hp = ckpt.get('hparams') or {}
    labels = ckpt.get('labels') or list(C.LABELS)
    model = CustomPlantCNN(num_classes=len(labels), dropout=hp.get('dropout', 0.5))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    thresholds = ckpt.get('thresholds') or {label: 0.5 for label in labels}
    return model, labels, thresholds