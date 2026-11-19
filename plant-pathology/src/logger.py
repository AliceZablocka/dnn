"""Logowanie treningu: plik tekstowy (logging), TensorBoard, hparams.json.

Każdy run ma własny folder runs/<timestamp>/:
    train_log.txt   - log tekstowy (dopisywany i flushowany po każdej linii, więc przeżywa awarię sesji)
    hparams.json    - hiperparametry zapisane na starcie runu
    events.*        - zdarzenia TensorBoard (tensorboard --logdir runs)

Zachowuje też get_logger(name, log_file), którego używa sanity_check.py.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from . import config as C

LOG_FORMAT = '%(asctime)s | %(levelname)s | %(message)s'


def get_logger(name: str, log_file, mode: str = 'w') -> logging.Logger:
    """Logger piszący do pliku i na konsolę. Sygnatura zgodna z sanity_check.py."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):      # bez dublowania linii przy ponownym uruchomieniu komórki
        h.close()
        logger.removeHandler(h)
    fmt = logging.Formatter(LOG_FORMAT)
    for h in (logging.FileHandler(log_file, mode=mode, encoding='utf-8'), logging.StreamHandler()):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def _tb_safe(hparams: Dict) -> Dict:
    """add_hparams przyjmuje tylko int/float/str/bool - resztę (None, listy) zamieniamy na str."""
    return {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in hparams.items()}


class RunLogger:
    """Jeden obiekt na run: logging + TensorBoard + hparams."""

    def __init__(self, hparams: Dict, run_dir=None, runs_dir=None, use_tensorboard: bool = True,
                 batch_log_every: int = 50):
        runs_dir = Path(runs_dir or C.RUNS_DIR)
        # run_dir podajemy tylko przy wznawianiu, żeby dopisać do tego samego runu
        self.run_dir = Path(run_dir) if run_dir else runs_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.hparams = dict(hparams)
        self.batch_log_every = batch_log_every

        self.logger = get_logger(f'train_{self.run_dir.name}', self.run_dir / 'train_log.txt', mode='a')

        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.run_dir))
            except ImportError:
                self.logger.warning('Brak pakietu tensorboard (pip install tensorboard) - loguję tylko do pliku.')

        hp_path = self.run_dir / 'hparams.json'
        if not hp_path.exists():
            hp_path.write_text(json.dumps(self.hparams, indent=2, default=str), encoding='utf-8')
        self.logger.info(f'Run dir: {self.run_dir}')
        self.logger.info('HPARAMS ' + json.dumps(self.hparams, default=str))

    # --- co batch ---
    def log_batch(self, epoch: int, batch_idx: int, n_batches: int, loss: float):
        step = epoch * n_batches + batch_idx
        if self.writer:
            self.writer.add_scalar('batch/train_loss', loss, step)
        if batch_idx % self.batch_log_every == 0:
            self.logger.info(f'epoch {epoch + 1} | batch {batch_idx + 1}/{n_batches} | loss {loss:.4f}')

    # --- co epokę ---
    def log_epoch(self, epoch: int, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, lr):
        """epoch liczone od 1."""
        vals = {'loss/train': train_loss, 'loss/val': val_loss,
                'acc/train': train_acc, 'acc/val': val_acc,
                'f1/train': train_f1, 'f1/val': val_f1, 'lr': lr}
        vals = {k: float(v) for k, v in vals.items()}
        if self.writer:
            for k, v in vals.items():
                self.writer.add_scalar(k, v, epoch)
            self.writer.flush()
        self.logger.info(
            f"EPOCH {epoch} | train loss {vals['loss/train']:.4f} acc {vals['acc/train']:.4f} f1 {vals['f1/train']:.4f}"
            f" | val loss {vals['loss/val']:.4f} acc {vals['acc/val']:.4f} f1 {vals['f1/val']:.4f}"
            f" | lr {vals['lr']:.2e}")

    # --- na końcu ---
    def finish(self, final_metrics: Dict):
        """final_metrics: np. {'hparam/best_val_f1': 0.81, 'hparam/best_epoch': 17}."""
        final_metrics = {k: float(v) for k, v in final_metrics.items()}
        (self.run_dir / 'final_metrics.json').write_text(json.dumps(final_metrics, indent=2), encoding='utf-8')
        if self.writer:
            self.writer.add_hparams(_tb_safe(self.hparams), final_metrics)
            self.writer.flush()
            self.writer.close()
        self.logger.info('FINAL ' + json.dumps(final_metrics))


def setup_run(hparams: Dict, run_dir: Optional[str] = None, **kwargs) -> RunLogger:
    return RunLogger(hparams, run_dir=run_dir, **kwargs)
