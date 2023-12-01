"""
TODO:
- checkpoints
- validation
"""
import contextlib
from pathlib import Path
from typing import Any, Callable, ContextManager, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.logger import MLLogger
from lib.storage import ExternalStorage, RunStorage
from lib.text_generator import TextGenerator
from lib.utils import get_lr, inf_loop, get_device


def move_batch_to_device(batch):
    device = get_device()
    batch['sequences'] = batch['sequences'].to(device)


def train_epoch(model, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, scheduler: Optional[Any] = None,
                len_epoch: Optional[int] = None) -> float:
    model.train()

    total_loss = 0.
    total_seqs_len = 0

    device = next(model.parameters()).device

    total_len = len(dataloader) if len_epoch is None else len_epoch
    for batch_index, batch in enumerate(tqdm(dataloader, desc='Training epoch', total=total_len)):
        move_batch_to_device(batch)
        # batch['sequences']: (B, max_len)
        next_tokens_logits = model(**batch)  # (B, max_len, vocab_size)
        tokens_logits = next_tokens_logits[:, :-1].transpose(1, 2)  # (B, vocab_size, max_len-1)
        tgt_tokens = batch['sequences'][:, 1:].to(device)  # (B, max_len-1)

        loss = criterion(tokens_logits, tgt_tokens)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.zero_grad()

        tgt_seq_lens = batch['lengths']
        seqs_len = sum(tgt_seq_lens)
        total_loss += loss * seqs_len
        total_seqs_len += seqs_len

        if batch_index + 1 >= len_epoch:
            break

    return total_loss / total_seqs_len


class ModelTrainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 run_storage: RunStorage,
                 scheduler: Optional[Any] = None,
                 external_storage: Optional[ExternalStorage] = None,
                 save_epochs_period: int = -1,
                 resume_checkpoint_filepath: Optional[Path] = None):
        self.run_storage = run_storage
        self.external_storage = external_storage
        self.save_epochs_period = save_epochs_period

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.start_epoch = 1

        if resume_checkpoint_filepath is not None:
            self._resume_checkpoint(resume_checkpoint_filepath)

    def train(self,
              train_dataloader: DataLoader,
              num_epochs: int,
              len_epoch: Optional[int] = None,
              val_dataloader: Optional[DataLoader] = None,
              logger_cm_fn: Optional[Callable[[], ContextManager[MLLogger]]] = None,
              prefixes_examples: Optional[List[str]] = None,
              text_generator: Optional[TextGenerator] = None):
        epoch = self.start_epoch

        criterion = nn.CrossEntropyLoss(ignore_index=0)

        if logger_cm_fn is None or logger_cm_fn() is None:
            logger_cm = contextlib.nullcontext(None)
        else:
            logger_cm = logger_cm_fn()

        if len_epoch is not None:
            train_dataloader = inf_loop(train_dataloader)

        with logger_cm as logger:
            while epoch <= num_epochs:
                if logger is not None:
                    logger.log_metrics(data={}, period='epoch', period_index=epoch, commit=False)

                train_loss = train_epoch(
                    self.model, dataloader=train_dataloader, optimizer=self.optimizer, criterion=criterion,
                    scheduler=self.scheduler, len_epoch=len_epoch
                )

                if self.scheduler is not None:
                    last_lr = self.scheduler.get_last_lr()
                else:
                    last_lr = get_lr(self.optimizer)

                if logger is not None:
                    log_data = {
                        'train/loss': train_loss.item(),
                        # 'val/loss': val_loss.item(),
                        'lr': last_lr,
                    }
                    logger.log_metrics(data=log_data, period='epoch', commit=False)

                if prefixes_examples is not None and logger is not None:
                    print('Generating examples...')
                    generated_texts = [
                        text_generator.generate_text(prefix=prefix)
                        for prefix in prefixes_examples
                    ]
                    logger.log_generated_texts(generated_texts)

                if logger is not None:
                    logger.commit(period='epoch')

                if (epoch - 1) % self.save_epochs_period == 0:
                    self._save_checkpoint(epoch + 1)

                epoch += 1

    def _save_checkpoint(self, epoch: int):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()

        checkpoint_name = 'last'
        print('Saving checkpoint...')
        self.run_storage.save_checkpoint(checkpoint_name, state)
        if self.external_storage is not None:
            print('Exporting checkpoint...')
            self.external_storage.export_checkpoint(self.run_storage, checkpoint_name)

    def _resume_checkpoint(self, checkpoint_filepath: Path):
        state = torch.load(checkpoint_filepath)
        self.start_epoch = state['epoch']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
