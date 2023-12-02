"""
TODO:
- checkpoints
- validation
"""
import contextlib
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.encoder import TextEncoder
from lib.logger import MLLogger
from lib.storage import ExternalStorage, RunStorage
from lib.text_generator import GreedyGenerator, RandomGenerator
from lib.utils import get_device, get_grad_norm, get_lr, inf_loop


def move_batch_to_device(batch):
    device = get_device()
    batch['sequences'] = batch['sequences'].to(device)


def train_epoch(model, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, scheduler: Optional[Any] = None,
                len_epoch: Optional[int] = None) -> Dict[str, Any]:
    model.train()

    processed_tokens = 0
    total_loss = 0.
    total_seqs_len = 0
    sum_grad_norms = 0.
    num_batches = 0

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
        sum_grad_norms += get_grad_norm(model)
        model.zero_grad()
        num_batches += 1

        tgt_seq_lens = batch['lengths'] - 1  # all tokens except for the first one are predicted and used in loss
        seqs_len = sum(tgt_seq_lens)
        processed_tokens += seqs_len
        total_loss += loss.item() * seqs_len
        total_seqs_len += seqs_len

        if batch_index + 1 >= len_epoch:
            break

    return {
        'loss': total_loss / total_seqs_len,
        'tokens': processed_tokens,
        'grad_norm': sum_grad_norms / num_batches,
    }


class ModelTrainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 run_storage: RunStorage,
                 text_encoder: TextEncoder,
                 scheduler: Optional[Any] = None,
                 external_storage: Optional[ExternalStorage] = None,
                 save_epochs_period: int = -1,
                 resume_checkpoint_filepath: Optional[Path] = None):
        self.run_storage = run_storage
        self.external_storage = external_storage
        self.save_epochs_period = save_epochs_period

        self.text_encoder = text_encoder

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.start_epoch = 1

        if resume_checkpoint_filepath is not None:
            self._resume_checkpoint(resume_checkpoint_filepath)

    def train(self,
              train_dataloader: DataLoader,
              num_epochs: Optional[int] = None,
              len_epoch: Optional[int] = None,
              val_dataloader: Optional[DataLoader] = None,
              logger_cm_fn: Optional[Callable[[], ContextManager[MLLogger]]] = None,
              prefixes_examples: Optional[List[str]] = None,
              max_gen_seq_len: int = 256):
        """
        :param num_epochs: if None, then training is infinite
        """
        if num_epochs is None:
            print('The training loop will not stop as `num_epochs` is None')
        epoch = self.start_epoch

        criterion = nn.CrossEntropyLoss(ignore_index=0)

        if logger_cm_fn is None or logger_cm_fn() is None:
            logger_cm = contextlib.nullcontext(None)
        else:
            logger_cm = logger_cm_fn()

        if len_epoch is not None:
            train_dataloader = inf_loop(train_dataloader)

        total_processed_tokens_cnt = 0

        with logger_cm as logger:
            while num_epochs is None or epoch <= num_epochs:
                if logger is not None:
                    logger.log_metrics(data={}, period='batch', period_index=epoch * len_epoch, commit=False)

                epoch_metrics = train_epoch(
                    self.model, dataloader=train_dataloader, optimizer=self.optimizer, criterion=criterion,
                    scheduler=self.scheduler, len_epoch=len_epoch
                )
                total_processed_tokens_cnt += epoch_metrics['tokens']

                if self.scheduler is not None:
                    last_lr = self.scheduler.get_last_lr()
                else:
                    last_lr = get_lr(self.optimizer)

                if logger is not None:
                    log_data = {
                        'train/loss': epoch_metrics['loss'],
                        'train/processed_tokens': total_processed_tokens_cnt,
                        'train/grad_norm': epoch_metrics['grad_norm'],
                        # 'val/loss': val_loss.item(),
                        'lr': last_lr,
                    }
                    logger.log_metrics(data=log_data, period='batch', commit=False)

                if prefixes_examples is not None and logger is not None:
                    self._log_predictions(prefixes_examples, logger)

                if logger is not None:
                    logger.commit(period='batch')

                if epoch % self.save_epochs_period == 0:
                    self._save_checkpoint(epoch + 1)

                epoch += 1

    def _log_predictions(self, prefixes: List[str], logger: MLLogger,
                         max_gen_seq_len: int = 256):
        print('Generating examples...')

        gen_kwargs = {
            'model': self.model,
            'encoder': self.text_encoder,
            'max_len': max_gen_seq_len,
        }
        generators = {
            'greedy': GreedyGenerator(**gen_kwargs),
            'random, t=0,5': RandomGenerator(temp=0.5, **gen_kwargs),
            'random, t=1,0': RandomGenerator(temp=1., **gen_kwargs),
        }

        columns = ['prefix'] + list(generators.keys())
        data = []
        for prefix in prefixes:
            row = [prefix]
            for generator in generators.values():
                gen_text = generator.generate_text(prefix)
                row.append(gen_text)
            data.append(row)

        logger.log_table('generated examples', columns, data)

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
