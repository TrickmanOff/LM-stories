"""
TODO:
- checkpoints
- validation
"""
import contextlib
from typing import Any, Callable, ContextManager, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.logger import MLLogger
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
    def train(self, model, optimizer: torch.optim.Optimizer,
              train_dataloader: DataLoader,
              num_epochs: int,
              len_epoch: Optional[int] = None,
              val_dataloader: Optional[DataLoader] = None,
              scheduler: Optional[Any] = None,
              logger_cm_fn: Optional[Callable[[], ContextManager[MLLogger]]] = None,
              prefixes_examples: Optional[List[str]] = None,
              text_generator: Optional[TextGenerator] = None):
        epoch = 1

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
                    model, dataloader=train_dataloader, optimizer=optimizer, criterion=criterion,
                    scheduler=scheduler, len_epoch=len_epoch
                )

                if scheduler is not None:
                    last_lr = scheduler.get_last_lr()
                else:
                    last_lr = get_lr(optimizer)

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

                epoch += 1

        return model
