import contextlib
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Subset

from lib.data import TinyStoriesTextDataset, TokenizedTextDataset, TokenizedTextDataloader
from lib.encoder import BPETextEncoder
from lib.logger import WandbCM
from lib.model import SimpleTransformer
from lib.storage import ExperimentsStorage, ExternalStorage
from lib.train import ModelTrainer
from lib.utils import get_device, get_params_count


CONFIG_DIRPATH = Path(__file__).parent / 'config'


def init_logger(wandb_run_name: str = ''):
    logger_config = json.load(open(CONFIG_DIRPATH / 'logger.json', 'r'))

    @contextlib.contextmanager
    def logger_cm():
        try:
            with WandbCM(experiment_id=wandb_run_name, **logger_config) as wandb_logger:
                yield wandb_logger
        finally:
            pass
    return logger_cm


def train(model_name: str = 'test',
          wandb_run_name: Optional[str] = None,
          num_epochs: Optional[int] = None,
          run_name: Optional[str] = None,
          encoder_name: str = 'tiny_stories_encoder_4k',
          save_epochs_period: int = 1,
          external_storage: Optional[ExternalStorage] = None,
          model_config: Optional[Dict] = None,
          train_config: Optional[Dict] = None,
          shuffle_dataset: bool = True):
    print('The training script is being run...')
    wandb_run_name = model_name if wandb_run_name is None else wandb_run_name

    # awful but very simple config processing
    paths_config = json.load(open(CONFIG_DIRPATH / 'paths.json', 'r'))
    model_config = json.load(open(CONFIG_DIRPATH / 'model.json', 'r')) if model_config is None else model_config
    train_config = json.load(open(CONFIG_DIRPATH / 'train.json', 'r')) if train_config is None else train_config

    exps_storage = ExperimentsStorage(experiments_dir=paths_config.get('experiments_dir', 'saved/models'),
                                      encoders_dir=paths_config.get('encoders_dir', 'saved/encoders'))
    run_storage = exps_storage.get_run(exp_name=model_name, run_name=run_name, create_run_if_no=True)
    run_storage.save_config(model_config)

    text_dataset = None

    if external_storage is not None and encoder_name in external_storage.get_available_encoders():
        print(f'The trained text encoder {encoder_name} will be downloaded')
        external_storage.import_encoder(exps_storage, encoder_name)
    encoder = BPETextEncoder(name=encoder_name, encoder_dirpath=exps_storage.get_encoder_dirpath(encoder_name))
    if not encoder.is_trained:  # build encoder if it is not ready
        text_dataset = TinyStoriesTextDataset(paths_config['dataset_dir'])
        encoder.train(iter(Subset(text_dataset, torch.arange(10*100_000))))
    if external_storage is not None and encoder_name not in external_storage.get_available_encoders():
        external_storage.export_encoder(exps_storage, encoder_name)

    try:
        dataset = TokenizedTextDataset(None, encoder, paths_config['dataset_dir'])
        train_dataloader = TokenizedTextDataloader(dataset, batch_size=train_config['batch_size'],
                                                   shuffle=shuffle_dataset)
    except Exception:
        if text_dataset is None:
            text_dataset = TinyStoriesTextDataset(paths_config['dataset_dir'])
        dataset = TokenizedTextDataset(text_dataset, encoder, paths_config['dataset_dir'])
        train_dataloader = TokenizedTextDataloader(dataset, batch_size=train_config['batch_size'],
                                                   shuffle=shuffle_dataset)

    # logger
    logger_cm_fn = init_logger(wandb_run_name)

    # model
    device = get_device()
    model = SimpleTransformer(vocab_size=encoder.vocab_size, max_len=512, **model_config)
    print(f'Model has {get_params_count(model)} trainable parameters')

    model.to(device=device)

    # optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=train_config['lr'])

    # scheduler
    scheduler = None

    # text generator
    prefixes_examples = [
        'Once upon a time',
        'Today Henry met Susie',
        'Bob has always wanted',
        'Each Sunday Helen',
        'When Tyler went on his first trip',
        'Misha has been playing with his robot for a long time',
    ]

    trainer = ModelTrainer(model=model, optimizer=optimizer, run_storage=run_storage,
                           text_encoder=encoder,
                           scheduler=scheduler, external_storage=external_storage,
                           save_epochs_period=save_epochs_period)
    trainer.train(train_dataloader, num_epochs=num_epochs,
                  logger_cm_fn=logger_cm_fn,
                  prefixes_examples=prefixes_examples,
                  len_epoch=train_config['len_epoch'],
                  max_gen_seq_len=256)

    return model


if __name__ == '__main__':
    train()
