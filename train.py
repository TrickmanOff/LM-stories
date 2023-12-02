import contextlib
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Subset

from lib.data import TinyStoriesTextDataset, TokenizedTextDataset, TokenizedTextDataloader
from lib.encoder import BPETextEncoder
from lib.logger import WandbCM
from lib.model import SimpleTransformer
from lib.storage import ExperimentsStorage, ExternalStorage
from lib.text_generator import GreedyGenerator
from lib.train import ModelTrainer
from lib.utils import get_device


CONFIG_DIRPATH = Path(__file__).parent / 'config'


def init_logger(model_name: str = ''):
    logger_config = json.load(open(CONFIG_DIRPATH / 'logger.json', 'r'))

    @contextlib.contextmanager
    def logger_cm():
        try:
            with WandbCM(experiment_id=model_name, **logger_config) as wandb_logger:
                yield wandb_logger
        finally:
            pass
    return logger_cm


def train(num_epochs: int = 10,
          model_name: str = 'test',
          run_name: Optional[str] = None,
          encoder_name: str = 'tiny_stories_encoder',
          save_epochs_period: int = 1,
          dtype: torch.dtype = torch.float32,
          external_storage: Optional[ExternalStorage] = None):
    print('The training script is being run...')

    # awful but very simple config processing
    paths_config = json.load(open(CONFIG_DIRPATH / 'paths.json', 'r'))
    model_config = json.load(open(CONFIG_DIRPATH / 'model.json', 'r'))
    train_config = json.load(open(CONFIG_DIRPATH / 'train.json', 'r'))

    exps_storage = ExperimentsStorage(experiments_dir=paths_config.get('experiments_dir', 'saved/models'),
                                      encoders_dir=paths_config.get('encoders_dir', 'saved/encoders'))
    run_storage = exps_storage.get_run(exp_name=model_name, run_name=run_name, create_run_if_no=True)
    run_storage.save_config(model_config)
    # getting the dataset
    text_dataset = TinyStoriesTextDataset(paths_config['dataset_dir'])

    if external_storage is not None and encoder_name in external_storage.get_available_encoders():
        print(f'The trained text encoder {encoder_name} will be downloaded')
        external_storage.import_encoder(exps_storage, encoder_name)
    encoder = BPETextEncoder(name=encoder_name, encoder_dirpath=exps_storage.get_encoder_dirpath(encoder_name))
    if not encoder.is_trained:  # build encoder if it is not ready
        encoder.train(iter(Subset(text_dataset, torch.arange(10*100_000))))
    if external_storage is not None and encoder_name not in external_storage.get_available_encoders():
        external_storage.export_encoder(exps_storage, encoder_name)

    text_dataset = Subset(text_dataset, torch.arange(100_000))
    dataset = TokenizedTextDataset(text_dataset, encoder, paths_config['dataset_dir'])
    train_dataloader = TokenizedTextDataloader(dataset, batch_size=train_config['batch_size'])

    # logger
    logger_cm_fn = init_logger(model_name)

    # model
    device = get_device()
    model = SimpleTransformer(vocab_size=encoder.vocab_size, **model_config)
    model.to(dtype=dtype, device=device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])

    # scheduler
    scheduler = None

    # text generator
    text_generator = GreedyGenerator(model, encoder, max_len=200)
    prefixes_examples = [
        'Once upon a time',
        'Today Henry met Susie',
        'Bob has always wanted',
        'Each Sunday Helen',
        'When Tyler went on his first trip',
    ]

    trainer = ModelTrainer(model=model, optimizer=optimizer, run_storage=run_storage,
                           scheduler=scheduler, external_storage=external_storage,
                           save_epochs_period=save_epochs_period)
    trainer.train(train_dataloader, num_epochs=num_epochs,
                  logger_cm_fn=logger_cm_fn,
                  prefixes_examples=prefixes_examples,
                  text_generator=text_generator,
                  len_epoch=train_config['len_epoch'])

    return model


if __name__ == '__main__':
    train()
