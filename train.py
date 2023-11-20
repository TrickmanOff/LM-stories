import contextlib
import json
from pathlib import Path

import torch

from lib.data import TinyStoriesTextDataset, TokenizedTextDataset, TokenizedTextDataloader
from lib.encoder import BPETextEncoder
from lib.logger import WandbCM
from lib.model import SimpleTransformer
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


def train(num_epochs: int = 10):
    model_name = 'test'

    paths_config = json.load(open(CONFIG_DIRPATH / 'paths.json', 'r'))
    model_config = json.load(open(CONFIG_DIRPATH / 'model.json', 'r'))
    train_config = json.load(open(CONFIG_DIRPATH / 'train.json', 'r'))

    encoder = BPETextEncoder(name='tiny_stories_encoder')

    text_dataset = TinyStoriesTextDataset(paths_config['dataset_dir'])
    dataset = TokenizedTextDataset(text_dataset, encoder, paths_config['dataset_dir'])
    train_dataloader = TokenizedTextDataloader(dataset, batch_size=train_config['batch_size'])

    # logger
    logger_cm_fn = init_logger(model_name)

    # model
    device = get_device()
    model = SimpleTransformer(vocab_size=encoder.vocab_size, **model_config)
    model.to(device)

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

    trainer = ModelTrainer()
    trainer.train(model, optimizer, train_dataloader, num_epochs=num_epochs, scheduler=scheduler,
                  logger_cm_fn=logger_cm_fn,
                  prefixes_examples=prefixes_examples,
                  text_generator=text_generator,
                  len_epoch=train_config['len_epoch'])

    return model


if __name__ == '__main__':
    train()
