import json
from pathlib import Path
from itertools import repeat

import requests
import torch
from tqdm import tqdm


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def get_device():
    return DEVICE


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def download_file(url, to_dirpath=None, to_filename=None):
    local_filename = to_filename or url.split('/')[-1]
    if to_dirpath is not None:
        local_filename = to_dirpath / local_filename
    chunk_size = 8192  # in bytes
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers['Content-length'])
        total = (total_size + chunk_size - 1) // chunk_size
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192), total=total, desc='Downloading file'):
                f.write(chunk)
    return local_filename
