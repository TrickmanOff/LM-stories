import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lib.encoder import TextEncoder
from lib.utils import download_file


class TextDataset(Dataset):
    def __init__(self, txts_filepath: Union[str, Path]):
        with open(txts_filepath, 'r') as file:
            self._texts = [line.strip() for line in file.readlines()]

    def __getitem__(self, item: int) -> str:
        return self._texts[item]

    def __len__(self) -> int:
        return len(self._texts)


class TinyStoriesTextDataset(TextDataset):
    URL = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz'

    def __init__(self, data_dir: Union[str, Path] = 'data/tiny-stories',
                 data_writeable_dir: Optional[Union[str, Path]] = None,
                 filtered_txts_filename: str = 'filtered.txt',
                 max_text_len: Optional[int] = 1_000,
                 chunks_cnt: Optional[int] = None):
        if data_writeable_dir is None:
            data_writeable_dir = Path(data_dir)
        self._data_writeable_dir = data_writeable_dir
        self._dataset_dirpath = Path(data_dir)

        self._data_writeable_dir.mkdir(parents=True, exist_ok=True)

        if not self._dataset_dirpath.exists():
            raise RuntimeError(f'Directory "{self._dataset_dirpath}" does not exist')

        texts_filepath = self._data_writeable_dir / filtered_txts_filename
        if not texts_filepath.exists():
            chunks_filepaths = self.get_chunks()
            if len(chunks_filepaths) == 0:
                self.download_dataset(self._dataset_dirpath)
                chunks_filepaths = self.get_chunks()
            if chunks_cnt is not None:
                chunks_filepaths = chunks_filepaths[:chunks_cnt]
            self._process_texts(texts_filepath, chunks_filepaths, max_text_len)
        super().__init__(texts_filepath)

    @staticmethod
    def _process_texts(target_filepath: Path, chunks_filepaths: List[Path], max_text_len: Optional[int] = 1_000):
        print('Filtering texts...')
        with open(target_filepath, 'w') as out_file:
            for chunk_filepath in tqdm(chunks_filepaths):
                chunk = json.load(open(chunk_filepath, 'r'))
                for item in chunk:
                    txt = item['story'].replace('\n', ' ') + '\n'
                    if max_text_len is not None and len(txt) > max_text_len:
                        continue
                    out_file.write(txt)

    def get_chunks(self) -> List[Path]:
        chunk_filenames = sorted(
            filename
            for filename in os.listdir(self._dataset_dirpath)
            if os.path.splitext(filename)[-1] == '.json'
        )
        chunk_filepaths = [self._dataset_dirpath / filename for filename in chunk_filenames]
        return chunk_filepaths

    def download_dataset(self, dirpath: Path):
        arch_filename = self.URL.split('/')[-1]
        arch_filepath = dirpath / arch_filename
        if not os.path.exists(arch_filepath):
            print('Downloading TinyStories dataset...')
            download_file(self.URL, dirpath, arch_filename)
        print('Unpacking archive...')
        shutil.unpack_archive(arch_filepath, dirpath)
        os.remove(str(arch_filepath))


class TokenizedTextDataset(Dataset):
    def __init__(self, text_dataset: TextDataset,
                 encoder: TextEncoder,
                 data_dirpath: Union[str, Path],
                 max_seq_len: Optional[int] = 256,
                 **kwargs):
        super().__init__()
        self._encoder = encoder
        self._text_dataset = text_dataset
        self._data_dirpath = Path(data_dirpath)
        self._max_seq_len = max_seq_len
        self._tokenized_texts_data, self._tokenized_texts_index = self.get_or_load_tokenized_texts(**kwargs)

    def get_or_load_tokenized_texts(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: data, index
        """
        data_filepath = self._data_dirpath / 'tokenized_texts_data.npy'
        index_filepath = self._data_dirpath / 'tokenized_texts_index.npy'
        if not data_filepath.exists() or not index_filepath.exists():
            self.tokenize_texts(data_filepath, index_filepath, **kwargs)
        data = np.load(str(data_filepath))
        index = np.load(str(index_filepath))
        return data, index

    def tokenize_texts(self, data_filepath: Path, index_filepath: Path, **kwargs):
        print('Tokenizing texts...')
        tokens_seqs = []
        for text in tqdm(self._text_dataset):
            encoded_text = np.array(self.encoder.encode(text, **kwargs), dtype=np.int16)
            if self._max_seq_len is None or len(encoded_text) <= self._max_seq_len:
                tokens_seqs.append(encoded_text)

        lens = np.array([0] + [len(seq) for seq in tokens_seqs], dtype=np.int64)
        cum_lens = np.cumsum(lens)
        flattened_tokens_seqs = np.hstack(tokens_seqs)

        np.save(str(data_filepath), flattened_tokens_seqs)
        np.save(str(index_filepath), cum_lens)

    @property
    def encoder(self) -> TextEncoder:
        return self._encoder

    def __getitem__(self, item: int) -> List[int]:
        index = self._tokenized_texts_index
        l = index[item]
        r = index[item + 1]
        # [l, r)
        return self._tokenized_texts_data[l:r].tolist()

    def __len__(self) -> int:
        return len(self._tokenized_texts_index) - 1


def collate_fn(tokens_seqs: List[List[int]]):
    """
    batch:

    seqs :   tensor of shape (B, max_len)
    length : tensor of shape (B,)
    """
    lengths = torch.LongTensor([len(seq) for seq in tokens_seqs])
    max_len = lengths.max()
    seqs_matrix = torch.zeros((len(tokens_seqs), max_len), dtype=torch.long)
    for i, seq in enumerate(tokens_seqs):
        seqs_matrix[i, :len(seq)] = torch.LongTensor(seq)
    return {'sequences': seqs_matrix, 'lengths': lengths}


class TokenizedTextDataloader(DataLoader):
    def __init__(self, dataset: TokenizedTextDataset, batch_size: int = 8, shuffle: bool = True, drop_last: bool = True):
        super().__init__(dataset, batch_size=batch_size, collate_fn=collate_fn,
                         shuffle=shuffle, drop_last=drop_last)
