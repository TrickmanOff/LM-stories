import json
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lib.encoder import TextEncoder


class TextDataset(Dataset):
    def __init__(self, txts_filepath: Union[str, Path]):
        with open(txts_filepath, 'r') as file:
            self._texts = file.readlines()

    def __getitem__(self, item: int) -> str:
        return self._texts[item]

    def __len__(self) -> int:
        return len(self._texts)


class TinyStoriesTextDataset(TextDataset):
    # URL = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz'

    def __init__(self, data_dirpath: Union[str, Path] = 'data/tiny-stories',
                 filtered_txts_filename: str = 'filtered.txt',
                 max_text_len: int = 1_000):
        self._dataset_dirpath = Path(data_dirpath)
        # self._last_chunk = None
        # self._last_chunk_index = -1
        if not self._dataset_dirpath.exists():
            raise RuntimeError(f'Directory "{data_dirpath}" does not exist')
        # self._chunk_size = 100_000
        # self._last_chunk_size = len(json.load(open(self._chunks_filepaths[-1], 'r')))
        texts_filepath = self._dataset_dirpath / filtered_txts_filename
        if not texts_filepath.exists():
            chunks_filepaths = self.get_chunks()
            self._process_texts(texts_filepath, chunks_filepaths, max_text_len)
        super().__init__(texts_filepath)

    @staticmethod
    def _process_texts(target_filepath: Path, chunks_filepaths: List[Path], max_txt_len: int = 1_000):
        print('Filtering texts...')
        with open(target_filepath, 'w') as out_file:
            for chunk_filepath in tqdm(chunks_filepaths):
                chunk = json.load(open(chunk_filepath, 'r'))
                for item in chunk:
                    txt = item['story'].replace('\n', ' ') + '\n'
                    if len(txt) > max_txt_len:
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

    # def __getitem__(self, item: int) -> str:
    #     chunk_idx = item // self._chunk_size
    #     if self._last_chunk_index == chunk_idx:
    #         chunk = self._last_chunk
    #     else:
    #         chunk = json.load(open(self._chunks_filepaths[chunk_idx], 'r'))
    #     self._last_chunk = chunk
    #     self._last_chunk_index = chunk_idx
    #     return chunk[item % self._chunk_size]['story']
    #
    # def __len__(self) -> int:
    #     return self._chunk_size * (len(self._chunks_filepaths) - 1) + self._last_chunk_size


class TokenizedTextDataset(Dataset):
    def __init__(self, text_dataset: TextDataset, encoder: TextEncoder,
                 data_dirpath: Union[str, Path],
                 **kwargs):
        super().__init__()
        self._encoder = encoder
        self._text_dataset = text_dataset
        self._data_dirpath = Path(data_dirpath)
        self._tokens_seqs = self.get_or_load_tokenized_texts(**kwargs)

    def get_or_load_tokenized_texts(self, **kwargs):
        tokenized_texts_filepath = self._data_dirpath / 'tokenized_texts.npy'
        if not tokenized_texts_filepath.exists():
            self.tokenize_texts(tokenized_texts_filepath, **kwargs)
        data = np.load(str(tokenized_texts_filepath), allow_pickle=True)
        tokens_rows, lens = data[()]['tokenized_sequences'], data[()]['length']
        tokens_seqs = []
        for row, len in zip(tokens_rows, lens):
            tokens_seqs.append(row[:len].tolist())
        return tokens_seqs

    def tokenize_texts(self, target_filepath: Path, **kwargs):
        print('Tokenizing texts...')
        tokens_seqs = [self.encoder.encode(text, **kwargs) for text in tqdm(self._text_dataset)]
        lens = np.array([len(seq) for seq in tokens_seqs], dtype=np.int32)
        max_len = lens.max()
        seqs_matrix = np.full((len(self._text_dataset), max_len), fill_value=self.encoder.PAD_ID, dtype=np.int32)
        for i, seq in enumerate(tokens_seqs):
            seqs_matrix[i, :len(seq)] = seq

        data = {'tokenized_sequences': seqs_matrix, 'length': lens}
        np.save(str(target_filepath), data)

    @property
    def encoder(self) -> TextEncoder:
        return self._encoder

    def __getitem__(self, item: int) -> List[int]:
        return self._tokens_seqs[item]

    def __len__(self) -> int:
        return len(self._tokens_seqs)


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
    def __init__(self, dataset: TokenizedTextDataset, batch_size: int = 8):
        super().__init__(dataset, batch_size=batch_size, collate_fn=collate_fn)
