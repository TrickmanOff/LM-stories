import os
from abc import abstractmethod
from pathlib import Path
from typing import Iterable, List, Union

import sentencepiece as spm


class TextEncoder:
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    @abstractmethod
    def encode(self, texts: Union[str, List[str]],
               add_bos: bool = True, add_eos: bool = True) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError()

    @abstractmethod
    def decode(self, encoded_texts: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        raise NotImplementedError()

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError()


class BPETextEncoder(TextEncoder):
    def __init__(self, encoder_dirpath: Union[str, Path] = 'saved/encoders', vocab_size: int = 30_000, name: str = 'encoder'):
        super().__init__()
        encoder_dirpath = Path(encoder_dirpath)
        encoder_dirpath.mkdir(parents=True, exist_ok=True)
        self._encoder_dirpath = encoder_dirpath
        self._vocab_size = vocab_size
        self._model_prefix = str(encoder_dirpath / name)
        self._tokenizer = None
        if not self.load():
            print('Call train() for the BPE encoder')

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def train(self, corpus: Union[Iterable, Path, str], verbose: bool = False):
        print(f'Training SentencePiece tokenizer')
        kwargs = {
            'model_type': 'bpe',
            'model_prefix': self._model_prefix,
            'vocab_size': self.vocab_size,
            'pad_id': self.PAD_ID,
            'unk_id': self.UNK_ID,
            'bos_id': self.BOS_ID,
            'eos_id': self.EOS_ID,
            'unk_surface': ' <unk>',
            'minloglevel': (0 if verbose else 1),
        }
        if isinstance(corpus, Path) or isinstance(corpus, str):
            kwargs['input'] = corpus
        else:
            kwargs['sentence_iterator'] = iter(corpus)
        spm.SentencePieceTrainer.train(**kwargs)
        print('Successfully trained SentencePiece tokenizer')
        self.load()

    def load(self) -> bool:
        """
        :return: True if successfully loaded
        """
        self._encoder_dirpath.mkdir(exist_ok=True)
        model_filepath = self._model_prefix + '.model'
        if os.path.exists(model_filepath):
            sp = spm.SentencePieceProcessor()
            sp.load(model_filepath)
            self._tokenizer = sp
            print('SentencePiece tokenizer loaded')
            return True
        else:
            print('No saved SentencePiece tokenizer model found')
            return False

    def encode(self, texts: Union[str, List[str]],
               add_bos: bool = True, add_eos: bool = True) -> Union[List[int], List[List[int]]]:
        return self._tokenizer.encode_as_ids(texts, add_bos=add_bos, add_eos=add_eos)

    def decode(self, encoded_texts: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        return self._tokenizer.decode_ids(encoded_texts)
