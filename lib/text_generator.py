from abc import abstractmethod
from typing import List, Optional

import torch

from lib.encoder import TextEncoder


class TextGenerator:
    def __init__(self, model, encoder: TextEncoder, max_len: int):
        self._model = model
        self._encoder = encoder
        self._max_len = max_len

    def generate_text(self, prefix: str = '') -> str:
        prefix_indices = self._encoder.encode(prefix, add_bos=True, add_eos=False)
        text_indices = self._generate_indices(prefix_indices)
        return self._encoder.decode(text_indices)

    def generate_indices(self, prefix_indices: Optional[List[int]] = None) -> List[int]:
        if prefix_indices is None:
            prefix_indices = [self._encoder.BOS_ID]
        return self._generate_indices(prefix_indices)

    @abstractmethod
    def _generate_indices(self, prefix_indices: List[int]) -> List[int]:
        """
        Generates a sequence with the given prefix
        """
        raise NotImplementedError()


class GreedyGenerator(TextGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def _generate_indices(self, prefix_indices: List[int]) -> List[int]:
        self._model.eval()
        batch = torch.tensor(prefix_indices).unsqueeze(0)  # (1, len)
        while batch.shape[-1] - 2 < self._max_len and batch[0, -1] != self._encoder.EOS_ID:
            next_token_logits = self._model(batch)[0, -1]  # (vocab_size,)
            next_token = next_token_logits.argmax()
            batch = torch.concat((batch, torch.tensor([[next_token]])), axis=-1)

        all_indices = batch.squeeze(0).tolist()
        if all_indices[-1] != self._encoder.EOS_ID:
            all_indices.append(self._encoder.EOS_ID)
        return all_indices