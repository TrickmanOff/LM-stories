from abc import abstractmethod
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from lib.encoder import TextEncoder


class TextGenerator:
    def __init__(self, model, encoder: TextEncoder, max_len: int = 256):
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
        device = next(self._model.parameters()).device
        batch = torch.tensor(prefix_indices).unsqueeze(0).to(device)  # (1, len)
        while batch.shape[-1] - 2 < self._max_len and batch[0, -1] != self._encoder.EOS_ID:
            next_token_logits = self._model(batch)[0, -1]  # (vocab_size,)
            next_token = next_token_logits.argmax()
            batch = torch.concat((batch, torch.tensor([[next_token]], device=device)), axis=-1)

        all_indices = batch.squeeze(0).tolist()
        if all_indices[-1] != self._encoder.EOS_ID:
            all_indices.append(self._encoder.EOS_ID)
        return all_indices


class RandomGenerator(TextGenerator):
    def __init__(self, temp: float = 1., top_k: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = temp
        self.top_k = top_k  # at each step take only k top probable tokens and then sample from them

    @torch.no_grad()
    def _generate_indices(self, prefix_indices: List[int]) -> List[int]:
        self._model.eval()
        device = next(self._model.parameters()).device
        batch = torch.tensor(prefix_indices).unsqueeze(0).to(device)  # (1, len)
        while batch.shape[-1] - 2 < self._max_len and batch[0, -1] != self._encoder.EOS_ID:
            next_token_logits = self._model(batch)[0, -1].float().cpu()  # (vocab_size,)
            if self.temp == 0. or self.top_k is not None:
                if self.temp == 0:
                    considered_tokens = [next_token_logits.argmax()]
                elif self.top_k is not None:
                    considered_tokens = torch.argsort(next_token_logits)[-self.top_k:]
                new_next_token_logits = torch.full_like(next_token_logits, -np.inf)
                new_next_token_logits[considered_tokens] = next_token_logits[considered_tokens]
                next_token_logits = new_next_token_logits

            if self.temp != 0.:
                next_token_logits /= self.temp

            next_token_probs = F.softmax(next_token_logits, dim=0)
            next_token = np.random.choice(range(len(next_token_logits)), p=next_token_probs.numpy(), size=1)[0]
            batch = torch.concat((batch, torch.tensor([[next_token]], device=device)), axis=-1)

        all_indices = batch.squeeze(0).tolist()
        if all_indices[-1] != self._encoder.EOS_ID:
            all_indices.append(self._encoder.EOS_ID)
        return all_indices

