import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len: int = 512):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        pos = torch.arange(max_len)
        i = torch.arange(0, embed_dim, 2)
        pos, i = torch.meshgrid(pos, i, indexing='ij')
        arg = pos / (10_000**(i / embed_dim))

        pe = torch.zeros(max_len, embed_dim)

        pe[:, ::2] = torch.sin(arg)
        pe[:, 1::2] = torch.cos(arg)[:, :arg.shape[1] - (embed_dim % 2)]

        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        """
        x: (B, len, embed_dim)
        """
        x = x + self.pe[:, :x.shape[1], :]
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int, max_len: int = 5_000,
                 nhead: int = 8,
                 **kwargs):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=embed_dim,
                                  padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)
        self.transformer_encoder = nn.TransformerEncoderLayer(embed_dim, nhead, batch_first=True,
                                                              **kwargs)

        self.head = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(self, seq_len: int, device=None) -> torch.BoolTensor:
        if torch.__version__[:3] == 2.1:
            return nn.Transformer.generate_square_subsequent_mask(seq_len,
                                                                  device=seq_len,
                                                                  dtype=torch.bool)
        else:
            mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
            mask = torch.triu(mask, 1)
            return mask

    def forward(self, sequences: Tensor, **kwargs) -> Tensor:
        """
        sequences: (B, max_len)

        :return next_tokens_logits: (B, max_len, vocab_size)
        """
        src_mask = self.generate_square_subsequent_mask(sequences.shape[-1],
                                                        device=sequences.device)  # (max_len, max_len)
        src_key_padding_mask = (sequences == 0).to(sequences.device)  # (B, max_len)

        embed_sequences = self.embed(sequences)              # (B, max_len, embed_dim)
        embed_sequences = self.pos_encoder(embed_sequences)  # (B, max_len, embed_dim)

        next_tokens_embeds = self.transformer_encoder(embed_sequences,
                                                      src_mask=src_mask,
                                                      src_key_padding_mask=src_key_padding_mask,
                                                      is_causal=True)   # (B, max_len, embed_dim)
        next_tokens_logits = self.head(next_tokens_embeds)  # (B, max_len, vocab_size)
        return next_tokens_logits
