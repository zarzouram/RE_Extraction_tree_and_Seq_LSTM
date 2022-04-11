from typing import Tuple
from torch import Tensor
import torch.nn as nn


class EmbeddingLayer(nn.Module):

    def __init__(self, token_vocab_sz: int, pos_vocab_sz: int,
                 token_emb_sz: int, pos_emb_sz: int, token_pad_value: float,
                 pos_pad_value: float, dropout: float):
        super(EmbeddingLayer, self).__init__()

        self.token_embeddings = nn.Embedding(token_vocab_sz,
                                             token_emb_sz,
                                             padding_idx=token_pad_value)

        self.pos_embeddings = nn.Embedding(pos_vocab_sz,
                                           pos_emb_sz,
                                           padding_idx=pos_pad_value)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, pos: Tensor) -> Tuple(Tensor, Tensor):

        return (self.dropout(self.token_embeddings(tokens)),
                self.dropout(self.pos_embeddings(pos)))
