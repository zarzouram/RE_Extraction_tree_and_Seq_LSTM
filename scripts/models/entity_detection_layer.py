from torch import Tensor
import torch.nn as nn


class EntityDetection(nn.Module):

    def __init__(
        self,
        entl_vocab_sz: int,
        entl_emb_sz: int,
        input_sz: int,
        output_sz: int,
        hidden_sz: int,
        dropout: float,
        entl_pad_value: float,
    ):
        super(EntityDetection, self).__init__()
        # self.entl_embeddings = nn.Embedding(entl_vocab_sz,
        #                                     entl_emb_sz,
        #                                     padding_idx=entl_pad_value)
        raise NotImplementedError

    def forward(self, inputs: Tensor, epsilon: float):
        raise NotImplementedError

    def inference(self, inputs: Tensor):
        raise NotImplementedError
