# import torch
import torch.nn as nn
from models.treelstm import TreeLSTM


class DepLayer(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 tree_h_size: int,
                 tree_bidir: bool = True):
        super(DepLayer).__init__()
        self.deptree = TreeLSTM(embedding_dim=embedding_dim,
                                h_size=tree_h_size,
                                bidirectional=tree_bidir)
