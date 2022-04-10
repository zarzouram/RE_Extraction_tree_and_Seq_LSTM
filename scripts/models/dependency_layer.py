# import torch
from typing import Tuple
import torch.nn as nn
from models.treelstm import TreeLSTM
from dgl import DGLGraph


class DepLayer(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 tree_h_size: int,
                 tree_bidir: bool = True):
        super(DepLayer).__init__()
        self.deptree = TreeLSTM(embedding_dim=embedding_dim,
                                h_size=tree_h_size,
                                bidirectional=tree_bidir)

    def forward(self, g: DGLGraph, ent_candidates: Tuple[int, int]):
        hp = self.deptree(g)

        # section 3.6 construct dp = [↑hpA; ↓hp1; ↓hp2]
        # ↑hpA: hidden state of dep_graphs' root
        # ↓hp1: hidden state of the first token in the candidate pair
        # ↓hp2: hidden state of the second token in the candidate pair
        # get ids of roots and tokens in relation
        root_id = g.ndata["root"] == 1
