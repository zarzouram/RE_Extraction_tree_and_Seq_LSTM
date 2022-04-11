# import torch
from torch import Tensor
from dgl import DGLGraph

import torch
import torch.nn as nn

from models.treelstm import TreeLSTM


class DepLayer(nn.Module):

    def __init__(self,
                 input_size: int,
                 tree_h_size: int,
                 tree_bidir: bool = True):
        super(DepLayer, self).__init__()
        self.deptree = TreeLSTM(input_size=input_size,
                                h_size=tree_h_size,
                                bidirectional=tree_bidir)

    def forward(self, g: DGLGraph):
        hp = self.deptree(g)  # type: Tensor

        # section 3.6 construct dp = [↑hpA; ↓hp1; ↓hp2]
        # ↑hpA: hidden state of dep_graphs' root
        # ↓hp1: hidden state of the first token in the candidate pair
        # ↓hp2: hidden state of the second token in the candidate pair
        # get ids of roots and tokens in relation
        num_dir = 2 if self.deptree.bidirectional else 1
        h_size = hp.size(-1)
        hp = hp.view(-1, num_dir, h_size)

        e1_nidx = g.ndata["e1"] == 1
        e2_nidx = g.ndata["e2"] == 1
        hp1 = hp[e1_nidx, 0, :].squeeze()  # ↓hp1
        hp2 = hp[e2_nidx, 0, :].squeeze()  # ↓hp2
        hp_d12 = torch.cat((hp1, hp2), dim=-1)  # relation from e1 to e2
        hp_d21 = torch.cat((hp2, hp1), dim=-1)  # relation from e2 to e1

        if self.deptree.bidirectional:
            root = g.ndata["root"] == 1
            hpA = hp[root, 1, :].squeeze()  # ↑hpA
            hp_d12 = torch.cat((hpA, hp_d12), dim=-1)  # relation from e1 to e2
            hp_d12 = torch.cat((hpA, hp_d21), dim=-1)  # relation from e1 to e2

        return hp_d12, hp_d21
