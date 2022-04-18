import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch import Tensor


class SeqEncoder(nn.Module):

    def __init__(self,
                 input_size: int,
                 h_size: int,
                 num_layers: int,
                 lstm_dropout: float,
                 output_dropout: float,
                 bidirectional: bool = True):
        super(SeqEncoder, self).__init__()

        self.h_size = h_size
        self.num_layers = num_layers
        self.num_dir = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=h_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=lstm_dropout if num_layers > 1 else 0)

        self.dropout = nn.Dropout(output_dropout)
        # learnable initial states
        self.init_states()

    def init_states(self):
        state_size = self.num_layers * self.num_dir
        h0 = torch.zeros(state_size, 1, self.h_size)
        c0 = torch.zeros(state_size, 1, self.h_size)
        xavier_normal_(h0, gain=nn.init.calculate_gain("sigmoid"))
        xavier_normal_(c0, gain=nn.init.calculate_gain("sigmoid"))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)

    def forward(self, seq: Tensor, lengths: Tensor, padding_value: float):

        b_sz = seq.size(0)
        lengths = lengths.cpu()

        x_packed = pack_padded_sequence(seq, lengths=lengths, batch_first=True)

        init_states = (self.h0.repeat(1, b_sz, 1), self.c0.repeat(1, b_sz, 1))
        output_packed, _ = self.lstm(x_packed, init_states)

        output, _ = pad_packed_sequence(output_packed,
                                        batch_first=True,
                                        padding_value=padding_value,
                                        total_length=lengths[0])

        return self.dropout(output)
