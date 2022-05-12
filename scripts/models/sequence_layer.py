import torch.nn as nn
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

    def forward(self, seq: Tensor, lengths: Tensor, padding_value: float):

        lengths = lengths.cpu()

        x_packed = pack_padded_sequence(seq, lengths=lengths, batch_first=True)

        output_packed, _ = self.lstm(x_packed)

        output, _ = pad_packed_sequence(output_packed,
                                        batch_first=True,
                                        padding_value=padding_value,
                                        total_length=lengths[0])

        return self.dropout(output)
