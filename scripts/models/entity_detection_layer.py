from collections import OrderedDict

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, calculate_gain


class EntityDetection(nn.Module):

    def __init__(
        self,
        entl_vocab_sz: int,
        entl_emb_sz: int,
        input_sz: int,
        hidden_sz: int,
        hidden_dropout: float,
        label_dropout: float,
        entl_pad_value: float,
    ):
        super(EntityDetection, self).__init__()

        self.output_size = entl_vocab_sz
        self.entl_emb_sz = entl_emb_sz
        self.pad = entl_pad_value

        self.entl_embeddings = nn.Embedding(entl_vocab_sz,
                                            entl_emb_sz,
                                            padding_idx=entl_pad_value)
        self.ent_clf = nn.Sequential(
            OrderedDict([("ht1", nn.Linear(input_sz, hidden_sz)),
                         ("tanht", nn.Tanh()),
                         ("dropoutt", nn.Dropout(hidden_dropout)),
                         ("ht2", nn.Linear(hidden_sz, entl_vocab_sz))]))

        self.dropout = nn.Dropout(label_dropout)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.init_clf()

    def init_clf(self):
        xavier_uniform_(self.ent_clf[0].weight.data,
                        gain=calculate_gain("tanh"))
        self.ent_clf[0].bias.data.fill_(0.)

    def forward(self, inputs: Tensor):

        # construct tensors
        batch_size, len_max = inputs.size()[:2]
        logits = []
        preds = []
        probs = []
        preds_emb = []
        init_pred = inputs.new_full(fill_value=self.pad,
                                    size=(batch_size, 1),
                                    dtype=torch.int)
        prev_pred = self.entl_embeddings(init_pred).squeeze(1)
        # predict labels token by token
        for i in range(len_max):
            x = torch.cat((inputs[:, i], prev_pred), dim=-1)
            logit = self.ent_clf(x)
            prob = self.softmax(logit)
            pred = torch.argmax(prob, dim=-1)
            prev_pred = self.dropout(self.entl_embeddings(pred).squeeze(1))

            logits.append(logit)
            preds.append(pred)
            probs.append(prob)
            preds_emb.append(prev_pred)

        logits = torch.stack(logits, dim=1)
        preds = torch.stack(preds, dim=1)
        probs = torch.stack(probs, dim=1)
        preds_emb = torch.stack(preds_emb, dim=1)

        return logits, preds, probs, preds_emb
