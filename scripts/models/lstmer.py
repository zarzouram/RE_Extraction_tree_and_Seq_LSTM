import torch
import torch.nn as nn

from embeddinglayer import EmbeddingLayer
from sequence_layer import SeqEncoder
from entity_detection_layer import EntityDetection
from dependency_layer import DepLayer


class LSTMER(nn.Module):

    def __init__(self,
                 token_vocab_sz: int,
                 pos_vocab_sz: int,
                 dep_vocab_sz: int,
                 entl_vocab_sz: int,
                 token_emb_sz: int,
                 pos_emb_sz: int,
                 dep_emb_sz: int,
                 entl_emb_sz: int,
                 token_pad_value: float,
                 pos_pad_value: float,
                 dep_pad_value: float,
                 entl_pad_value: float,
                 hs_size: int,
                 ht_size: int,
                 htree_size: int,
                 hp_size: int,
                 num_layers_seq: int,
                 ents_num: int,
                 rel_num: int,
                 dropout: float = 0.1,
                 schedule_k: float = 1.,
                 bidir_seq: bool = True,
                 bidir_tree: bool = True,
                 entity_segmenter: bool = False,
                 entity_pretrain: bool = False,
                 tree_type: str = "shortest_path"):

        super(LSTMER, self).__init__()

        self.tree_type = tree_type
        self.padding_values = [token_pad_value, pos_pad_value, entl_pad_value]
        self.k = schedule_k

        self.detect_entities = entity_segmenter
        self.pretrain_ents_detection = entity_pretrain

        self.hs_size = 2 * hs_size if bidir_seq else hs_size
        self.htree_size = 3 * htree_size if bidir_tree else 2 * htree_size
        self.entl_emb_sz = entl_emb_sz if entity_segmenter else 0

        self.embeddings = EmbeddingLayer(token_vocab_sz, pos_vocab_sz,
                                         token_emb_sz, pos_emb_sz,
                                         token_pad_value, pos_pad_value)

        self.seq_encoder = SeqEncoder(input_size=token_emb_sz,
                                      h_size=hs_size,
                                      num_layers=num_layers_seq,
                                      bidirectional=bidir_seq,
                                      dropout=dropout)
        if entity_segmenter:
            self.entity_detection = EntityDetection(
                entl_vocab_sz=entl_vocab_sz,
                entl_emb_sz=entl_emb_sz,
                input_sz=hs_size,
                output_sz=ents_num,
                hidden_sz=ht_size,
                dropout=dropout,
                entl_pad_value=entl_pad_value)

        if not entity_pretrain:
            self.tree_input_sz = self.hs_size + dep_emb_sz + self.entl_emb_sz
            rel_clf_input_sz = self.htree_size + 2 * self.hs_size
            self.dep_embeddings = nn.Embedding(dep_vocab_sz,
                                               dep_emb_sz,
                                               padding_idx=dep_pad_value)
            self.dep_tree = DepLayer(input_size=self.tree_input_sz,
                                     h_size=htree_size,
                                     bidirectional=bidir_tree)
            self.rel_clf = nn.Sequential(nn.Linear(rel_clf_input_sz, hp_size),
                                         nn.Tanh(), nn.Dropout(dropout),
                                         nn.Linear(hp_size, rel_num))

    def forward(self, batch):
        seq, es_idx, pos, dep, lens = batch[1:6]
        rellabels, rel_dir = batch[-2:]

        seq, pos = self.embeddings(seq, pos)
        seq_encode_input = torch.cat((seq, pos), dim=-1)
        seq_encoded = self.seq_encoder(seq_encode_input,
                                       self.padding_values[0])

        if self.detect_entities:
            # TODO: calculate epsilon using inverse sigmoid decay
            # ref. section 3.7. Make a class that have number of epochs
            # increased every training epoch
            epsilon = self.k
            edl_logits, edl_preds = self.entity_detection(seq, epsilon)

            if self.pretrain_ents_detection:
                return edl_logits

        dep = self.dep_embeddings(dep)
        if self.tree_type == "shortest_path":
            pass
        else:
            pass

        tree_input = torch.cat((seq_encoded, dep), dim=-1)
        if self.entity_detection:
            tree_input = torch.cat((tree_input, edl_preds), dim=-1)

