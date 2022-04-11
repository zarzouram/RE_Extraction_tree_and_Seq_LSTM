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
                 bidir_seq: bool = True,
                 bidir_tree: bool = True,
                 entity_segmenter: bool = False,
                 entity_pretrain: bool = False):

        super(LSTMER, self).__init__()

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

    def forward(self):
        pass
