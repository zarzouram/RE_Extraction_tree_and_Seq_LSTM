from dgl import DGLGraph

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

    def forward(self, batch, es_tag):
        seq, ents, pos, dep, lens = batch[1:6]
        rellabels, rel_dir = batch[-2:]

        bsz = seq.size(0)

        seq, pos = self.embeddings(seq, pos)
        seq_encode_input = torch.cat((seq, pos), dim=-1)
        seq_encoded = self.seq_encoder(seq_encode_input, lens,
                                       self.padding_values[0])

        if self.detect_entities:
            # TODO: calculate epsilon using inverse sigmoid decay
            # ref. section 3.7. Make a class that have number of epochs
            # increased every training epoch
            epsilon = self.k
            edl_logits, edl_preds = self.entity_detection(seq, epsilon)

            if self.pretrain_ents_detection:
                return edl_logits, edl_preds, None, None

        dep = self.dep_embeddings(dep)

        if es_tag is None:
            # BIOUL decode, get entity tokens indices from entity predictions
            # construct pair candidates, get dependency tree
            # tree, idx0, tokens_idx, e1_idx, e2_idx
            raise NotImplementedError
        else:
            # get entity tokens indices
            e1_idx = torch.nonzero(ents == es_tag[0], as_tuple=True)
            e2_idx = torch.nonzero(ents == es_tag[1], as_tuple=True)
            # get tree
            if self.tree_type == "shortest_path":
                tree = batch[7]  # type: DGLGraph
                num_nodes = tree._batch_num_nodes["_N"].cpu()
                idx0 = torch.repeat_interleave(torch.arange(bsz), num_nodes)
                tokens_idx = tree.ndata["ID"]
            elif self.tree_type == "full_tree":
                tree = batch[5]  # type: DGLGraph
                idx0 = slice(None)
                tokens_idx = slice(None)
            elif self.tree_type == "sub_tree":
                # TODO: implement subtree
                raise NotImplementedError
            else:
                raise ValueError(
                    "tree_type value should be either 'shortest_path' or 'full_tree' or 'sub_tree'"  # noqa: E501
                )

        tree_inputs = [seq_encoded[idx0, tokens_idx], dep[idx0, tokens_idx]]
        if self.entity_detection:
            tree_inputs.extend(edl_preds[idx0, tokens_idx])
        tree_inputs = torch.cat(tree_inputs, dim=-1)
        tree.ndata["emb"] = tree_inputs

        tree_d12_logits, tree_d21_logits = self.dep_tree(tree)

        # select entity
        _, num_e1 = torch.unique(e1_idx[0], return_counts=True)
        _, num_e2 = torch.unique(e2_idx[0], return_counts=True)
        e1_hstate = torch.split(seq_encoded[e1_idx],
                                split_size_or_sections=num_e1)
        e2_hstate = torch.split(seq_encoded[e2_idx],
                                split_size_or_sections=num_e2)
        e1_hstate = torch.cat([h.mean(dim=0) for h in e1_hstate], dim=0)
        e2_hstate = torch.cat([h.mean(dim=0) for h in e2_hstate], dim=0)

        rel_clf_input_d12 = torch.cat((tree_d12_logits, e1_hstate, e2_hstate),
                                      dim=-1)
        rel_clf_input_d21 = torch.cat((tree_d21_logits, e1_hstate, e2_hstate),
                                      dim=-1)
