from typing import Dict, List, Optional
from torch import Tensor

from dgl import DGLGraph

import torch
import torch.nn as nn

from .embeddinglayer import EmbeddingLayer
from .sequence_layer import SeqEncoder
from .entity_detection_layer import EntityDetection
from .dependency_layer import DepLayer


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
                 rel_num: int,
                 dropouts: Dict[str, float],
                 schedule_k: float = 1.,
                 bidir_seq: bool = True,
                 bidir_tree: bool = True,
                 end2end: bool = False,
                 pretrain: bool = False,
                 tree_type: str = "shortest_path"):

        super(LSTMER, self).__init__()

        # reference: arXiv:1601.00770v

        self.end2end = end2end
        self.tree_type = tree_type
        self.padding_values = [token_pad_value, pos_pad_value, entl_pad_value]
        self.k = schedule_k

        self.pretrain = pretrain

        self.hs_size = 2 * hs_size if bidir_seq else hs_size
        self.htree_size = 3 * htree_size if bidir_tree else 2 * htree_size
        self.entl_emb_sz = entl_emb_sz if end2end else 0

        # Word and POS embediings: Section 3.1
        self.embeddings = EmbeddingLayer(token_vocab_sz,
                                         pos_vocab_sz,
                                         token_emb_sz,
                                         pos_emb_sz,
                                         token_pad_value,
                                         pos_pad_value,
                                         dropout=dropouts["token_embd"])

        # Sequence layer: Section 3.2
        self.seq_encoder = SeqEncoder(input_size=token_emb_sz + pos_emb_sz,
                                      h_size=hs_size,
                                      num_layers=num_layers_seq,
                                      bidirectional=bidir_seq,
                                      lstm_dropout=dropouts["lstm"],
                                      output_dropout=dropouts["lstm_out"])

        # Entity detection layer: Section 3.3
        if end2end:
            self.entity_detection = EntityDetection(
                input_size=self.hs_size,
                entl_emb_sz=entl_emb_sz,
                input_sz=hs_size,
                output_size=entl_vocab_sz,
                hidden_sz=ht_size,
                dropout=dropouts["ent_h"],
                label_dropout=dropouts["ent_label"],
                entl_pad_value=entl_pad_value)

        if not pretrain:  # Pretrain mode, section 3.7
            # Dependency layer Sections 3.4 & 3.5 and relation extraction
            # section 3.6
            self.tree_input_sz = self.hs_size + dep_emb_sz + self.entl_emb_sz
            rel_clf_input_sz = self.htree_size + 2 * self.hs_size
            self.dep_embeddings = nn.Embedding(dep_vocab_sz,
                                               dep_emb_sz,
                                               padding_idx=dep_pad_value)
            self.dep_tree = DepLayer(input_size=self.tree_input_sz,
                                     h_size=htree_size,
                                     bidirectional=bidir_tree)
            self.rel_clf = nn.Sequential(nn.Linear(rel_clf_input_sz, hp_size),
                                         nn.Tanh(),
                                         nn.Dropout(dropouts["rel_h"]),
                                         nn.Linear(hp_size, rel_num))

    def forward(self, batch: List[Tensor], es_tag: Optional[List[int]]):

        if not self.end2end and (es_tag is None or not es_tag):
            raise NameError(
                "Missing 'es_tag' input. Expected a list of entity labels when not working in End to End mode."  # noqa: E501
            )

        seq, ents, pos, dep, lens = batch[1:6]
        bsz = seq.size(0)  # batch size

        # section-3.2: Sequence layer
        seq_embed = self.embeddings(seq, pos)
        seq_encoded = self.seq_encoder(seq_embed, lens, self.padding_values[0])

        # section-3.3: End to End mode (Entity detection layer)
        if self.end2end:
            # TODO: calculate epsilon using inverse sigmoid decay
            # ref. section 3.7. Make a class that have number of epochs
            # increased every training epoch
            epsilon = self.k
            edl_logits, edl_preds = self.entity_detection(seq, epsilon)

            # entity detection pretraining
            if self.pretrain:
                return edl_logits, edl_preds, None, None

        # section-3.4: dependency layer
        dep = self.dep_embeddings(dep)

        # Not pretraining and End to End
        if self.end2end:
            # TODO;
            # BIOUL decode, get entity tokens indices from entity predictions
            # construct pair candidates, get dependency tree
            # Need the following;
            # tree, idx0, tokens_idx, e1_idx_tuple, e2_idx_tuple
            raise NotImplementedError

        # Not pretraining and not End to End
        # entities are given: simeval task8
        else:
            # get entity tokens indices tuple(list(indx_0, index_1))
            e1_idx_tuple = torch.nonzero(ents == es_tag[0], as_tuple=True)
            e2_idx_tuple = torch.nonzero(ents == es_tag[1], as_tuple=True)

            # get tree
            if self.tree_type == "shortest_path":
                tree = batch[6]  # type: DGLGraph
                num_nodes = tree._batch_num_nodes["_N"].cpu()
                # indices for tokens in the shortest path
                idx0 = torch.repeat_interleave(torch.arange(bsz), num_nodes)
                tokens_idx = tree.ndata["ID"]
            elif self.tree_type == "full_tree":
                tree = batch[6]  # type: DGLGraph
                # indices for all nodes
                idx0 = slice(None)
                tokens_idx = slice(None)
            elif self.tree_type == "sub_tree":
                # TODO: implement subtree
                # indices for all nodes in the subtree
                # idx0 = slice()
                # tokens_idx = slice()
                raise NotImplementedError
            else:
                raise ValueError(
                    "tree_type value should be either 'shortest_path' or 'full_tree' or 'sub_tree'"  # noqa: E501
                )

        # Section 3.5
        # xₜ = [sₜ; vₜᵈ, vₜᵉ]
        xt = [seq_encoded[idx0, tokens_idx], dep[idx0, tokens_idx]]
        # end to end: add the embeddings of predicted entity labels [vₜᵉ]
        if self.end2end:
            xt.extend(edl_preds[idx0, tokens_idx])
        xt = torch.cat(xt, dim=-1)
        tree.ndata["emb"] = xt

        dp_12, dp_21 = self.dep_tree(tree)  # tree logits e1->e2, e2->e1

        # Section 3.6: Relation prediction layer
        # get hidden state for each entity's tokens
        # number of entity's tokens per sample
        _, num_e1 = torch.unique(e1_idx_tuple[0], return_counts=True)
        _, num_e2 = torch.unique(e2_idx_tuple[0], return_counts=True)

        # (Batch_size x num_ents_tokens, hidden_size) -->
        #     List[num_ents_tokens, hidden_size)], len(list) - batch_size
        e1_hstate = torch.split(
            seq_encoded[e1_idx_tuple],
            split_size_or_sections=num_e1.tolist())  # type: List[Tensor]
        e2_hstate = torch.split(
            seq_encoded[e2_idx_tuple],
            split_size_or_sections=num_e2.tolist())  # type: List[Tensor]

        # (Batch_size, hidden_size), means of entities hidden state across
        # thier tokens
        e1_hstate = torch.cat([h.mean(0).unsqueeze(0) for h in e1_hstate],
                              dim=0)
        e2_hstate = torch.cat([h.mean(0).unsqueeze(0) for h in e2_hstate],
                              dim=0)

        dp_12_prime = torch.cat((dp_12, e1_hstate, e2_hstate), dim=-1)
        dp_21_prime = torch.cat((dp_21, e1_hstate, e2_hstate), dim=-1)

        hpr_12 = self.rel_clf(dp_12_prime)  # e1->e2 relation preds logits
        hpr_21 = self.rel_clf(dp_21_prime)  # e2->e1 relation preds logits

        return hpr_12, hpr_21
