"""
A. Reference:
--------------
    This code is based on DGL's tree-LSTM implementation found in the paper
    [3]
    DGL Implementation can be found at
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm/tree_lstm.py

B. Papers list:
---------------

    1. End-to-End Relation Extraction using LSTMs on Sequences and Tree
    Structures https://arxiv.org/abs/1601.00770

    2. Improved Semantic
    Representations From Tree-Structured Long Short-Term Memory Networks.
    https://arxiv.org/abs/1503.00075

    4. A Shortest Path Dependency Kernel
    for Relation Extraction https://dl.acm.org/doi/10.3115/1220575.1220666

C. General Description:
-----------------------
    This code is based on [1] and is used to classify relations between two
    entities in a sentence. As indicated in [4], the shortest path between two
    entities in the same sentence contains all the information required to
    identify those relationships—the treeLSTM process a sentence over its
    dependency tree to extract the relation between a couple of entities.

    The treeLSTM used in LSTM-ER [1] is derived from the N-ary treeLSTM
    architecture [2]. The dependency tree in nature has a varying number of
    children. However, the N-ary design needs a fixed number of child nodes.
    For example, in [2], the N-ary tree is used with a constituency binary tree
    where each node has at least two nodes. To overcome this problem, In [2],
    the dependency tree nodes are categorized into two classes: the
    shortest-path nodes are one class, and the other nodes are the second
    class. Nodes that belong to the same class share the same weights.

D. Abbreviation:
---------------------
    B      : Batch size
    H      : LSTM's Hidden size
    E      : Embedding size, feature size
    SEQ    : Max Sequence length in the batch
    NE-OUT : Output dimension for NER module '
    Nt     : total number of nodes (Nt) in the batch '
    Nchn   : Number of children nodes in the batch
    DEP    : Dependency embedding size

E. TreeLSTMCell Impelementation:
--------------------------------
    For each node, `message_func` sends the children's information to
    `reduce_func` function. The following info is sent:

        h:      child nodes' hiddens state      Size: (Nt, Nchn, H)
        c:      child nodes' cell state         Size: (Nt, Nchn, H)
        type_n: child nodes' type               Size: (Nt, Nchn)

    The data is retained in the `nodes.mailbox`. We should receive h and c in a
    tensor of size (Nt, Nchn, H). Because the number of children for each node
    may vary, a tensor of size (Nt, Nchn, H) cannot be formed. Thus, the
    `reduce_function` collects/groups the information according to the `Nchn`
    dim, i.e., nodes with the same number of children are in one group. The
    `reduce_function` calls itself iteratively to process each group
    separately.

    In the `reduce_function`, each node's hidden state and cell state are
    calculated based on the children's information. The calculated states have
    a dim of (number of nodes in the group, H). The function then stacks all
    results vertically and sends them to the `apply_node_func`. The final size
    should be (Nt, H). In the `apply_node_func`, the values of the gates for
    each node are calculated.

    For the leaf nodes, where there are no children, the code starts at
    `apply_node_func`; The hidden state is initialized, then the gate values
    are calculated.

    E1. The forget gate eqn:
    -----------------------
        Assuming the following:
        1. The number of nodes in a graph (G) = n.

        2. For a node-t ∈ G & 1<=t<=n:

            a. number of children of node-t: Nchn(t) = ℓ,

            b. For an arbitry node (node-r), r ≠ t and r ∈ G: Nchn(r) may
               not be equal to ℓ

            c. The hidden state for the children nodes is denoted as htl.
               htl = [hi] & 1 <= i <= ℓ.
               htl ∈ ℝ^(ℓ × H), hi ∈ ℝ^H

            d. Each child node is either of type_n0 or type_n1.
               The hidden state for typn_0 nodes is denoted as h_t0 and
               for type_n1 nodes is denoted as h_t1,
               where:
               h_t0 = Sum( [hi | 1 <= i <= ℓ & m(i)=type_n0] )
               h_t1 = Sum( [hi | 1 <= i <= ℓ & m(i)=type_n1] )
               m(.) is a mapping function that maps the node type

            e. Node-t have ℓ forget gates; a gate for each child node

        In [1] eqn 4, the second part of the forget gate (Sum(U*h)) could
        be written as follows:
            - For each node-k in the child nodes of node_t:
              The forget gate (ftk, 1 <= k <= ℓ) is either a type_0 (f0) or (f1)
              type_1.
              where:
              f0 = U00 h_t0 + U01 h_t1,  eq(a)
              f1 = U10 h_t0 + U11 h_t1   eq(b)

    E2. i,o,u eqn:
    --------------
        For node_t:
        i_t = U_i0 . h_t0 + U_i1 . h_t1   eq(c)
        o_t = U_o0 . h_t0 + U_o1 . h_t1   eq(d)
        u_t = U_u0 . h_t0 + U_u1 . h_t1   eq(e)

    E3. Proof of eq(a, b):
    ----------------------
        - Assuming a node-t = node-1 in a graph:
        - node-1 have 4 child nodes: Nct=[n1, n2, n3, n4].
        - The types of child nodes are as follows [0, 1, 1, 0]
        - Ignoring the fixed parts in the forget gates' equation: Wx & b:
        - From eq 4 in [1]
          for a node-k that is a child of node-1
              f1k = Sum(U_m(k)m(1l) * h1l) &
              1 <= l < 4 & m(1l) & m(k) is either 0 or 1

                        node-t
                      /  |  \  \
                    ht1 ht2 ht3 ht4
                    /    |    \  \
              k=   1     2     3  4
           m(k)=   0     1     1  0

        - For each child node, the equations are:
            child-node-1: f11 = U00 h11 + U01 h12 + U01 h13 + U00 h14
            child-node-2: f12 = U10 h11 + U11 h12 + U11 h13 + U10 h14
            child-node-3: f13 = U10 h11 + U11 h12 + U11 h13 + U10 h14
            child-node-4: f14 = U00 h11 + U01 h12 + U01 h13 + U00 h14

        - The equation of child-node 1,4 (type_n0) are equal to each
            other, the same are for child nodes 2,3, (type_n1).

        - Further reduction can be done as follows:
            forget type_0:
                f0 = U00 (h11 + h14) + U01 (h12 + h13)
            forget type_1:
                f1 = U10 (h11 + h14) + U11 (h12 + h13)
            h_t0 = (h11 + h14)
            h_t1 = (h12 + h13),
            see section E1 above.

            Thus:
            f0 = U00 h_t0 + U01 h_t1
            f1 = U10 h_t0 + U11 h_t1
            where ht_0 is hidden states for type_n0 child nodes and ht_1 is
            hidden states for type_n1 child nodes.

    E4. Impelemntation:
    --------------------
        Step:1 Get ht_0 anf ht_1:
        *************************
            1. Get hidden states for each node type: ht_0, ht_1
                a. Get nodes that are belong to each node type
                    (type: 0 & 1)
                b. Get h and c for each node type "ht_0, ht_1"
                c. If there is no specific node type,
                    the respective ht_0 or ht_1 is zeros

        Step:2 i,o,t gates: based on eqs(c,d,e) Under section E2:
        *********************************************************
            a. [ht_0, ht_1] [   Uiot   ] = [i, o, t]
                (Nt , 2H)    (2H , 3H)   = (Nt , 3H)

            b. `reduce_func` return [i, o, t]

        Step:3 Forget gate: based on eqs(a,b) Under section E1:
        *****************************************************************
            a. [ht_0, ht_1] [    Uf    ] =  [f0, f1]    from eqs(a,b)
                (Nt , 2H)     (2H , 2H)  =  (Nt , 2H)

            b. calculate ftk, here it denoted by Ftm(k), i.e. Ft0, Ft1. A forget
            gate per node type. Note that F10=f11=f14 & F11=f12=f13

            c. update cell status = ∑ ftl * ctl
               ∑ ftl * ctl = f11*c11 + f12*c12 + f13*c13 + f14*c14 + f15*c15
                           = F10*c11  + F11*c12  + F11*c13  + F10*c14
                           = F10(c11 + c14) + F11(c12 + c13)
"""

# pytorch
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

# DGL
import dgl
from dgl import DGLGraph


class TreeLSTMCell(nn.Module):

    def __init__(self, input_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(input_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))

        self.W_f = nn.Linear(input_size, h_size, bias=False)
        self.U_f = nn.Linear(2 * h_size, 2 * h_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros(1, h_size))

        self.norm = nn.ModuleList([nn.LayerNorm(h_size) for _ in range(3)])
        self.norm_f = nn.LayerNorm(h_size)
        self.norm_c = nn.LayerNorm(h_size)

    def message_func(self, edges):
        return {
            "h": edges.src["h"],
            "c": edges.src["c"],
            "n_type": edges.src["n_type"],
        }

    def reduce_func(self, nodes):

        c_child = nodes.mailbox["c"]  # (Nt, Nchn, H)
        h_child = nodes.mailbox["h"]  # (Nt, Nchn, H)
        hidden_size = c_child.size(2)

        # Step 1
        type_n = nodes.mailbox["n_type"]  # (Nt)
        type_n0_id = type_n == 0
        type_n1_id = type_n == 1

        # creat mask matrix with the same size of h and c with zeros at
        # either type_0 node ids or type_1 node ids
        mask = torch.zeros_like(h_child, requires_grad=False)
        mask[type_n0_id] = 1  # mask one at type_0 nodes
        ht_0 = mask * h_child  # (Nt, Nchn, H)
        ct_0 = mask * c_child  # (Nt, Nchn, H)
        ht_0 = torch.sum(ht_0, dim=1)  # sum over child nodes => (Nt, H)
        ct_0 = torch.sum(ct_0, dim=1)  # (Nt, H)

        mask = torch.zeros_like(h_child, requires_grad=False)
        mask[type_n1_id] = 1
        ht_1 = mask * h_child  # (Nt, Nchn, H)
        ct_1 = mask * c_child  # (Nt, Nchn, H)
        ht_1 = torch.sum(ht_1, dim=1)  # sum over child nodes => (Nt, H)
        ct_1 = torch.sum(ct_1, dim=1)  # (Nt, H)

        # # Step 2
        h_iou = torch.cat((ht_0, ht_1), dim=1)  # (Nt, 2H)

        # Step 3.a
        # (Nt, 2H) => (Nt, 2, H)
        f = self.U_f(torch.cat((ht_0, ht_1), dim=1)).view(-1, 2, hidden_size)

        # Steps 3.b.c
        X = self.W_f(nodes.data["emb"])  # (Nt, H)
        F_tmk = X[:, None, :] + f + self.b_f  # (Nt, 2, H)
        F_tmk = torch.sigmoid(self.norm_f(F_tmk))
        c_tmk = torch.stack([ct_0, ct_1], dim=1)  # (Nt, 2, H)
        c_cell = torch.sum(F_tmk * c_tmk, dim=1)  # (Nt, H)

        return {"h": h_iou, "c": c_cell}

    def apply_node_func(self, nodes):

        h_cell = nodes.data["h"]
        c_cell = nodes.data["c"]

        # Initialization for leaf nodes
        # The leaf nodes have no child the h_child is initialized.
        if nodes._graph.srcnodes().nelement() == 0:  # leaf nodes
            # initialize h states, for node type-0 and node type-1
            # NOTE: initialization for node type-0 == node type-1
            h_cell = torch.cat((h_cell, h_cell), dim=1)  # (Nt, Nchn*H)

        # (Nt x 3*H)
        iou = self.W_iou(nodes.data["emb"]) + self.U_iou(h_cell) + self.b_iou
        iou = torch.chunk(iou, chunks=3, dim=1)  # (Nt x H) for each of i,o,u
        i, o, u  = [self.norm[i](iou[i]) for i in range(3)]
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        c = i * u + c_cell
        h = o * torch.tanh(self.norm_c(c))

        return {"h": h, "c": c}


class TreeLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        h_size,
        bidirectional,
    ):

        super(TreeLSTM, self).__init__()

        self.bidirectional = bidirectional
        self.h_size = h_size
        self.TeeLSTM_cell = TreeLSTMCell(input_size, h_size)

        # learnable initial states
        self.init_states()

    def init_states(self):
        h0 = torch.zeros(1, self.h_size)
        c0 = torch.zeros(1, self.h_size)
        xavier_uniform_(h0, gain=nn.init.calculate_gain("sigmoid"))
        xavier_uniform_(c0, gain=nn.init.calculate_gain("sigmoid"))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)

    def forward(self, g: DGLGraph):
        """A modified N-ary tree-lstm (LSTM-ER) network
        ----------
        g:      dgl.DGLGraph
                Batch of Trees for computation
        Returns
        -------
        logits: Tensor
        """
        # initialize hiddedn and cell state
        h0 = self.h0.repeat(g.num_nodes(), 1)
        c0 = self.c0.repeat(g.num_nodes(), 1)
        g.ndata["h"] = h0
        g.ndata["c"] = c0

        # copy graph
        if self.bidirectional:
            g_copy = g.clone()

        # propagate bottom top direction
        dgl.prop_nodes_topo(
            g,
            message_func=self.TeeLSTM_cell.message_func,
            reduce_func=self.TeeLSTM_cell.reduce_func,
            apply_node_func=self.TeeLSTM_cell.apply_node_func,
        )
        logits = g.ndata.pop("h")

        if self.bidirectional:
            # propagate top bottom direction
            dgl.prop_nodes_topo(
                g_copy,
                message_func=self.TeeLSTM_cell.message_func,
                reduce_func=self.TeeLSTM_cell.reduce_func,
                apply_node_func=self.TeeLSTM_cell.apply_node_func,
                reverse=True,
            )
            logits_tb = g_copy.ndata.pop("h")

            # concatenate both tree directions
            logits = torch.cat((logits, logits_tb), dim=-1)

        return logits
