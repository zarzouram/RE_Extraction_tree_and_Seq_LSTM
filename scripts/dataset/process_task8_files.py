from typing import Dict, List, Tuple, Union
from pathlib import Path
from tqdm import tqdm
import re

from collections import Counter

import networkx as nx
import dgl
from dgl.traversal import topological_nodes_generator as traverse_topo
from dgl import DGLGraph

import torch

import stanza


def check_shortest_tree(u, v, path_nodes, path_tree, dep_tree):
    assert nx.is_tree(dgl.to_networkx(path_tree))

    # get shortest path nodes using networks
    path_nx = nx.DiGraph(list(zip(u, v)))
    path_g = dgl.node_subgraph(dep_tree, [n for n in path_nx.nodes()])
    short_nodes = path_g.ndata["_ID"]

    short_nodes, _ = torch.sort(short_nodes)
    path_nodes, _ = torch.sort(path_nodes)

    assert torch.all(path_nodes == short_nodes).item()


def check_ent_tag(ent_tags: List[str], tokens: List[str], e1: str, e2: str,
                  tags: Dict[str, str]):
    """confirm that entity tagging is ok.
    """
    # each sample shoul have e1 and e2
    cnt = Counter(ent_tags)
    assert tags["e1"] in cnt and tags[
        "e2"] in cnt, "one or more entity not found"

    # get token ids for e1 and e2 (could be multiple tokens)
    idx_1 = [i for i, tag in enumerate(ent_tags) if tag == "1"]
    idx_2 = [i for i, tag in enumerate(ent_tags) if tag == "2"]

    # assert that tagged tokens are actually what extracted by regex
    for i in idx_1:
        assert tokens[i].split("_")[0] in e1.lower()
    for i in idx_2:
        assert tokens[i].split("_")[0] in e2.lower()


def check_shortest_path(u, v):
    assert u != [] and v != [], "no path found"
    Gpath = nx.DiGraph(list(zip(u, v)))
    assert nx.is_tree(Gpath), "path is not a tree"


def tag_token(ents_start: List[int], ents_end: List[int], token_start: int,
              tags: Dict[str, str]) -> str:
    """ return tag if a token belongs to either e1 or e2.
    takes token start and end position, comparing it with entity start and
    end position to determine if it is e1, e2, or not either of them.
    """

    e1 = (token_start >= ents_start[0]) and (token_start < ents_end[0])
    e2 = (token_start >= ents_start[1]) and (token_start < ents_end[1])
    assert not e1 or not e2

    return tags["e1"] * e1 + tags["e2"] * e2 if e1 or e2 else tags["other"]


def get_path(tree, left, right):
    """Get a path from a left token to a right token in a dependency tree.
    """
    # The shortest path from left to right tokens in a depedency tree is only
    # one path or none.
    # The two nodes share a common head; each one may belong to a same branch
    # or different branch. One example could be:
    #     1) left --> node_al --> node_t <-- right
    # the above example shows that the left and right node are under the
    # node_t. Other example is:
    #     2) left --> node_a --> right --> node_t
    # Here, the path should be [left, node_a, right]. Note, that there is no
    # path from right to left nodes.

    # The idea is to traverse through edges from left and right nodes (go to
    # the heas node), at some point the two pathes may intersect at some node
    # (node_t), in the above example. The path would be either the union
    # between two the two pathes till the intersection node (example-1), or the
    # left path till the right node (example-2).

    # The dependency tree is represented in a list of tuples of edges.
    # Say we have an arbitrary tree:
    #       3
    #      / \
    #     8   5
    #    / \
    #   11 20
    # The above tree is represented as:
    #   [(11, 8), (20, 8), (8, 3), (5, 3), (3, 3)]
    # The list of edges above could be unpacked into two lists u, and v.
    # u is the a list of the first element in the edge tuples (list of nodes)
    # and v is the list of the second element (list of heads)

    n_right, n_left = right, left  # right and left nodes
    # right, left pathes (u,v)
    u_right, u_left, v_right, v_left = [], [], [], []
    for _ in range(len(tree)):

        # traverse through edges (left path and right path):
        # add the node and its head to the respective u, v array. The next node
        # will be the head. Stop at the root node (self looping)
        if n_right is not None and tree[n_right] != n_right:
            u_right.append(n_right)
            v_right.append(tree[n_right])
            n_right = v_right[-1]
        else:
            n_right = None

        if n_left is not None and tree[n_left] != n_left:
            u_left.append(n_left)
            v_left.append(tree[n_left])
            n_left = v_left[-1]
        else:
            n_left = None

        # Both right and left nodes are in the sama path.
        if right == n_left:
            U = u_left
            V = v_left
            return U, V

        # Both paths are intersected. Get the union of the intesected pathes.
        elif n_right in v_left:
            if left not in u_right:
                intersect = v_left.index(v_right[-1])
                U = u_left[:intersect + 1] + u_right
                V = v_left[:intersect + 1] + v_right
            else:
                # if the we have path:
                #   a --> ... --> b --> c (intersection)
                # right path (a, b), n_right=c
                # the left path is (b, c)
                # Here there is an intersection at the left path (b c)
                # at c. This is a special case that we do want not to allow. No
                # path from b to a. The right node (a) is found in the left
                # path.
                U, V = [], []

            return U, V

        elif n_left in v_right:
            intersect = v_right.index(v_left[-1])
            U = u_left + u_right[:intersect + 1]
            V = v_left + v_right[:intersect + 1]
            return U, V

    else:
        return [], []


def process_file(
    file_path: Path, split: str, stanza_models_dir: str,
    tags: Dict[str, Dict[str, Union[str, int]]], check: bool
) -> Tuple[List[List[str]], List[DGLGraph], List[int], List[Tuple[int, int]]]:
    """"
    read semeval task-8 text files and return the following: 1) list of tokens,
    2) list of pos, 3) dependency relation per token.
    """

    # open file
    with open(file_path, "r") as file:
        texts = file.readlines()

    # define stanza pipline, tokenize, POS tagging, and
    # dependency parsing
    nlp = stanza.Pipeline(lang="en",
                          processors="tokenize,pos,lemma,depparse",
                          tokenize_no_ssplit=True,
                          model_dir=stanza_models_dir)

    # set some regular expressions
    # detect tokens betweem entity tags <e1><\ or <e2>
    entity_regex = re.compile(r"(?<=<e[12]>)\w+(?:\W? ?\w+)*(?=</e[12]>)")
    # detect entity tags themselves <e1>, <e2>
    tag_regex = re.compile(r"</?e[12]>")
    # detect the douple quotes at the beginning and the end of the text
    dquotes_regex = re.compile(r"^\"|\"$")

    # variables required by paper
    token_lst: List[List[str]] = []  # list of "tokens list"
    pos_lst: List[List[str]] = []  # list of "token's pos list"
    dep_tag_lst: List[List[str]] = []  # list of "token's dep tag"
    rel_lst: List[List[str]] = []  # list of "entity relation list"
    rel_dir_lst: List[List[str]] = []  # list of "relation direction list"
    dep_tree_lst: List[DGLGraph] = []  # list of dependency trees in dgl graph
    shortest_path: List[DGLGraph] = []  # path from e1 to e2 in dgl graph

    # variables needed during training
    length: List[int] = []  # sentence length
    ent_lst: List[List[str]] = []  # list of tokens' tag "tag(e1, o, e2)"
    ids = []

    # Other variables
    ent_len_betn: List[int] = []  # length between ents' last token

    pbar = tqdm(range(0, len(texts), 4))
    for ln in pbar:
        idx, text = texts[ln].split("\t")  # read sample text and sample id
        pbar.set_description(f"Processing {split} file - index: {idx}")

        text = dquotes_regex.sub(string=text.strip(), repl="")  # remove ""
        matches = list(entity_regex.finditer(text))  # find tokens betn tags
        text_notags = tag_regex.sub(string=text,
                                    repl="")  # remove tags from text

        # get the start and end positions, corrected for tags removal
        e1s, e1e = matches[0].start() - 4, matches[0].end() - 4
        e2s, e2e = matches[1].start() - 13, matches[1].end() - 13

        # tokenization, and pos, dependency tagging
        doc = nlp(text_notags)
        token: List[str] = []
        ent: List[str] = []
        e1_ids: List[str] = []
        e2_ids: List[str] = []
        pos: List[str] = []
        dep_tree: List[int] = []  # dependency tree
        dep_tag: List[str] = []  # dependency tags
        # short_path: List[int] = []  # shortest path between e1 and e2
        for sent in doc.sentences:
            for word in sent.words:
                token.append(f"{word.text.lower()}_{word.upos}")
                pos.append(word.upos)  # pos for each token

                # extract token tagged by <e1> and <e2>
                e_tag = tag_token([e1s, e2s], [e1e, e2e], word.start_char,
                                  tags["ent"])
                ent.append(e_tag)
                if e_tag == tags["ent"]["e1"]:
                    e1_ids.append(word.id - 1)
                elif e_tag == tags["ent"]["e2"]:
                    e2_ids.append(word.id - 1)

                # dependency tree in a list:
                # each token id (list's id) refers to its head,
                # root refers to itself
                dep_tree.append(word.head - 1 if word.head > 0 else word.id -
                                1)
                dep_tag.append(word.deprel)  # dependency tag

        # read relation label and direction 0 is label(e1,e2), 1 is
        # label(e2,e1)
        if texts[ln + 1].strip() == "Other":
            rel_label = "Other"
            rel_dir = tags["rel"]["other"]
        else:
            rel_label, dir_str = texts[ln + 1].split("(")
            rel_dir = tags["rel"]["left"] if dir_str[1] == "1" else tags[
                "rel"]["right"]

        # dependency tree, remove self looping at root node
        u_dep, v_dep = zip(*[(ud, vd) for ud, vd in enumerate(dep_tree)
                             if ud != vd])
        u_dep, v_dep = torch.LongTensor(u_dep), torch.LongTensor(v_dep)
        dep_g = dgl.graph((u_dep, v_dep))  # type: DGLGraph

        # get shortest path from e1 to e2
        u, v = get_path(tree=dep_tree, left=e1_ids[-1], right=e2_ids[-1])
        if not u:
            u, v = get_path(tree=dep_tree, left=e2_ids[-1], right=e1_ids[-1])

        # convert trees into dgl grahs
        # shortest path tree
        # from u,v -> create shortest path graph
        #   There is an issue using u,v directly creating graph using
        #   dgl. The dgl expected that u,v are tensors that represent
        #   consecutive integer nodes' Ids. Thus, when creating graph using dgl
        #   with, say, u,v = (15, 12), the dgl will creat a graph with 16 nodes
        #   and one edge from 15 to 12.
        dep_nodes = torch.arange(dep_g.num_nodes())
        path_nodes = torch.LongTensor(u + [v[-1]])  # v[-1]: root
        nmap = {n: i
                for i, n in enumerate(path_nodes.tolist())}  # nodes id map
        u_ = torch.LongTensor([nmap[n] for n in u])
        v_ = torch.LongTensor([nmap[n] for n in v])
        path_g = dgl.graph((u_, v_))  # type: DGLGraph
        path_g.ndata["ID"] = path_nodes

        # all other nodes not in the shortest path
        opath_idx = torch.all(dep_nodes[:, None] != path_nodes, dim=1)
        opath_nodes = dep_nodes[opath_idx]

        # assigne some information to our nodes
        short_ndata = torch.zeros(path_g.num_nodes(), dtype=torch.long)
        dep_ndata = torch.zeros(dep_g.num_nodes(), dtype=torch.long)

        # 1. nodes belong to short path = 0 elsr 1
        path_g.nodes[:].data["n_type"] = short_ndata
        dep_g.nodes[opath_nodes].data["n_type"] = dep_ndata[opath_nodes] + 1

        # 2. trees root
        path_g.nodes[nmap[v[-1]]].data["root"] = torch.LongTensor([1])
        root = list(traverse_topo(dep_g))[-1]
        dep_g.nodes[root].data["root"] = torch.LongTensor([1])

        # 3. assign ents last tokens in trees
        path_g.nodes[nmap[e1_ids[-1]]].data["e1"] = torch.LongTensor([1])
        path_g.nodes[nmap[e2_ids[-1]]].data["e2"] = torch.LongTensor([1])

        # stack data in list
        dep_tree_lst.append(dep_g)
        shortest_path.append(path_g)
        ids.append(idx)
        token_lst.append(token)
        length.append(len(token))
        ent_len_betn.append(e2_ids[-1] - e1_ids[-1] - 1)
        ent_lst.append(ent)
        pos_lst.append(pos)
        dep_tag_lst.append(dep_tag)
        rel_lst.append(rel_label)
        rel_dir_lst.append(rel_dir)

        if check:
            e1, e2 = matches[0].group(), matches[1].group()
            check_ent_tag(ent, token, e1, e2, tags["ent"])
            check_shortest_path(u, v)
            check_shortest_tree(u, v, path_nodes, path_g, dep_g)

    return (ids, token_lst, ent_lst, pos_lst, dep_tag_lst, dep_tree_lst,
            shortest_path, length, ent_len_betn, rel_lst, rel_dir_lst)
