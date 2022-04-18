import torch
from torch import Tensor
import dgl
from dgl.heterograph import DGLHeteroGraph
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SemEvalTask8(Dataset):

    def __init__(self, datset_path: str):
        super().__init__()

        # loading data from dataset path
        data = torch.load(datset_path)
        self.ids = data["idx"]
        self.tokens = data["tokens"]
        self.ents = data["ents"]
        self.poss = data["pos"]
        self.deps = data["dep"]
        self.lengths = data["length"]
        self.deptree = data["dep_tree"]
        self.shortest_path = data["shortest_path"]
        self.rellabels = data["rels"]
        self.reldir = data["rels_dir"]

    def __getitem__(self, i: int):
        ids: Tensor = self.ids[i]
        tokens: Tensor = self.tokens[i]
        ents: Tensor = self.ents[i]
        poss: Tensor = self.poss[i]
        deps: Tensor = self.deps[i]
        lengths: Tensor = self.lengths[i]
        deptree: DGLHeteroGraph = self.deptree[i]
        shortest_path: DGLHeteroGraph = self.shortest_path[i]
        rellabels: Tensor = self.rellabels[i]
        reldir: Tensor = self.reldir[i]

        return [
            ids, tokens, ents, poss, deps, lengths, deptree, shortest_path,
            rellabels, reldir
        ]

    def __len__(self):
        return len(self.ids)


class collate_fn(object):

    def __init__(self, padding_values, tree_type="shortest_path"):
        self.tp = padding_values[0]
        self.pp = padding_values[1]
        self.dp = padding_values[2]
        self.ep = padding_values[3]
        self.tree_type = tree_type

    def __call__(self, batch):
        # padd sequences and batch trees

        data = list(zip(*batch))

        lengths = torch.stack(data[5]).type(torch.int64).requires_grad_(False)
        lengths, sorted_idxs = torch.sort(lengths, descending=True)

        ids = torch.stack(data[0])[sorted_idxs]
        deptree = dgl.batch([data[6][i] for i in sorted_idxs])
        shortest_path = dgl.batch([data[7][i] for i in sorted_idxs])

        rellabels = torch.stack(data[-2])[sorted_idxs]
        reldir = torch.stack(data[-1])[sorted_idxs]

        tokens = pad_sequence(data[1], padding_value=self.tp, batch_first=True)
        ents = pad_sequence(data[2], padding_value=self.ep, batch_first=True)
        poss = pad_sequence(data[3], padding_value=self.pp, batch_first=True)
        deps = pad_sequence(data[4], padding_value=self.dp, batch_first=True)

        tokens = tokens[sorted_idxs]
        ents = ents[sorted_idxs]
        poss = poss[sorted_idxs]
        deps = deps[sorted_idxs]

        if self.tree_type == "shortest_path":
            tree = shortest_path
        elif self.tree_type == "full_tree":
            tree = deptree
        elif self.tree_type == "sub_tree":
            raise NotImplementedError
        else:
            raise ValueError(
                "tree_type value should be either 'shortest_path' or 'full_tree' or 'sub_tree'"  # noqa: E501
            )

        return (ids, tokens, ents, poss, deps, lengths, tree, rellabels,
                reldir)


if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    apath = Path("dataset/semeval_task8/processed/")
    for split in ["train", "val", "test"]:
        p = str(apath / f"{split}.pt")
        ds = SemEvalTask8(p)

        loader_params = {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
        }
        data_loader = DataLoader(ds,
                                 **loader_params,
                                 collate_fn=collate_fn([0, 0, 0, 0]))

        for data in tqdm(data_loader):
            pass

    print("done")
