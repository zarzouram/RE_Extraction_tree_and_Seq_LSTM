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
        self.poss = data["pos"]
        self.deps = data["dep"]
        self.lengths = data["length"]
        self.e1last = data["e1_last"]
        self.e2last = data["e2_last"]
        self.deptree = data["dep_tree"]
        self.shortest_path = data["shortest_path"]
        self.rellabels = data["rels"]
        self.reldir = data["rels_dir"]

    def __getitem__(self, i: int):
        ids: Tensor = self.ids[i]
        tokens: Tensor = self.tokens[i]
        poss: Tensor = self.poss[i]
        deps: Tensor = self.deps[i]
        lengths: Tensor = self.lengths[i]
        e1last: Tensor = self.e1last[i]
        e2last: Tensor = self.e2last[i]
        deptree: DGLHeteroGraph = self.deptree[i]
        shortest_path: DGLHeteroGraph = self.shortest_path[i]
        rellabels: Tensor = self.rellabels[i]
        reldir: Tensor = self.reldir[i]

        return [
            ids, tokens, poss, deps, lengths, e1last, e2last, deptree,
            shortest_path, rellabels, reldir
        ]

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    def collate_fn(batch):
        # padd sequences and batch trees

        data = list(zip(*batch))

        ids = torch.stack(data[0])
        lengths = torch.stack(data[4])
        e1last = torch.stack(data[6])
        e2last = torch.stack(data[6])
        rellabels = torch.stack(data[-2])
        reldir = torch.stack(data[-1])

        deptree = dgl.batch(data[7])
        shortest_path = dgl.batch(data[8])

        tokens = pad_sequence(data[1], batch_first=True)
        poss = pad_sequence(data[2], batch_first=True)
        deps = pad_sequence(data[3], batch_first=True)

        return (ids, tokens, poss, deps, lengths, e1last, e2last, deptree,
                shortest_path, rellabels, reldir)

    apath = Path("dataset/semeval_task8/processed/")
    for split in ["train", "val", "test"]:
        p = str(apath / f"{split}.pt")
        ds = SemEvalTask8(p)

        loader_params = {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
        }
        data_loader = DataLoader(ds, **loader_params, collate_fn=collate_fn)

        for data in tqdm(data_loader):
            pass

    print("done")
