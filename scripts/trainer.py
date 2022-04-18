from typing import List, Dict, Optional, Union
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn

from torchtext.vocab.vocab import Vocab

from torch.utils.tensorboard import SummaryWriter

from scripts.utils.utils import seed_everything


class Trainer():

    def __init__(self,
                 optim: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.StepLR,
                 device: torch.device,
                 es_tag: List[int],
                 modes: Dict[str, bool],
                 epochs: int,
                 early_stop: int,
                 lr_patience: int,
                 grad_clip: float,
                 checkpoints_dir: str,
                 pad_values: Dict[str, Union[int, float]],
                 rels_vocab: Vocab,
                 rels_dir_str: List[str],
                 rel_dir_idx: List[int],
                 neg_rel: str,
                 resume: Optional[str] = None) -> None:

        super(Trainer, self).__init__()
        self.device = device
        self.resume = resume
        self.epochs_num = epochs - 1  # epoch count start from 0
        self.epoch = 0
        # stop trianing if the model doesn't improve for n-validation epochs
        self.stop = early_stop
        # number of validation epochs in which model doesn't improve
        self.bad_epochs_num = 0
        # number of validation epochs to wait before decreases the lr if model
        # does not improve
        self.lr_patience = lr_patience

        # pad values
        self.pad_val = pad_values["token_pad_value"]
        self.es_tag = None if modes["end2end"] else es_tag

        # rels vocabs
        self.rels_vocab = rels_vocab
        self.rels_dir = rels_dir_str
        self.rel_dir_idx = rel_dir_idx
        self.neg_rel = neg_rel

        # criterion, optims and schedulers
        self.end2end = modes["end2end"]
        if self.end2end:
            self.ent_labrl_criterion = nn.CrossEntropyLoss(
                ignore_index=self.pad_val).to(device)
        self.rels_criterion = nn.CrossEntropyLoss().to(device)
        self.optim = optim
        self.cheduler = scheduler
        self.grad_clip_c = grad_clip  # gradient clip coeffecient

        if resume is None:
            time_tag = str(datetime.now().strftime("%d%m.%H%M"))
        else:
            time_tag = Path(resume).parent
        # Tensorboard writer
        log_dir = f"logs/exp_{time_tag}"
        self.logger = SummaryWriter(log_dir=f"{log_dir}/logs")
        self.loss_logger = SummaryWriter(log_dir=f"{log_dir}/loss")
        self.f1_logger = SummaryWriter(log_dir=f"{log_dir}/f1")
        self.acc_logger = SummaryWriter(log_dir=f"{log_dir}/acc")
        self.pr_logger = SummaryWriter(log_dir=f"{log_dir}/precsn_recall")

        # make folder for the experment
        checkpoints_dir = Path(checkpoints_dir) / f"{time_tag}"  # type: Path
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path = str(checkpoints_dir)

    def get_ground_truth(self, rels, rels_dir):
        gt_12 = torch.zeros_like(rels)
        gt_21 = torch.zeros_like(rels)

        # other relation (no direction will be in both 12 and 21)
        other_idx = torch.nonzero(rels_dir == self.rel_dir_idx[0])
        gt_12[other_idx] = rels[other_idx]
        gt_21[other_idx] = rels[other_idx]

        # e1,e2 direction
        dir_12 = torch.nonzero(rels_dir == self.rel_dir_idx[1])
        gt_12[dir_12] = rels[dir_12]

        # e2,e1 direction
        dir_21 = torch.nonzero(rels_dir == self.rel_dir_idx[2])
        gt_21[dir_21] = rels[dir_21]

        return gt_12, gt_21

    def run(self, model: torch.nn.Module,
            data_iters: List[torch.utils.data.DataLoader], seed: int):

        seed_everything(seed)
        if self.resume:
            pass

        model = model.to(self.device)

        # start
        model.train()
        main_pb = tqdm(range(self.epochs_num), unit="epoch")
        r = 0.
        for self.epoch in range(self.epochs_num):
            main_pb.set_description(f"loss: {r:.3f}")

            pb = tqdm(data_iters[0],
                      leave=False,
                      total=len(data_iters[0]),
                      unit="step")
            for step, batch in enumerate(pb):
                # set progress bar description and metrics
                pb.set_description(f"train: Step-{step+1:<3d}")
                batch = [b.to(self.device) for b in batch]
                self.optim.zero_grad()
                logits12, logits21 = model(batch, self.es_tag)
                gt_12, gt_21 = self.get_ground_truth(*batch[-2:])
                gt = torch.hstack((gt_12, gt_21))
                logits = torch.vstack((logits12, logits21))
                loss = self.rels_criterion(logits, gt)
                loss.backward()
                r = loss.item()
                self.optim.step()
