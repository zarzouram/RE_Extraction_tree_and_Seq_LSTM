from typing import List, Dict, Optional, Tuple, Union
from torch import Tensor

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn

from torchtext.vocab.vocab import Vocab

from torch.utils.tensorboard import SummaryWriter

from scripts.utils.utils import seed_everything
from scripts.metrics import TrackMetrics


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
                 scorer_path: str,
                 key_answers_path: str,
                 results_path: str,
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
        self.rel_dir_idx = rel_dir_idx  # nodir, dir_12, dir_21
        self.neg_rel = rels_vocab[neg_rel]

        # criterion, optims and schedulers
        self.end2end = modes["end2end"]
        self.pretrain = modes["pretrain"]
        if self.end2end or self.pretrain:
            # entity detection loss
            self.ent_labrl_criterion = nn.CrossEntropyLoss(
                ignore_index=self.pad_val).to(device)
        elif not self.pretrain:
            # relation extraction loss
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

        self.metrics_tracker = TrackMetrics(scorer_path, results_path,
                                            key_answers_path)
        self.results_path = results_path
        self.key_answers_path = key_answers_path

    def write_key_answers(self, gt: Tensor):
        pass

    def write_answers(self, rel_preds: Tensor, dir_preds: Tensor):
        pass

    def get_ground_truth(self, rels: Tensor,
                         rels_dir: Tensor) -> Tuple[Tensor, Tensor]:
        # initialize grounf truth with no dir
        gt_12 = torch.zeros_like(rels) + self.rel_dir_idx[0]
        gt_21 = torch.zeros_like(rels) + self.rel_dir_idx[0]

        # e1,e2 direction
        dir_12 = torch.nonzero(rels_dir == self.rel_dir_idx[1])
        gt_12[dir_12] = rels[dir_12]

        # e2,e1 direction
        dir_21 = torch.nonzero(rels_dir == self.rel_dir_idx[2])
        gt_21[dir_21] = rels[dir_21]

        return gt_12, gt_21

    def get_predictions(self, logits_12: Tensor, logits_21: Tensor) -> Tensor:
        with torch.no_grad():
            logits_12p = logits_12.clone()  # (num_pair x num_rels)
            logits_21p = logits_21.clone()  # (num_pair x num_rels)

            # Get predections in both directions
            preds_12_wneg = torch.argmax(logits_12p, dim=-1)
            preds_21_wneg = torch.argmax(logits_21p, dim=-1)

            # predicts negative relations iff both 12 and 21 directions are
            # negative relation, otherwise take the max prob of non negative rel
            neg_rel12 = preds_12_wneg == self.neg_rel  # neg relations in dir 12
            neg_rel21 = preds_21_wneg == self.neg_rel  # neg relations in dir 21
            neg_rels = torch.bitwise_and(neg_rel12, neg_rel21)

            # remove the neg_relation from logits, then get predictions
            logits_12p[:, self.neg_rel] = -1 * torch.inf
            logits_21p[:, self.neg_rel] = -1 * torch.inf
            probs_12, rel_preds_12 = torch.max(logits_12p, dim=-1)
            probs_21, rel_preds_21 = torch.max(logits_21p, dim=-1)
            probs = torch.stack((probs_12, probs_21), dim=-1)  # (num_pair x 2)
            rel_preds_2dirs = torch.stack((rel_preds_12, rel_preds_21), dim=-1)

            # predections for non negative relation
            dir_preds = torch.argmax(probs, dim=-1)  # 0: dir_12, 1: dir_21

            # relation predictions, idx
            rel_preds = rel_preds_2dirs[torch.arange(dir_preds.size(0)),
                                        dir_preds]
            rel_preds[neg_rels] = preds_12_wneg[neg_rels]

            # direction prediction
            dir_12 = dir_preds == 0
            dir_preds[dir_12] = self.rel_dir_idx[1]
            dir_preds[~dir_12] = self.rel_dir_idx[2]
            dir_preds[neg_rels] = self.rel_dir_idx[0]

        return rel_preds, dir_preds

    def run(self, model: torch.nn.Module,
            data_iters: List[torch.utils.data.DataLoader], seed: int):

        seed_everything(seed)
        if self.resume:
            pass

        model = model.to(self.device)

        # start training
        model.train()
        main_pb = tqdm(range(self.epochs_num), unit="epoch")
        for self.epoch in main_pb:
            main_pb.set_description("Training")

            pb = tqdm(data_iters[0],
                      leave=False,
                      total=len(data_iters[0]),
                      unit="step")
            for step, batch in enumerate(pb):
                # set progress bar description and metrics
                pb.set_description(f"train: Step-{step+1:<3d}")
                # move data to device
                batch = [b.to(self.device) for b in batch]

                self.optim.zero_grad()
                outputs = model(batch, self.es_tag)

                # get groung truth
                gt_12, gt_21 = self.get_ground_truth(*batch[-2:])

                if self.pretrain or self.end2end:
                    # calculate entity detection loss
                    # outputs = edl_logits, edl_preds
                    raise NotImplementedError

                if not self.pretrain:
                    logits_12, logits_21 = outputs

                    # calcuate loss, metics scors
                    loss_12 = self.rels_criterion(logits_12, gt_12)
                    loss_21 = self.rels_criterion(logits_21, gt_21)
                    loss = loss_12 + loss_21

                    # get predections
                    rel_preds, dir_preds = self.get_predictions(
                        logits_12, logits_21)

                    if self.end2end:
                        # loss = loss + entity detection loss
                        # change predictions according to section 3.6
                        raise NotImplementedError

                loss.backward()
                self.optim.step()
