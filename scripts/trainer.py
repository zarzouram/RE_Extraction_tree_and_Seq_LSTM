from typing import List, Dict, Optional, Tuple, Union
from torch import Tensor

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.nn.functional import log_softmax
from torch.nn.utils import clip_grad_norm_

from sklearn.metrics import f1_score

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
                 lr_patience: int,
                 grad_clip: float,
                 checkpoints_dir: str,
                 pad_values: Dict[str, Union[int, float]],
                 ents_vocab: Vocab,
                 rels_vocab: Vocab,
                 rels_dir_str: List[str],
                 rel_dir_idx: List[int],
                 neg_rel: str,
                 scorer_path: str,
                 key_answers_path: str,
                 results_path: str,
                 tunning: bool = False,
                 resume: Optional[str] = None) -> None:

        super(Trainer, self).__init__()
        self.device = device
        self.resume = resume
        self.epochs_num = epochs - 1  # epoch count start from 0
        self.epoch = 0
        self.best_metric = -1e10000

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

        # ents labels id without pad
        ent_ids = ents_vocab.get_stoi()
        ent_pad = pad_values["entl_pad_value"]
        self.ent_id = [i for _, i in ent_ids.items() if i != ent_pad]

        # criterion, optims and schedulers
        self.end2end = modes["end2end"]
        self.pretrain = modes["pretrain"]
        if self.end2end or self.pretrain:
            # entity detection loss
            self.ent_label_criterion = nn.CrossEntropyLoss(
                ignore_index=ent_pad).to(device)
        elif not self.pretrain:
            # relation extraction loss
            self.rels_criterion = nn.CrossEntropyLoss().to(device)

        self.optim = optim
        self.scheduler = scheduler
        self.grad_clip = grad_clip  # gradient clip coeffecient

        self.tunning = tunning
        if resume is None:
            time_tag = str(datetime.now().strftime("%d%m.%H%M"))
        else:
            time_tag = Path(resume).parent
        # Tensorboard writer
        log_dir = f"logs/exp_{time_tag}"
        self.logger = SummaryWriter(log_dir=f"{log_dir}/logs")
        loss_logger = SummaryWriter(log_dir=f"{log_dir}/loss")
        f1_logger = SummaryWriter(log_dir=f"{log_dir}/f1")
        acc_logger = SummaryWriter(log_dir=f"{log_dir}/acc")
        pr_logger = SummaryWriter(log_dir=f"{log_dir}/precsn_recall")
        ent_f1_logger = SummaryWriter(log_dir=f"{log_dir}/ent_f1")
        self.writers = {
            "loss": loss_logger,
            "precision": pr_logger,
            "recall": pr_logger,
            "f1": f1_logger,
            "accuracy": acc_logger,
            "ent_f1": ent_f1_logger
        }

        # make folder for the experment
        checkpoints_dir = Path(checkpoints_dir) / f"{time_tag}"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path = str(checkpoints_dir)

        self.metrics_tracker = TrackMetrics(scorer_path, results_path,
                                            key_answers_path)
        self.results_path = results_path
        self.key_answers_path = key_answers_path

    def load_checkpoint(self):
        # load checkopoint
        state = torch.load(self.resume, map_location=torch.device("cpu"))
        model_state = state["model"]
        self.optim.load_state_dict(state["optim"])
        if self.scheduler and state["scheduler"]:
            self.scheduler = state["scheduler"]
        self.epoch = state["epoch"] + 1
        self.bad_epochs_num = state["bad_epochs_num"]
        self.best_metric = state["best_metric"]
        self.metrics_tracker.load_states(state["metrics"])

        return model_state

    def save_checkpoint(self, model, is_best: bool):
        model_state = model.state_dict()
        optim_state = self.optim.state_dict()
        if self.scheduler:
            scheduler_state = self.scheduler.state_dict()
        else:
            scheduler_state = None
        state = {
            "model": model_state,
            "optim": optim_state,
            "scheduler": scheduler_state,
            "epoch": self.epoch,
            "bad_epochs_num": self.bad_epochs_num,
            "best_metric": self.best_metric,
            "metrics": {
                "running": self.metrics_tracker.running,
                "metrics": self.metrics_tracker.metrics
            }
        }

        # set save path
        file_name = "checkpoint"
        if self.pretrain:
            file_name = f"{file_name}_pretrain"
        save_path = Path(self.checkpoints_path) / f"{file_name}.pth.tar"
        torch.save(state, save_path)

        if is_best:
            file_name = f"{file_name}_best"
            save_path = Path(self.checkpoints_path) / f"{file_name}.pth.tar"
            torch.save(state, save_path)

    def plot(self, phase):
        metrics = self.metrics_tracker.metrics[phase]
        for n, vs in metrics.items():
            self.writers[n].add_scalars(n, {phase: vs[-1]}, self.epoch)

    def write_key_answers(self, idx: Tensor, rel_gt: Tensor, dir_gt: Tensor):
        rel_str = self.rels_vocab.lookup_tokens(rel_gt.tolist())
        dir_str = [
            self.rels_dir[d] if d != self.rel_dir_idx[0] else ""
            for d in dir_gt
        ]

        answer_key_str = ""
        for i, r, d in zip(idx.tolist(), rel_str, dir_str):
            answer_key_str += f"{str(i)}\t{r}{d}\n"

        with open(self.key_answers_path, "w") as f:
            f.write(answer_key_str)

    def write_answers(self, idx: Tensor, rel_preds: Tensor, dir_preds: Tensor):
        rel_str = self.rels_vocab.lookup_tokens(rel_preds.tolist())
        dir_str = [
            self.rels_dir[d] if d != self.rel_dir_idx[0] else ""
            for d in dir_preds
        ]

        proposed_answer_str = ""
        for i, r, d in zip(idx.tolist(), rel_str, dir_str):
            proposed_answer_str += f"{str(i)}\t{r}{d}\n"

        with open(self.results_path, "w") as f:
            f.write(proposed_answer_str)

    def get_dir_ground_truth(self, rels: Tensor,
                             rels_dir: Tensor) -> Tuple[Tensor, Tensor]:
        # initialize grounf truth with neg rel
        gt_12 = torch.zeros_like(rels) + self.neg_rel
        gt_21 = torch.zeros_like(rels) + self.neg_rel

        # e1,e2 direction
        dir_12 = torch.nonzero(rels_dir == self.rel_dir_idx[1])
        gt_12[dir_12] = rels[dir_12]

        # e2,e1 direction
        dir_21 = torch.nonzero(rels_dir == self.rel_dir_idx[2])
        gt_21[dir_21] = rels[dir_21]

        return gt_12, gt_21

    def get_predictions(self, logits_12: Tensor, logits_21: Tensor) -> Tensor:

        prob_12 = log_softmax(logits_12, dim=-1)  # (num_pair x num_rels)
        prob_21 = log_softmax(logits_21, dim=-1)  # (num_pair x num_rels)

        # Get predections in both directions
        preds_12_wneg = torch.argmax(prob_12, dim=-1)
        preds_21_wneg = torch.argmax(prob_21, dim=-1)

        # predicts negative relations iff both 12 and 21 directions are
        # negative relation, otherwise take the max prob of non negative rel
        neg_rel12 = preds_12_wneg == self.neg_rel  # neg relations in dir 12
        neg_rel21 = preds_21_wneg == self.neg_rel  # neg relations in dir 21
        neg_rels = torch.bitwise_and(neg_rel12, neg_rel21)

        # remove the neg_relation from logits, then get predictions
        prob_12p, prob_21p = prob_12.clone(), prob_21.clone()
        prob_12p[:, self.neg_rel] = -1 * torch.inf
        prob_21p[:, self.neg_rel] = -1 * torch.inf
        probs_max12, rel_preds_12 = torch.max(prob_12p, dim=-1)
        probs_max21, rel_preds_21 = torch.max(prob_21p, dim=-1)
        probs = torch.stack((probs_max12, probs_max21), dim=-1)
        rel_preds_2dirs = torch.stack((rel_preds_12, rel_preds_21), dim=-1)

        # predections for non negative relation
        dir_preds = torch.argmax(probs, dim=-1)  # 0: dir_12, 1: dir_21

        # relation predictions, idx
        rel_preds = rel_preds_2dirs[torch.arange(dir_preds.size(0)), dir_preds]
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

        # training
        main_pb = tqdm(range(self.epoch, self.epochs_num), unit="epoch")
        for self.epoch in main_pb:
            main_pb.set_description("Training")

            model.train()
            pb = tqdm(data_iters[0],
                      leave=False,
                      total=len(data_iters[0]),
                      unit="step")
            for step, batch in enumerate(pb):
                # set progress bar description and metrics
                pb.set_description(f"train: Step-{step:<3d}")

                # move data to device
                batch = [b.to(self.device) for b in batch]

                model.zero_grad()
                outputs = model(batch, self.es_tag)

                # entity detection part
                if self.pretrain or self.end2end:
                    # calculate entity detection loss
                    _, _, ent_logits, ent_preds, ent_probs = outputs
                    num_ents = ent_logits.size(-1)
                    loss = self.ent_label_criterion(
                        ent_logits.view(-1, num_ents), batch[2].view(-1))
                    f1_ent = f1_score(batch[2].view(-1).tolist(),
                                      ent_preds.view(-1).tolist(),
                                      labels=self.ent_id,
                                      average="macro")
                    f1_ent = round(f1_ent, 4)

                # relation part
                if not self.pretrain:
                    # get ground truth
                    gt_12, gt_21 = self.get_dir_ground_truth(*batch[-2:])

                    # calcuate loss, metics scores
                    logits_12, logits_21, *_ = outputs
                    loss_12 = self.rels_criterion(logits_12, gt_12)
                    loss_21 = self.rels_criterion(logits_21, gt_21)
                    rel_loss = loss_12 + loss_21

                    # get predections
                    rel_preds, dir_preds = self.get_predictions(
                        logits_12, logits_21)

                    if self.end2end:
                        # loss = rel_loss + entity detection loss
                        # change predictions according to section 3.6
                        raise NotImplementedError
                    else:
                        loss = rel_loss

                loss.backward()
                clip_grad_norm_(model.parameters(), self.grad_clip)
                self.optim.step()

                self.metrics_tracker.update_running({"loss": loss.item()},
                                                    "train")
                if not self.pretrain:
                    # calculate and track metrices
                    samples_ids = batch[0]
                    self.write_key_answers(samples_ids, *batch[-2:])
                    self.write_answers(samples_ids, rel_preds, dir_preds)
                    self.metrics_tracker.calculate_scores("train")
                else:
                    self.metrics_tracker.update_running({"ent_f1": f1_ent},
                                                        "train")

            self.metrics_tracker.update("train")

            # eval
            model.eval()
            pb = tqdm(data_iters[1],
                      leave=False,
                      total=len(data_iters[1]),
                      unit="step")
            for step, batch in enumerate(pb):
                # set progress bar description and metrics
                pb.set_description(f"Validation: Step-{step:<3d}")

                with torch.no_grad():
                    # move data to device
                    batch = [b.to(self.device) for b in batch]
                    outputs = model(batch, self.es_tag)

                    # entity detection part
                    if self.pretrain or self.end2end:
                        # calculate entity detection loss
                        _, _, ent_logits, ent_preds, ent_probs = outputs
                        num_ents = ent_logits.size(-1)
                        loss_val = self.ent_label_criterion(
                            ent_logits.view(-1, num_ents), batch[2].view(-1))
                        f1_ent = f1_score(batch[2].view(-1).tolist(),
                                          ent_preds.view(-1).tolist(),
                                          labels=self.ent_id,
                                          average="macro")
                        f1_ent = round(f1_ent, 4)

                    # relation part
                    if not self.pretrain:
                        # get ground truth
                        gt_12, gt_21 = self.get_dir_ground_truth(*batch[-2:])

                        # calcuate loss, metics scors
                        logits_12, logits_21, *_ = outputs
                        loss_12 = self.rels_criterion(logits_12, gt_12)
                        loss_21 = self.rels_criterion(logits_21, gt_21)
                        rel_loss_val = loss_12 + loss_21

                        # get predections
                        rel_preds, dir_preds = self.get_predictions(
                            logits_12, logits_21)

                        if self.end2end:
                            # loss = loss + entity detection loss
                            # change predictions according to section 3.6
                            raise NotImplementedError
                        else:
                            loss_val = rel_loss_val

                    self.metrics_tracker.update_running(
                        {"loss": loss_val.item()}, "val")

                    if not self.pretrain:
                        # calculate and track metrices
                        samples_ids = batch[0]
                        self.write_key_answers(samples_ids, *batch[-2:])
                        self.write_answers(samples_ids, rel_preds, dir_preds)
                        self.metrics_tracker.calculate_scores("val")
                    else:
                        self.metrics_tracker.update_running({"ent_f1": f1_ent},
                                                            "val")

            self.metrics_tracker.update("val")

            self.plot("train")
            self.plot("val")
            if self.pretrain:
                metric = self.metrics_tracker.metrics["val"]["ent_f1"][-1]
            else:
                metric = self.metrics_tracker.metrics["val"]["f1"][-1]
            is_best = bool(metric >= self.best_metric)
            if is_best:
                self.best_metric = metric
            self.save_checkpoint(model, is_best)
