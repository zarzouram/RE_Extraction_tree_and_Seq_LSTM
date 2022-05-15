from typing import List

import subprocess
import re

import torch
from torch import Tensor
from torch.nn.functional import log_softmax


def get_predictions(logits_12: Tensor, logits_21: Tensor, neg_rel: int,
                    rel_dir_idx: List[int]) -> Tensor:

    prob_12 = log_softmax(logits_12, dim=-1)  # (num_pair x num_rels)
    prob_21 = log_softmax(logits_21, dim=-1)  # (num_pair x num_rels)

    # Get predections in both directions
    preds_12_wneg = torch.argmax(prob_12, dim=-1)
    preds_21_wneg = torch.argmax(prob_21, dim=-1)

    # predicts negative relations iff both 12 and 21 directions are
    # negative relation, otherwise take the max prob of non negative rel
    neg_rel12 = preds_12_wneg == neg_rel  # neg relations in dir 12
    neg_rel21 = preds_21_wneg == neg_rel  # neg relations in dir 21
    neg_rels = torch.bitwise_and(neg_rel12, neg_rel21)

    # remove the neg_relation from logits, then get predictions
    prob_12p, prob_21p = prob_12.clone(), prob_21.clone()
    prob_12p[:, neg_rel] = -1 * torch.inf
    prob_21p[:, neg_rel] = -1 * torch.inf
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
    dir_preds[dir_12] = rel_dir_idx[1]
    dir_preds[~dir_12] = rel_dir_idx[2]
    dir_preds[neg_rels] = rel_dir_idx[0]

    return rel_preds, dir_preds


def decode_vocab(vocab, idx):
    return vocab.lookup_tokens(idx)


def write_results(idx: List[int], rel_str: List[str], dir_str: List[str],
                  key_answers_path: str):
    results_str = ""
    for i, r, d in zip(idx, rel_str, dir_str):
        if r != "Other":
            results_str += f"{str(i)}\t{r}{d}\n"
        else:
            results_str += f"{str(i)}\t{r}\n"

    with open(key_answers_path, "w") as f:
        f.write(results_str)


def calculate_scores(scorrer_path: str, results_path: str,
                     key_answers_path: str):

    official_res = "<<< (9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL >>>:"  # noqa: E501

    # pattren for precision, recall, f1 scores
    macro_results = r"MACRO-averaged result \(excluding Other\):\n"
    prf_pattern = re.compile(rf"(?<={macro_results}).+")

    # pattren for accurcy
    accuracy_results = r"Accuracy \(calculated for the above confusion matrix\) = "  # noqa: E501
    a_pattern = re.compile(rf"(?<={accuracy_results}).+")

    # pattern to find labels score
    ls_reg = r"(?:(?:[A-z][a-z]{4,12})-(?:[A-z][a-z]{4,12})|_Other)(?= : +).+"
    ls_reg = re.compile(ls_reg)

    # scores values pattren
    score_pattern = re.compile(r"\d+\.\d+")

    # run official scorer code
    result = subprocess.check_output(
        [scorrer_path, results_path, key_answers_path],
        stderr=subprocess.STDOUT)
    result = result.decode("utf-8")
    result = result.split(official_res)[-1]

    # get precision, recall, f1 scores
    result_prf = prf_pattern.search(result)
    prf = score_pattern.findall(result_prf.group())
    p, r, f1 = [float(score) for score in prf]

    # get accuracy
    result_acc = a_pattern.search(result)
    acc = score_pattern.findall(result_acc.group())
    acc = float(acc[0])

    # get labels score
    ls = {}
    labels_score = ls_reg.findall(result)
    for label_score in labels_score:
        label, scores = label_score.split(" :")
        prf = score_pattern.findall(scores)
        lp, lr, lf = [float(score) for score in prf]
        ls[label.strip()] = [lp, lr, lf]

    return p, r, f1, acc, ls, result
