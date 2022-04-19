from typing import Dict

from collections import defaultdict
from statistics import mean

from pathlib import Path
import subprocess
import re


class TrackMetrics:

    def __init__(self, scorer_path, results_path, answer_keys_path) -> None:

        self.reset_running()
        self.metrics = self.init_metrics()

        self.scrr_p = scorer_path  # path to the official pl scorer code
        self.rp = results_path  # path to the preditions text file
        self.akp = answer_keys_path  # path to the ground truth text file

        # find the result part in the scorer retirn
        self.official_res = "<<< (9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL >>>:"  # noqa: E501

        # get precision, recall, f1 scores
        macro_results = r"MACRO-averaged result \(excluding Other\):\n"
        self.prf_pattern = re.compile(rf"(?<={macro_results}).+")

        # accurcy
        accuracy_results = r"Accuracy \(calculated for the above confusion matrix\) = "  # noqa: E501
        self.a_pattern = re.compile(rf"(?<={accuracy_results}).+")

        self.score_pattern = re.compile(r"\d+\.\d+")  # get score value

    def create_default_dict(self):

        metrics_dict = {
            "train": defaultdict(list, {}),
            "val": defaultdict(list, {})
        }

        return metrics_dict

    def reset_running(self):
        self.running = self.create_default_dict()

    def init_metrics(self):
        return self.create_default_dict()

    def update_running(self, metrics: Dict[str, float], phase: str) -> None:
        for name, value in metrics.items():
            self.running[phase][name].append(value)

    def update(self, phase: str):
        for name, values in self.running[phase].items():
            self.metrics[phase][name].append(mean(values))
        self.reset_running()

    def metrics_states(self):
        return {"running": self.running, "metrics": self.metrics}

    def load_states(self, states: dict):
        self.running = states["running"]
        self.metrics = states["metrics"]

    def calculate_scores(self, phase: str):
        # run official scorer code
        result = subprocess.check_output([self.scrr_p, self.rp, self.akp])
        result = result.decode("utf-8")

        # get precision, recall, f1 scores
        result_prf = self.prf_pattern.search(result)
        prf = self.score_pattern.findall(result_prf.group())
        p, r, f1 = [float(score) for score in prf]

        # get accuracy
        result_acc = self.a_pattern.search(result)
        acc = self.score_pattern.findall(result_acc.group())
        acc = float(acc[0])

        # delete answer keys and answers files
        Path(self.akp).unlink()
        Path(self.rp).unlink()

        self.update_running(
            {
                "precision": p,
                "recall": r,
                "f1": f1,
                "accuracy": acc
            }, phase)
