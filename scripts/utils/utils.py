from typing import List, Optional, Tuple, Union

import random
import os

from collections import Counter
from itertools import count
import re

import matplotlib.colors as mc
import colorsys
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.legend import Legend
from matplotlib.transforms import Bbox

import numpy as np
from numpy.typing import NDArray
import torch
import dgl


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed = seed


# %%
def adjust_lightness(color, amount=1.5):
    try:
        c = mc.cnames[color]
    except:  # noqa: E722
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_bar(rel: List[List[str]], dir: List[List[str]]) -> None:

    legend_title = ["train", "val", "test"]
    reg = re.compile(r"[A-Z-]")

    wf, hf = (1.2, 1.3)
    fig_w, fig_h = plt.rcParamsDefault["figure.figsize"]
    figsize = (fig_w * wf * 2, fig_h * hf * 2)
    fig, ax = plt.subplots(figsize=figsize, facecolor="w")
    fig.suptitle("Relation Labels Distribution per split", fontsize=18)

    # get color cycles
    colors = plt.get_cmap("Paired")
    colors_o = [plt.get_cmap("tab20c").colors[i] for i in [-4, -3, -2]]

    legends = []
    # bar plot for each data
    for i, (rl, rd) in enumerate(zip(rel, dir)):
        r12 = Counter([f"{r}" for r, d in zip(rl, rd) if d == "(e1,e2)"])
        r21 = Counter([f"{r}" for r, d in zip(rl, rd) if d == "(e2,e1)"])
        ro = Counter([f"{r}" for r, d in zip(rl, rd) if r == "Other"])
        if len(r12) != len(r21):
            if set(r12.keys()) - set(r21.keys()):
                diff = list(set(r12.keys()) - set(r21.keys()))[0]
                r21[diff] = 0
            else:
                diff = list(set(r21.keys()) - set(r12.keys()))[0]
                r12[diff] = 0

        r12_label, r12_count = zip(*sorted(r12.items(), key=lambda d: d[0]))
        _, r21_count = zip(*sorted(r21.items(), key=lambda d: d[0]))
        ro_count = ro["Other"]

        bar_width = 0.4
        bar_labels = list(map(lambda a: "".join(reg.findall(a)), r12_label))
        iter = count(start=bar_width * i, step=bar_width * (len(rel) + 1))
        xticks = list(round(next(iter), 1) for _ in range(len(r12_count) + 1))

        r12_dist = np.array(r12_count) / len(rl) * 100
        r21_dist = np.array(r21_count) / len(rl) * 100
        r0_dist = colors_o[i]

        bar12 = ax.bar(x=xticks[:-1],
                       height=r12_dist,
                       width=bar_width,
                       color=colors(i * 2),
                       ec=adjust_lightness(colors(i * 2)))
        bar21 = ax.bar(x=xticks[:-1],
                       height=r21_dist,
                       bottom=r12_dist,
                       width=bar_width,
                       color=colors(i * 2 + 1),
                       ec=adjust_lightness(colors(i * 2 + 1)))
        baro = ax.bar(x=[xticks[-1]],
                      height=[ro_count / len(rl) * 100],
                      width=bar_width,
                      color=r0_dist)

        leg = Legend(parent=ax,
                     handles=[bar12, bar21, baro],
                     labels=["e1 -> e2", "e2 -> e1", "Other"],
                     bbox_to_anchor=(0, 1),
                     loc="upper left",
                     frameon=False,
                     prop={"size": 13},
                     title_fontsize=14,
                     borderpad=0,
                     bbox_transform=ax.transAxes,
                     title=legend_title[i])
        leg_artist = ax.add_artist(leg)
        legends.append(leg_artist)

        if i == 1:
            bar_labels.append("Other")
            _ = ax.set_xticks(xticks)
            _ = ax.set_xticklabels(bar_labels)
            _ = ax.tick_params(axis="both", labelsize=14)

    _ = ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    _ = ax.spines["top"].set_visible(False)
    _ = ax.spines["right"].set_visible(False)
    _ = ax.spines["left"].set_visible(False)
    _ = ax.spines["bottom"].set_color("#DDDDDD")
    _ = ax.tick_params(bottom=False, left=False)
    _ = ax.yaxis.grid(True, color="#EEEEEE")
    _ = ax.xaxis.grid(False)
    _ = ax.set_xlabel("Relation Label", labelpad=15, fontsize=16)
    _ = ax.set_ylabel("Frequency", labelpad=15, fontsize=16)

    fig.canvas.draw()
    _ = plt.tight_layout(w_pad=1)
    inv_transform = fig.transFigure.inverted()
    for j, leg in enumerate(legends):
        f = leg.get_frame()
        x0, y0, w, h = f.get_bbox().bounds
        bboxbase_new = Bbox.from_bounds(x0 * j, y0, w, h)
        bboxbase_new = Bbox(inv_transform.transform(bboxbase_new)).bounds
        x0, y0, _, _ = bboxbase_new
        leg.set_bbox_to_anchor((x0, 1))


def plot_hist(data: List[NDArray],
              fig_data: dict,
              bins: Optional[Union[List[List[float]], List[int]]] = None,
              norm_pdf: bool = False,
              count: bool = False) -> Tuple[plt.Figure, plt.Axes]:

    # print histograms with normal distribution if required
    if "figsize_factor" in fig_data:
        wf, hf = fig_data["figsize_factor"]
    else:
        wf, hf = (1.2, 1.3)

    fig_w, fig_h = plt.rcParamsDefault["figure.figsize"]
    figsize = (fig_w * wf * len(data), fig_h * hf)
    figs, axes = plt.subplots(nrows=1,
                              ncols=len(data),
                              figsize=figsize,
                              squeeze=False,
                              facecolor="w")
    axes_ = np.array(axes).reshape(-1)

    # get color cycles
    hist_colors = plt.get_cmap("Accent")
    line_colors = plt.get_cmap("tab10")
    text_colors = plt.get_cmap("Set1")
    # plot histogram for each data
    for i, (ax, d) in enumerate(zip(axes_, data)):
        if bins is None:
            bins = [30] * len(data)
        density, _bins, _ = ax.hist(d,
                                    bins=bins[i],
                                    density=True,
                                    alpha=0.5,
                                    color=hist_colors(i),
                                    ec=adjust_lightness(hist_colors(i)),
                                    label=fig_data["label_h"][i])

        _ = ax.set_xticks(_bins)
        _ = ax.set_xticklabels([str(round(float(b), 5)) for b in _bins],
                               rotation=90)

        # show counts on hist
        if count:
            counts, _ = np.histogram(d, _bins)
            Xs = [(e + s) / 2 for s, e in zip(_bins[:-1], _bins[1:])]
            for x, y, count in zip(Xs, density, counts):
                _ = ax.text(x,
                            y * 1.02,
                            count,
                            horizontalalignment="center",
                            rotation=45,
                            color=text_colors(i))

        # plot normal probability dist
        if norm_pdf:
            # calc normal distribution of bleu4
            d_sorted = np.sort(d)
            mu = np.mean(d)
            sig = np.std(d)
            data_norm_pdf = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(
                -np.power((d_sorted - mu) / sig, 2.) / 2)

            _ = ax.plot(d_sorted,
                        data_norm_pdf,
                        color=line_colors(i),
                        linestyle="--",
                        linewidth=2,
                        label=fig_data["label_l"][i])

        _ = ax.legend()
        _ = ax.set_xlabel(fig_data["xlabel"])
        _ = ax.set_ylabel(fig_data["ylabel"])
        y_lim = ax.get_ylim()
        _ = ax.set_ylim((y_lim[0], y_lim[1] * 1.1))

    figs.suptitle(fig_data["title"])

    return figs, axes
