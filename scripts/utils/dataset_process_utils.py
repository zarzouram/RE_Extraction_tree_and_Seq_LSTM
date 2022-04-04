from typing import List, Optional, Tuple

import argparse

from collections import Counter, OrderedDict
from itertools import chain

from sklearn.model_selection import train_test_split

import torch
from torchtext.vocab import vocab
from torchtext.vocab.vocab import Vocab


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LT2311 H20 Mohamed's Project")

    parser.add_argument("--dataset_dir",
                        type=str,
                        default="dataset/semeval_task8/original/",
                        help="Directory contains task8 dataset.")

    parser.add_argument("--train_path",
                        type=str,
                        default="TRAIN_FILE.TXT",
                        help="Relative path to the train text file.")

    parser.add_argument("--test_path",
                        type=str,
                        default="TEST_FILE_FULL.TXT",
                        help="Relative path to the test text file.")

    parser.add_argument("--config_path",
                        type=str,
                        default="scripts/config.json",
                        help="Directory contains configuration file's path.")

    parser.add_argument("--save_dir",
                        type=str,
                        default="dataset/semeval_task8/processed",
                        help="Directory to save the output files.")

    parser.add_argument("--check",
                        action="store_const",
                        default=True,
                        const=False)

    args = parser.parse_args()

    return args


def encode_data(data: List[List[str]],
                vocab: Vocab) -> Tuple[List[List[int]], List[int]]:

    encoded = [torch.LongTensor(vocab.lookup_indices(d)) for d in data]

    return encoded


def build_vocab(data: List[List[str]],
                min_freq: int = 1,
                special_tokens: Optional[List[str]] = None) -> Vocab:

    data_flat = list(chain.from_iterable(data))  # Type: List[str]
    data_bow = OrderedDict(Counter(data_flat).most_common())
    data_vocab: Vocab = vocab(data_bow,
                              min_freq=min_freq,
                              specials=special_tokens)

    return data_vocab


def split_train_data(data, labels, val_size, SEED):
    # split the training data into two splits validation and training

    # check that all lables counts >=2, if not remove it and asdd to train.
    label, label_count = zip(*Counter(labels).most_common())

    label_low_freq = []
    i = -1
    while label_count[i] == 1:
        label_low_freq.append(label[i])
        i -= 1

    data_new, labels_new = zip(*[(d, y) for d, y in zip(data, labels)
                                 if y not in label_low_freq])

    # split data after emoving data that occur only once
    x_train, x_val, y_train, y_val, = train_test_split(data_new,
                                                       labels_new,
                                                       test_size=val_size,
                                                       random_state=SEED,
                                                       stratify=labels_new)
    for llf in label_low_freq:
        idx = labels.index(llf)
        x_train.append(data[idx])
        y_train.append(labels[idx])

    return x_train, x_val, y_train, y_val
