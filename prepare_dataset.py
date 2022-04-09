from pathlib import Path
import json
from itertools import zip_longest

import torch

from scripts.utils.dataset_process_utils import parse_arguments, build_vocab
from scripts.utils.dataset_process_utils import split_train_data, encode_data
from scripts.utils.utils import seed_everything
from scripts.dataset.process_task8_files import process_file

if __name__ == "__main__":

    # parse command
    args = parse_arguments()
    dataset_dir = args.dataset_dir  # semeval task8 directory to text files
    save_dir = Path(args.save_dir)  # directory to save output files
    check = args.check

    # set some pathes
    train_path = Path(dataset_dir) / args.train_path  # path to train txt file
    test_path = Path(dataset_dir) / args.test_path  # path to test txt file
    config_path = args.config_path  # path to configuration file

    # load config files
    with open(config_path, "r") as json_file:
        config = json.load(json_file)

    # seed
    seed = config["seed"]
    seed_everything(seed)

    config = config["dataset"]
    stanza_models_dir = config["stanza_models_dir"]

    # read deaset files and extract data
    data = {}
    for split_path, split in zip([train_path, test_path], ["train", "test"]):
        data[split] = process_file(split_path, split, stanza_models_dir,
                                   config["tags"], check)

    # split training data into validation and training splits
    # get indices for tain and val split based of the distribution of relation
    # label + relation direction
    print("splitting training data...")
    ids = list(range(len(data["train"][0])))
    y = [
        r + d if r != "Other" else r for r, d in list(zip(*data["train"][-2:]))
    ]
    train_id, val_id, _, _ = split_train_data(ids, y, config["val_size"], seed)
    # split training data based on the ids
    train_data, val_data = [], []
    for t_idx, v_idx in zip_longest(train_id, val_id):
        train_data.append(
            [data["train"][i][t_idx] for i in range(len(data["train"]))])
        if v_idx is not None:
            val_data.append(
                [data["train"][i][v_idx] for i in range(len(data["train"]))])

    data["train"] = list(zip(*train_data))
    data["val"] = list(zip(*val_data))
    print("splitting finished.")

    print("saving files...")
    # create vocab
    _, tokens, _, pos, dep, _, _, _, _, _, _, rels, rels_dir = data["train"]

    tokens_vocab = build_vocab(list(tokens), special_tokens=["<unk>", "<pad>"])
    pos_vocab = build_vocab(list(pos), special_tokens=["<unk>", "<pad>"])
    dep_vocab = build_vocab(list(dep), special_tokens=["<unk>", "<pad>"])
    rels_vocab = build_vocab([list(rels)])
    rels_dir_vocab = build_vocab([list(rels_dir)])

    tokens_vocab.set_default_index(tokens_vocab["<unk>"])
    pos_vocab.set_default_index(pos_vocab["<unk>"])
    dep_vocab.set_default_index(dep_vocab["<unk>"])

    # encode, convert to torch, save
    for split in ["train", "val", "test"]:
        data2save = {}
        tokens = list(data[split][1])
        pos = list(data[split][3])
        dep = list(data[split][4])
        rels = list(data[split][-2])
        rels_dir = list(data[split][-1])
        tokens_encoded = encode_data(tokens, tokens_vocab)
        pos_encoded = encode_data(pos, pos_vocab)
        dep_encoded = encode_data(dep, dep_vocab)
        rels_encoded = encode_data([rels], rels_vocab)[0]
        rels_dir_encoded = encode_data([rels_dir], rels_dir_vocab)[0]

        # save data
        idx = torch.LongTensor([int(i) for i in data[split][0]])
        trees = list(data[split][6:8])
        length = torch.LongTensor(data[split][9])
        e1_last, e2_last = zip(*data[split][5])
        e1_last, e2_last = torch.LongTensor(e1_last), torch.LongTensor(e2_last)

        data2save["idx"] = idx
        data2save["tokens"] = tokens_encoded
        data2save["pos"] = pos_encoded
        data2save["dep"] = dep_encoded
        data2save["length"] = length
        data2save["e1_last"] = e1_last
        data2save["e2_last"] = e2_last
        data2save["dep_tree"] = trees[0]
        data2save["shortest_path"] = trees[1]
        data2save["path_nodes"] = list(data[split][8])
        data2save["rels"] = rels_encoded
        data2save["rels_dir"] = rels_dir_encoded

        # write to desk
        file_name = f"{split}.pt"
        save_path = str(save_dir / file_name)
        torch.save(data2save, save_path)

        raw_file_name = f"raw_{split}.pt"
        raw_save_path = str(save_dir / raw_file_name)
        torch.save(data[split], raw_save_path)

    # save vocab
    file_name = "vocabs.pt"
    save_path = str(save_dir / file_name)
    vocabs = {
        "tokens": tokens_vocab,
        "pos": pos_vocab,
        "dep": dep_vocab,
        "rels": rels_vocab,
        "rel_dir": rels_dir_vocab
    }
    torch.save(vocabs, save_path)

    # write answer keys
    answer_keys = ""
    for lid, label, ldir in zip(idx, rels, rels_dir):
        answer_dir = ldir if label != "Other" else ""
        answer_keys += f"{lid}\t{label}{answer_dir}\n"
    file_name = "answer_keys.txt"
    save_path = str(save_dir / file_name)
    with open(save_path, "w") as file:
        file.write(answer_keys)

    print("done.")
