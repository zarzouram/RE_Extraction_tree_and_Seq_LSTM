import warnings

from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from scripts.dataset.dataloader import SemEvalTask8, collate_fn
from scripts.models.lstmer import LSTMER

from scripts.trainer import Trainer

from scripts.utils.utils import seed_everything
from scripts.utils.gpu_cuda_helper import select_device
from scripts.utils.train_utils import parse_arguments, get_w2v_vectors
from scripts.utils.train_utils import next_layer

from dgl.base import DGLWarning

if __name__ == "__main__":

    warnings.simplefilter("ignore", DGLWarning)

    # parse command arguments
    args = parse_arguments()
    ds_dir = args.dataset_dir  # processed dataset
    w2vm_path = args.word2vec_path
    resume = args.resume if args.resume != "" else None

    scorer_path = str(Path(args.scorer_path).resolve().expanduser())
    key_answers_path = Path(args.scorer_result_dir) / "answer_key.txt"
    results_path = Path(args.scorer_result_dir) / "proposed_answer.txt"

    # device
    device = select_device(args.device)
    print(f"selected device is {device}.\n")
    print("loading dataset...")

    # load confuguration file
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # seed
    seed = config["seed"]
    seed_everything(seed)

    # load vocabs, get pretrained vector weights
    vocabs = torch.load(Path(ds_dir) / "vocabs.pt")
    word_vocabs = vocabs["tokens"]
    w2v_vectors = get_w2v_vectors(word_vocabs, w2vm_path)

    # some vocab info
    pad_values = {
        "token_pad_value": word_vocabs["<pad>"],
        "pos_pad_value": vocabs["pos"]["<pad>"],
        "dep_pad_value": vocabs["dep"]["<pad>"],
        "entl_pad_value": vocabs["ents"]["<pad>"]
    }
    vocab_sizes = {
        "token_vocab_sz": len(word_vocabs),
        "pos_vocab_sz": len(vocabs["pos"]),
        "dep_vocab_sz": len(vocabs["dep"]),
        "entl_vocab_sz": len(vocabs["ents"]),
        "rel_num": len(vocabs["rels"])
    }

    # data iterators
    tree_type = config["model_hyperp"]["dep_tree"]["tree_type"]
    train_ds = SemEvalTask8(Path(ds_dir) / "train.pt")
    val_ds = SemEvalTask8(Path(ds_dir) / "val.pt")
    train_loader = DataLoader(train_ds,
                              **config["dataloader_parms"],
                              collate_fn=collate_fn(list(pad_values.values()),
                                                    tree_type))
    val_loader = DataLoader(val_ds,
                            **config["dataloader_parms"],
                            collate_fn=collate_fn(list(pad_values.values()),
                                                  tree_type))
    print("loading dataset finished.\n")

    # Construct the models and optimizer
    print("constructing model")
    model_hyp = config["model_hyperp"]
    model = LSTMER(**vocab_sizes,
                   **pad_values,
                   **model_hyp["embeddings"],
                   **model_hyp["entity_detect"],
                   **model_hyp["dep_tree"],
                   **model_hyp["rel_extract"],
                   **model_hyp["modes"],
                   dropouts=model_hyp["dropouts"])
    model.embeddings.token_embeddings.from_pretrained(w2v_vectors,
                                                      freeze=False)
    print(list(next_layer(model.named_children())))

    # Optimizer, scheduler
    lr = config["optim_params"]["lr"]
    l2reg = config["optim_params"]["l2reg"]
    gamma = config["optim_params"]["gamma"]
    skip_l2reg_list = [
        p for n, p in model.named_parameters() if "bias" in n or "norm" in n
    ]
    l2reg_list = [
        p for n, p in model.named_parameters()
        if "weight" in n and "norm" not in n
    ]
    optim_parms = [{
        "params": skip_l2reg_list,
        "weight_decay": 0.
    }, {
        "params": l2reg_list,
        "weight_decay": l2reg
    }]
    optim = Adam(optim_parms, lr=lr)
    scheduler = StepLR(optim, step_size=1, gamma=gamma)

    e1_tag = config["dataset"]["tags"]["ent"]["e1"]
    e2_tag = config["dataset"]["tags"]["ent"]["e2"]
    es_tag = [vocabs["ents"][e1_tag], vocabs["ents"][e2_tag]]
    rels_vocab = vocabs["rels"]
    rels_dir = vocabs["rel_dir"].get_itos()
    ndir = vocabs["rel_dir"][config["dataset"]["tags"]["rel"]["other"]]
    rel_12 = vocabs["rel_dir"][config["dataset"]["tags"]["rel"]["left"]]
    rel_21 = vocabs["rel_dir"][config["dataset"]["tags"]["rel"]["right"]]
    train = Trainer(optim=optim,
                    scheduler=scheduler,
                    device=device,
                    es_tag=es_tag,
                    modes=model_hyp["modes"],
                    resume=resume,
                    checkpoints_dir=args.checkpoint_dir,
                    pad_values=pad_values,
                    rels_vocab=rels_vocab,
                    rels_dir_str=rels_dir,
                    rel_dir_idx=[ndir, rel_12, rel_21],
                    neg_rel=config["dataset"]["tags"]["rel"]["neg_rel"],
                    results_path=str(results_path),
                    scorer_path=scorer_path,
                    key_answers_path=str(key_answers_path),
                    **config["train_parms"])
    train.run(model, [train_loader, val_loader], seed)
