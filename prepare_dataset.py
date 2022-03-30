from pathlib import Path
import json

from scripts.utils.dataset_process_utils import parse_arguments
from scripts.utils.dataset_process_utils import split_train_data
from scripts.utils.utils import seed_everything
from scripts.dataset.process_task8_files import process_file

if __name__ == "__main__":

    # parse command
    args = parse_arguments()
    dataset_dir = args.dataset_dir  # semeval task8 directory to text files
    save_dir = args.save_dir  # directory to save output files
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
    ids = list(range(len(data["train"][0])))
    y = [
        r + d if r != "Other" else r for r, d in list(zip(*data["train"][-2:]))
    ]
    train_id, val_id, _, _ = split_train_data(ids, y, config["val_size"], seed)
    train_data, val_data = [], []
