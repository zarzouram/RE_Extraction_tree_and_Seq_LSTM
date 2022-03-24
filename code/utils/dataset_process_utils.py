import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LT2311 H20 Mohamed's Project")

    parser.add_argument("--dataset_dir",
                        type=str,
                        default="code/dataset",
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
                        default="code/config.json",
                        help="Directory contains configuration file's path.")

    parser.add_argument("--save_dir",
                        type=str,
                        default="code/dataset",
                        help="Directory to save the output files.")

    parser.add_argument(
        "--device",
        type=str,
        default="gpu",  # gpu, cpu
        help="device to be used, gpu or cpu.")

    args = parser.parse_args()

    return args
