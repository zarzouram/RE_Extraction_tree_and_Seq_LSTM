import argparse
from torch import Tensor

from gensim.models import KeyedVectors
from gensim.models.keyedvectors import KeyedVectors as VectorsData

import torch
from torchtext.vocab.vocab import Vocab
from torch.nn.init import xavier_uniform_


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LT2326 H21 Mohamed's Project")

    parser.add_argument("--dataset_dir",
                        type=str,
                        default="dataset/semeval_task8/processed/",
                        help="Directory contains task8 dataset.")

    parser.add_argument("--config_path",
                        type=str,
                        default="scripts/config.json",
                        help="Path for the configuration json file.")

    parser.add_argument("--checkpoint_dir",
                        type=str,
                        default="/srv/data/zarzouram/lt2011/checkpoints/",
                        help="Directory to save checkpoints.")

    parser.add_argument(
        "--scorer_path",
        type=str,
        default=  # noqa: E251
        "dataset/semeval_task8/SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl",  # noqa: E501
        help="Path to the official scorer path.")

    parser.add_argument("--scorer_result_dir",
                        type=str,
                        default="dataset/semeval_task8",
                        help="Directory to save temporary scorer results.")

    parser.add_argument(
        "--word2vec_path",
        type=str,
        default=  # noqa: E251
        "/srv/data/zarzouram/resources/embeddings/Word2Vec/wikien_223/model.bin",  # noqa: E501
        help="Directory to save checkpoints.")

    parser.add_argument(
        '--device',
        type=str,
        default="gpu",  # gpu, cpu
        help='Device to be used either gpu or cpu.')

    parser.add_argument("--resume",
                        type=str,
                        default="",
                        help='checkpoint filename.')

    args = parser.parse_args()

    return args


def init_unk_vector(size: int) -> Tensor:
    """initialize unkown word vectors. A function that takes in a Tensor and
        returns a weight Tensor of the same size"""
    weight_unk = torch.ones(1, size)
    return xavier_uniform_(weight_unk).view(-1)


def get_w2v_vectors(vocab: Vocab, w2v_model_path: str):
    # loading word2vec model
    w2v = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
    w2v: VectorsData
    tokens = vocab.get_stoi()  # tokens to index mapping

    w2v_dim = w2v.vector_size
    w2v_vectors = torch.from_numpy(w2v.vectors)  # all word2vec vectors
    w2v_stoi = w2v.key_to_index  # w2v vocabulary: string to index mapping
    # if our token is not found in word2vetor vocab initialize a sampled vector
    vectors = []
    for t in tokens:
        if t == "<pad>":
            vectors.append(torch.zeros(w2v_dim))
            continue

        idx = w2v_stoi.get(t)
        if idx is None:
            vectors.append(init_unk_vector(w2v_dim))
        else:
            vectors.append(w2v_vectors[idx])

    return torch.stack(vectors)
