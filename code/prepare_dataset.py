from tqdm import tqdm

import re
from pathlib import Path
import json

from collections import Counter

import stanza
from utils.dataset_process_utils import parse_arguments


def check_bio_tag(bio_tag, e1, e2):
    cnt = Counter(bio_tag)
    assert cnt["b"] == 2
    if "i" in cnt:
        idx_b = [i for i, tag in enumerate(bio_tag) if tag == "b"]
        e1_new = re.sub(pattern=token[idx_b[0]], string=e1, repl="")
        e2_new = re.sub(pattern=token[idx_b[1]], string=e2, repl="")
        idx_i = [i for i, tag in enumerate(bio_tag) if tag == "i"]
        for id_i in idx_i:
            assert token[id_i] in e1_new or token[id_i] in e2_new


def tag_bio(ents_start, ents_end, token_start, token_end):

    o = (token_start < ents_start[0]) or (token_start >= ents_end[1]) or (
        (token_start >= ents_end[0]) and (token_start < ents_start[1]))

    b = token_start in ents_start

    i = (token_start > ents_start[0]
         and token_start < ents_end[0]) or (token_start > ents_start[1]
                                            and token_start < ents_end[1])

    assert (not o and not b and i) or (not o and b and not i) or (o and not b
                                                                  and not i)

    return "b" * b + "o" * o + "i" * i


if __name__ == "__main__":

    # parse command
    args = parse_arguments()
    dataset_dir = args.dataset_dir  # semeval task8 directory to text files
    save_dir = args.save_dir  # directory to save output files
    device = args.device

    # set some pathes
    train_path = Path(dataset_dir) / args.train_path
    test_path = Path(dataset_dir) / args.test_path
    config_path = args.config_path  # path to configuration file

    # load config files
    with open(config_path, "r") as json_file:
        config = json.load(json_file)["dataset"]

    # define stanza pipline, tokenize, POS tagging, and
    # dependency parsing
    stanza_models_dir = config["stanza_models_dir"]
    nlp = stanza.Pipeline(lang="en",
                          processors="tokenize,pos,lemma,depparse",
                          model_dir=stanza_models_dir)

    with open(train_path, "r") as file:
        texts = file.readlines()

    entity_regex = re.compile(r"(?<=<e[12]>)\w+(?:\W?\w)+(?=</e[12]>)")
    tag_regex = re.compile(r"</?e[12]>")
    d_quotes = re.compile(r"^\"|\"$")
    for ln in tqdm(range(25488, len(texts), 4)):
        idx, text = texts[ln].split("\t")
        text = d_quotes.sub(string=text.strip(), repl="")
        matches = list(entity_regex.finditer(text))
        text_notags = tag_regex.sub(string=text, repl="")
        e1s, e1e = matches[0].start() - 4, matches[0].end() - 4
        e2s, e2e = matches[1].start() - 13, matches[1].end() - 13
        e1, e2 = text_notags[e1s:e1e], text_notags[e2s:e2e]

        doc = nlp(text_notags)
        token = []
        pos = []
        bio_tag = []
        dep_tag = []
        dep_tree = []
        for sent in doc.sentences:
            for word in sent.words:
                token.append(word.text)
                pos.append(word.upos)
                bio_tag.append(
                    tag_bio([e1s, e2s], [e1e, e2e], word.start_char,
                            word.end_char))
                dep_tree.append(word.head - 1 if word.head > 0 else word.id -
                                1)
                dep_tag.append(word.deprel)
        check_bio_tag(bio_tag, e1, e2)

# %%
