# Relation Extraction using LSTMs on Sequences and Tree Structures

## Introduction

This repository hosts the course project for the "LT2311 H20 Språkteknologisk"
Course. The main goal of this project is to implement End-to-End Relation
Extraction using LSTMs on Sequences and Tree Structures [(Miwa & Bansal, ACL
2016)](#1) using DGL and PyTorch.

The document will first discuss the project's scope and show how to run the
code. The report will then discuss the model, its hyperparameters, loss, and
performance metrics. In the end, I will compare the model's performance
documented in the paper with my implemented model's performance.

## Project Description

[(Miwa & Bansal, ACL 2016)](#1) proposed an end-to-end model that extracts
entities and relations between them. The model is tested using two proprietary
datasets, ACE2005 and ACE2004, in an end-to-end fashion. Another open-source
dataset is used, which is SemEval-2010 Task 8. The SemEval-2010 Task 8 dataset
does not have entity tagging but has tagging for relations information only.
Thus, the model cannot be end-to-end trained. In other words, it takes the
indices for the entities to classify the relation between them.

The project considers the evaluation against SemEval-2010 Task 8 only. Thus the
current implementation does not support the end-to-end relation extraction.

## Run

### Requirements

The code was tested using python 3.8.12. Use `pip install -r requirements.txt`
to install the required libraries.

### Prepare dataset

To process SemEval-2010 Task 8 dataset run the folowing:

```bash
python prepare_dataset.py [ARGUMENTS]
```

The following arguments are expected:

  1. `dataset_dir`: Parent directory contains SemEval-2010 Task 8 text files.
  2. `train_path`: Relative path to the the train text file
  3. `test_path`: Relative path to the the test text file
  4. `config_path`: Path to the configuration file
  5. `save_dir`: Directory to save the output files

You can run the code using the default values of the arguments above.

The code will save under the `save_dir` the following files:

  1. An answer key text file `answer_keys.txt` in a format needed by task8
     scorer.
  2. Three `pth` that contains the processed data without mapping the string
     data to intger indices.
  3. Three `pth` that contains the processed data after encoding the string
     data.
  4. `vocab.pth` that contains the vocabulary dictionaries (string to indices
     and indices to string mapping).

### Train the model

To train the model run the following

```bash
python run_train.py [ARGUMENTS]
```
The following arguments are expected:

  1. `dataset_dir`: Parent directory contains SemEval-2010 Task 8 processed
     files.
  2. `config_path`: Path to the configuration file
  3. `checkpoint_dir`: Directory to save model checkpoints
  4. `scorer_path`: Path to the official scorer file
  5. `word2vec_path`: Path to the bin file of the word2vec trained on Wikipedia
  6. `device`: either gpu or cpu
  7. `resume`: if train resuming is needed pass the checkpoint filename
  8. `pretrained`: Path to the pretrained model which is trained to detect
     entities only.

You can run the code using the default values of the arguments above.

### Model testing

Model testing and evaluation is done in `run_test.ipynb` notebook.

## The Model

The model consists of the following modules:

  1. A sequence layer: a bidirectional one-layer LSTM. The layer creates
     learning embeddings for words and POS tags. The word embeddings are
     initialized by the pre-trained word2vec embeddings trained on Wikipedia.

  2. An entity detection layer: two fully connected layers stacked above the
     sequence layer. The first layer has a tanh activation function. The layer
     detects the entity labels for each token.

  3. A dependency layer: A bidirectional tree-LSTM creates a learning
     representation for a tree structure containing the two target words. The
     tree-LSTM takes a concatenation of the dependency label embedding, the
     corresponding BiLSTM hidden states, and the corresponding word label
     prediction from the entity detection layer. The tree structure could be as
     follows:

     1. The shortest path between the target words in the dependency tree.

     2. The dependency subtree under the lowest common ancestor of the target
        word pair. **[not impelemnted]**

     3. The full dependency tree of the whole sentence.

  4. A relation label detection layer: two fully connected layers stacked above
     the dependency layer. The first layer has a tanh activation function. The
     layer detects the relation label of the two target words.

When the model detects the relation label in an end-to-end fashion, the
dependency layer processes the tree structures for each possible combination of
the last words of the detected entities. As discussed above, my implementation
does not support the end-to-end operation, so the model expects to have the
target words indices instead of processing the detected entities combinations.
Figure 1 shows the model architecture.

<img
src="https://github.com/zarzouram/RE_Extraction_tree_and_Seq_LSTM/blob/main/imgs/model.png"
width="100%" padding="100px 100px 100px 100px">

Figure 1: Model Architecture. Green parts are not impelemented. Edited from
[(Miwa & Bansal, ACL 2016)](#1).

## References

<a id="1">(Miwa & Bansal, ACL 2016)</a> Miwa, M., & Bansal, M. (2016).
End-to-end relation extraction using lstms on sequences and tree structures.
arXiv preprint [arXiv:1601.00770v3](https://arxiv.org/abs/1601.00770).
