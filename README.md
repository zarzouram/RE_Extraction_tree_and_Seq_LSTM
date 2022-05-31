# Relation Extraction using LSTMs on Sequences and Tree Structures

- [1. Introduction](#1-introduction)
- [2. Project Description](#2-project-description)
- [3. Run](#3-run)
  - [3.1. Requirements](#31-requirements)
  - [3.2. Prepare dataset](#32-prepare-dataset)
  - [3.3. Train the model](#33-train-the-model)
  - [3.4. Model testing](#34-model-testing)
- [4. Background](#4-background)
  - [4.1. The shortest path hypothesis](#41-the-shortest-path-hypothesis)
- [5. Model](#5-model)
  - [5.1. The authors' motivation](#51-the-authors-motivation)
    - [5.1.1. Neural Network Structure](#511-neural-network-structure)
    - [5.1.2. Tree structure](#512-tree-structure)
  - [5.2. The Original Model](#52-the-original-model)
- [6. Changes and pitfalls](#6-changes-and-pitfalls)
- [7. Datasets](#7-datasets)
- [8. Testing](#8-testing)
- [9. References](#9-references)

## 1. Introduction

This repository hosts the course project for the "LT2311 H20 Språkteknologisk"
Course. The main goal of this project is to implement End-to-End Relation
Extraction using LSTMs on Sequences and Tree Structures [(Miwa & Bansal, ACL
2016)](#1) using DGL and PyTorch.

The document will first discuss the project's scope and show how to run the
code. The report will then discuss the model, its hyperparameters, loss, and
performance metrics. In the end, I will compare the model's performance
documented in the paper with my implemented model's performance.

## 2. Project Description

[(Miwa & Bansal, ACL 2016)](#1) proposed an end-to-end model that extracts
entities and relations between them. The model is tested using two proprietary
datasets, ACE2005 and ACE2004, in an end-to-end fashion. Another open-source
dataset is used, which is SemEval-2010 Task 8. The SemEval-2010 Task 8 dataset
does not have entity tagging but has tagging for relations information only.
Thus, the model cannot be end-to-end trained. In other words, it takes the
indices for the entities to classify the relation between them.

The project considers the evaluation against SemEval-2010 Task 8 only. Thus the
current implementation does not support the end-to-end relation extraction.

## 3. Run

### 3.1. Requirements

The code was tested using python 3.8.12. Use `pip install -r requirements.txt`
to install the required libraries.

### 3.2. Prepare dataset

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

### 3.3. Train the model

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

### 3.4. Model testing

Model testing and evaluation is done in `run_test.ipynb` notebook.

## 4. Background

Relation extraction (RE) is a subtask of information extraction (IE) that aims
at detecting and classifying semantic relationships present in texts between
two or more entities. In the end-to-end extraction type, the task usually
consists of entity detection and classification and relation detection and
classification for the detected entities.

### 4.1. The shortest path hypothesis

In 2005 Mooney and Bunescu introduced a hypothesis related to determining a
relationship between two mentioned entities located in the same sentence. The
hypothesis suggests that the shortest path between two entities in the
undirected dependency graph almost solely encodes the information required to
determine the relationship between these two entities [(Bunescu & Mooney,
2005)](#2).

The hypothesis has three assumptions [(Bunescu & Mooney, 2005)](#2); first, it
assumes that the entities are mentioned in the same sentence. Second, the
information needed to extract the relationship is independent of the text
preceding or following the sentence. Finally, nodes represent the words in the
undirected dependency graph while the edges represent the dependency.

## 5. Model

### 5.1. The authors' motivation

#### 5.1.1. Neural Network Structure

The authors utilize the shortest path hypothesis discussed in [Section
4.1](#41-the-shortest-path-hypothesis).  Originally the hypothesis was
introduced and tested using a kernel-based model [(Bunescu & Mooney,
2005)](#2). Other studies successfully use the shortest path hypothesis using
neural-based models [(Miwa & Bansal, ACL 2016)](#1). In that sense, the
proposed model utilizes the dependency graph. However, the authors argued that
only relying on dependency information is not enough. As proof of concept and
to make the analysis more manageable, I checked the dependency relations
between entities where the shortest path is only two nodes (accounts for nearly
30% of the SemEval-2010 Task 8 dataset).  Figure 1 shows that the nominal
modifiers `nmod` is dominant across all relations, making it more challenging
to establish a relation between entities using only the dependency information.

<img
src="https://github.com/zarzouram/RE_Extraction_tree_and_Seq_LSTM/blob/main/imgs/depanalysis.png"
width="80%" padding="100px 100px 100px 100px">

Figure 1

Moreover, studies preceding Miwa’s and Bansal’s work ([Li et al., 2015](#3);
[Socher et al., 2012](#4); [Xu et al., 2015](#5)) show performance limitations
when extracting relation between two entities using LSTM neural networks [(Miwa
& Bansal, ACL 2016)](#1). The authors suggest that having such low performance
is due to focusing only on utilizing one linguistic structure (a tree structure
in this case) to extract the relation. The authors were able to push this low
performance to exceed state-of-the-art (when publishing their study) by jointly
modeling both entities and relations utilizing both sequence and tree LSTM.

#### 5.1.2. Tree structure

Unlike sequential LSTMs, the tree-LSTMs do not process the words in sequential
order. Instead, they process them according to their location in the tree data
structure representing the complex linguistic unit, like the dependency tree in
our example. Tai et al. introduce two types of tree-LSTMs: the child-sum and
N-ary LSTMs [(Tai et al., 2015)](#6). N-ary tree-LSTM needs a fixed number of
children for each node; thus, it is ideal for processing binarized constituency
trees. The child-sum can deal with different numbers of children, where the
state for each child has its weight in the forget gate. Thus, the child-sum
LSTM can selectively demolish or include the states of each child node. When
operating on the dependency tree, the child-sum LSTM can attend to certain
dependency relations more than others.

However, the goal is to attend to all nodes that belong to the shortest
path. This goal is achieved by sharing weights for the nodes that belong to the
shortest path and assigning different weights for all other nodes.

### 5.2. The Original Model

The model consists of the following modules:

1. **A sequence layer**: a bidirectional one-layer LSTM. The layer
   creates learning embeddings for words and POS tags. The word
   embeddings are initialized by the pre-trained word2vec embeddings
   trained on Wikipedia.

2. **An entity detection layer**: two fully connected layers stacked
   above the sequence layer. The first layer has a tanh activation
   function. The layer detects the entity labels for each token.

3. **A dependency layer**: A bidirectional tree-LSTM creates a learning
   representation for a tree structure containing the two target words.
   The tree-LSTM concatenates the dependency label embedding, the
   corresponding Bi-LSTM hidden states, and the related word label
   prediction from the entity detection layer. The tree structure could
   be as follows:

    1. The shortest path between the target words in the dependency
        tree.

    2. The dependency subtree under the lowest common ancestor of the
        target word pair.

    3. The full dependency tree of the whole sentence.

4. **A relation label detection layer**: two fully connected layers
   stacked above the dependency layer. The first layer has a tanh
   activation function. The layer detects the relation label of the two
   target words.

When the model detects the relation label in an end-to-end fashion, the
dependency layer processes the tree structures for each possible
combination of the last words of the detected entities.

## 6. Changes and pitfalls

As discussed in [Section 2](#2-project-description), I built a pipeline to test
my model implementation —mainly reimplementing Miwa’s and Bansal’s work [(Miwa
& Bansal, ACL 2016)](#1).  Three datasets were used: ACE 2004, ACE 2005, and
SemEval-2010 Task 8.  Unfortunately, ACE 2004 and ACE 2005 are proprietary
datasets, so I used SemEval-2010 Task 8 only.

As SemEval-2010 Task 8 does not have entity labeling, the implementation
pipeline does not support the end-to-end operation. The dependency layer
expects to have the target word indices instead of processing the detected
entities combinations. Also, I omit the entity detection layer. [Figure 2](#F2)
shows the model architecture.

<img
src="https://github.com/zarzouram/RE_Extraction_tree_and_Seq_LSTM/blob/main/imgs/model.png"
width="100%" padding="100px 100px 100px 100px">

<a id="F2">Figure 2:</a> Model Architecture. Parts highlighted in green are not impelemented.
Edited from [(Miwa & Bansal, ACL 2016)](#2).

One of the pitfalls of using a "non-end-to-end" pipeline is that the
model designed by the authors depends on pre-training of the entity
detection module, and skipping the pretraining phase could negatively
affect the performance. Also, the dependency layer will not take the
detected entity labels as inputs.

I used negative sampling in my implementation. The paper states the
following:

> "*we assign two labels to each word pair in prediction since we
> consider both left-to-right and right-to-left directions.*"

The above statement is hard to interpret for me. Suppose we have
detected/have two entities *e*<sub>*right*</sub> and
*e*<sub>*left*</sub>, so we will construct two pairs: the first
pair (*p*<sub>1</sub>) is
(*e*<sub>*right*</sub>,*e*<sub>*left*</sub>) and the
second pair (*p*<sub>2</sub>) is
(*e*<sub>*left*</sub>,*e*<sub>*right*</sub>). The truth
relation label we have is
*R*(*e*<sub>*left*</sub>,*e*<sub>*right*</sub>). According
to the statement above, I am unsure how to assign the label to the pairs
*p*<sub>1</sub> and *p*<sub>2</sub>. In my implementation, I assigned
the relation label *R* to *p*<sub>2</sub> (according to the direction)
and assigned the relation label *Other* to the other pair, which
is *p*<sub>1</sub>.

Also, I misunderstood the shortest path hypothesis. I thought I should find the
path between the two entities in a directed dependency graph.  However, the
hypothesis originally stated that the path is extracted from an undirected
version of the dependency graph [(Bunescu & Mooney, 2005)](#2). This affects
the implementation because you cannot find a path for a pair, say
(*w*<sub>1</sub>,*w*<sub>2</sub>), and the *w*<sub>2</sub> is directly
connected to *w*<sub>1</sub> in the dependency graph. Thus, I built one graph
for every two pairs and differentiated between the pairs by swapping the
location of the hidden state vectors generated by the top-down treeLSTM when
concatenating them in one vector to send them to the relation label detection
layer. The implementation should consider building a separate tree (shortest
path) for each pair.

## 7. Datasets

The SemEval-2010 Task 8  [(Hendrickx et al., 2019)](#7) dataset has nine
relation labels constructed between two nominals plus the "Other" label for no
relation. It has 8,000 training samples and 2,717 test samples. Eight hundred
samples are randomly selected from the training dataset to form the development
dataset. The dataset has its official scorer which produced the following:

- Confusion matrix

- Precision, recall, and F1 score for each label

- F1-score (Macro-F1) on the nine relation types.

## 8. Testing

The implemented model achieved a Macro-F1 score of 0.763 compared to 0.844
reported in the paper. The difference in performance could be due to the
differences in implementation discussed in [Section 6. Changes and pitfalls](#6-changes-and-pitfalls).

As shown in [Figure 3](#F3), the –confusion matrix, the Instrument-Agency relation has
the lowest true positive value (marked by green), hence the lowest accuracy.
However, from the test notebook, we notice that the Other label has the lowest
F1 score. By reviewing the confusion matrix, one can notice that the Other
label has the highest confusion; see the raw and the column marked by the blow
arrow. Also, [Figure 3](#F3) shows an easy confusion between the "Other" label and all
other labels marked by a blue square. Moreover, it is easier to mistake the
Entity-Destination and Content-Container labels for the Other label than the
vice versa (marked by a yellow square).

My analysis is that because the Instrument-Agency label has the lowest
occurrence in the train data test, it has the lowest true positive value. Also,
because we add a lot of negative samples "the Other label", the model hardly
can differentiate between the Other label and all other labels. Also, using one
tree graph per pair (right-to-left and left-to-right) as discussed in [Section
6. Changes and pitfalls](#6-changes-and-pitfalls), may contribute to this
confusing issue.

<img
src="https://github.com/zarzouram/RE_Extraction_tree_and_Seq_LSTM/blob/main/imgs/cm.png"
width="100%" padding="100px 100px 100px 100px">

<a id="F3">Figure 3</a>: Confusion Matrix - Shortest Path Mode.

## 9. References

<a id="1">(Miwa & Bansal, ACL 2016)</a> Miwa, M., & Bansal, M. (2016).
End-to-end relation extraction using lstms on sequences and tree structures.
arXiv preprint [arXiv:1601.00770v3](https://arxiv.org/abs/1601.00770).

<a id="2">(Bunescu & Mooney, 2005)</a> R. C. Bunescu and R. J. Mooney, “A
shortest path dependency kernel for relation extraction,” in *Proceedings of
the conference on human language technology and empirical methods in natural
language processing*, 2005, pp. 724–731.

<a id="3">(Li et al., 2015)</a> J. Li, M.-T. Luong, D. Jurafsky, and E. Hovy,
“When Are Tree Structures Necessary for Deep Learning of Representations?,”
*arXiv:1503.00185 \[cs\]*, Aug. 2015, Accessed: May 25, 2022.  \[Online\].
Available: <http://arxiv.org/abs/1503.00185>.

<a id="4">(Socher et al., 2012)</a> Y. Xu, L. Mou, G. Li, Y. Chen, H. Peng, and
Z. Jin, “Classifying relations via long short term memory networks along
shortest dependency paths,” in *Proceedings of the 2015 conference on empirical
methods in natural language processing*, 2015, pp. 1785–1794.

<a id="5">(Xu et al., 2015)</a> R. Socher, B. Huval, C. D. Manning, and A. Y.
Ng, “Semantic compositionality through recursive matrix-vector spaces,” in
*Proceedings of the 2012 joint conference on empirical methods in natural
language processing and computational natural language learning*, 2012, pp.
1201–1211.

<a id="6">(Tai et al., 2015)</a> K. S. Tai, R. Socher, and C. D. Manning,
“Improved Semantic Representations From Tree-Structured Long Short-Term Memory
Networks,” *arXiv:1503.00075 \[cs\]*, May 2015, Accessed: May 25, 2022.
\[Online\].  Available: <http://arxiv.org/abs/1503.00075>.

<a id="7">(Hendrickx et al., 2019)</a>I. Hendrickx et al., “Semeval-2010 task 8:
Multi-way classification of semantic relations between pairs of nominals,”
arXiv preprint arXiv:1911.10422, 2019
