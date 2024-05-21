# README

## 别急

### quick start

```bash
# python env
conda create -n dm python=3.10
conda activate dm
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118 # u should change the cuda version according to your system. This works for me.
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.3.0+cu118.html # accordingly
pip -r requirements.txt
# conda install jupyter # for me to debug


# get data and embeddings
conda activate dm
mkdir data
cd data
wget https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz
gunzip titleabs.tsv.gz
wget https://download.loping151.com/files/dm2024/data/embeddings_cls.pth
wget https://download.loping151.com/files/dm2024/data/embeddings_mean.pth
wget https://download.loping151.com/files/dm2024/data/pre_tokenize.pth
# If you can't access ipv6, you will not be able to download the above 2 files.
# Then you should run: python embeddings.py to generate them.
cd ..


# run code, only baseline2 is finished now:
python train.py # --help or see args.py for args
python validate.py # also in args.py 
```

### Experiments

~~1. use the given 128 dim embedding to do classification (baseline1)~~

2. use scibert to directly get embeddings for each paper and do classification without node2vec (baseline2): now reach 0.678

3. use bert to train embeddings for each paper with node2vec and do classification: slow walker

4. use a random initialized bert, no node2vec: now reach 0.203

5. use random embedding: should be around 0.025. actually 0.107

### Key info to confirm
- The ogbn-arxiv dataset is a directed graph, each directed edge indicates that one paper cites another one.

Each paper comes with a 128-dimensional feature vector obtained by averaging the embeddings of the words obtained by running the skip-gram model over the MAG corpus.

- MAG is dead now.

- The task is to predict the 40 subject areas of arXiv CS papers, e.g., cs.AI, cs.LG, and cs.OS.

- Split: train: until 2017, validate: in 2018, and test: since 2019. For us, it doesn't matter since full label is given.

- Benchmark: https://paperswithcode.com/sota/node-property-prediction-on-ogbn-arxiv

### how to read the code

**If a function starts with '_', you don't need to pay much attention to it.**

not finished. ask me.

### file organize

```
- data/
  |- ogbn_arxiv/
  |- titleabs.tsv
  |- embeddings_cls.pth
  |- embeddings_mean.pth
  |- pre_tokenize.pth
- *.py
- readme.md
```
