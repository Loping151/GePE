# README

## 急

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
wget https://download.loping151.com/files/dm2024/data/pre_tokenize_distilbert.pth
# If you can't access ipv6, you will not be able to download the above 2 files.
# Then you should run: python embeddings.py to generate them.
cd ..


# run code, only baseline2 is finished now:
python train.py # --help or see args.py for args
python validate_cls.py --model_type pretrained_bert --pretrain your_model.pth # classifiation validation. also in args.py 
python validate_lp.py --model_type pretrained_bert --pretrain your_model.pth # link prediction. also in args.py


# start recommendation application
PYTHONPATH=. python app/rs.py --model_type scibert # the default pretrained model path is set in args.py

export no_proxy="localhost,127.0.0.1" # you'd probably used a proxy to access huggingface in China. If so, add this.
PYTHONPATH=. python app/rs_gradio.py --model_type scibert
```

### Experiments

~~1. use the given 128 dim embedding to do classification (baseline1)~~

2. use scibert to directly get embeddings for each paper and do classification without node2vec (baseline2): now reach 0.678

3. use bert to train embeddings for each paper with node2vec and do classification: slow walker

4. use a random initialized bert, no node2vec: now reach 0.203

5. use random embedding: should be around 0.025. actually 0.107

6. use a base embedding for word2vec, reach 0.598

### Key info to confirm
- Benchmark: https://paperswithcode.com/sota/node-property-prediction-on-ogbn-arxiv

### file organize

```
- data/
  |- ogbn_arxiv/
  |- titleabs.tsv
  |- embeddings_cls.pth
  |- embeddings_mean.pth
  |- pre_tokenize.pth
  |- other similar stuff
- *.py
- readme.md
```
