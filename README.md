<h1 align="center">
GePE: Generalizable Paper Embedding with Language-Driven Biased Random Walk
</h1>
<p align="center">
    Project of AI3602 Data Mining, 2024 Spring, SJTU
    <br />
    <a href="https://github.com/Loping151"><strong>Kailing Wang</strong></a>
    &nbsp;
    <a href="https://github.com/Shi-Soul"><strong>Weiji Xie</strong></a>
    &nbsp;
    <a href="https://github.com/xxyQwQ"><strong>Xiangyuan Xue</strong></a>
    &nbsp;
</p>
<p align="center">
    <a href="https://github.com/Loping151/DungeonMaster"> <img alt="Github Repository" src="https://img.shields.io/badge/Github-Repository-blue?logo=github&logoColor=blue"> </a>
    <a href="assets/slides.pdf"> <img alt="Presentation Slides" src="https://img.shields.io/badge/Presentation-Slides-green?logo=googlenews&logoColor=green"> </a>
    <a href='assets/report.pdf'> <img alt='Project Report' src='https://img.shields.io/badge/Project-Report-red?style=flat&logo=googlescholar&logoColor=red'> </a>
</p>

This project aims to ...

## 🛠️ Requirements

You can install them following the instructions below.

* Create a new conda environment and activate it:
  
    ```bash
    conda create -n gepe python=3.10
    conda activate gepe
    ```

* Install [pytorch](https://pytorch.org/get-started/previous-versions/) with appropriate CUDA version and corresponding `pyg-lib`, e.g.
  
    ```bash
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install pyg-lib -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
    ```

* Then install other dependencies:
  
    ```bash
    pip install -r requirements.txt
    ```

Latest version is recommended for all the packages, but make sure that your CUDA version is compatible with your `pytorch`.

## ⚓ Preparation

Before training, you should prepare the necessary dataset and embedding. In this project, we use [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) dataset for experiments. Although the graph can be downloaded automatically, you have to download the raw texts of titles and abstracts manually by running the following commands:

```bash
mkdir data && cd data
wget https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz
gunzip titleabs.tsv.gz
```

## 🚀 Training

...

## 💯 Evaluation

...

## 🤖 Demo

...