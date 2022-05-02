# Semantics Hierarchical Graph Attention Embedding Network for Entity Alignment

## Dataset
We use four entity alignment datasets EN-FR-15K, EN-DE-15K, D-W-15K, and	D-Y-15K in our experiments, which can be downloaded from [OpenEA](https://github.com/nju-websoft/OpenEA)


## Installation

We recommend creating a new conda environment to install and run SHGAT.
```bash
conda create -n KGAlign python==3.7
conda activate KGAlign
pip install  -r requirements.txt
```
SHGAT use Bert to convert attribute-literals into vectors. You should download bert-base-uncased

### Running

First, put download datasets and bert-base-uncased. Then, change the path in run/args/KGAlign_args_15K.json and modules/utils/literal_encoder.py

run python run/main.py


> If you have any difficulty or question in running code and reproducing experimental results, please leave messages

