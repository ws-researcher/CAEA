# Semantics Hierarchical Graph Attention Embedding Network for Entity Alignment

## Dataset
We use four entity alignment datasets EN-FR-15K, EN-DE-15K, D-W-15K, and	D-Y-15K in our experiments, which can be downloaded from [OpenEA](https://github.com/nju-websoft/OpenEA)


## Installation

We recommend creating a new conda environment to install and run SHGAT.
```bash
conda create -n KGA python==3.7
conda activate KGA
pip install  -r requirements.txt
```
SHGAT use Bert to convert attribute-literals into vectors. You should download [bert-base-uncased](https://huggingface.co/bert-base-uncased/tree/main) and save it to KGA/bert-base-uncased folder.

### Running

run python main.py. In order to improve the training efficiency, the program will first embed the literal representation of the entire KG, which will take a long time. Then start training and testing.


> If you have any difficulty or question in running code and reproducing experimental results, please leave messages

