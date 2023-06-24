import argparse
import os

import random
import numpy as np
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--dataDir", type=str, default="./dataSet/CSKG") # MED-BBK  CSKG

parser.add_argument("--input_dim", type=int, default=150)
parser.add_argument("--output_dim", type=int, default=150)
parser.add_argument("--rel_dim", type=int, default=100)
parser.add_argument("--attr_dim", type=int, default=100)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--head", type=int, default=1)

parser.add_argument("--epoach", type=int, default=1100)
from modelUtil import train

def set_seeds(seed=0):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = str(seed)

if __name__ == '__main__':
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--maxAttrNum", type=int, default=20)

    args = parser.parse_args()

    set_seeds(0)

    train(args)