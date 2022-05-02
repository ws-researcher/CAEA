import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser()
parser.add_argument("--dataDir", type=str, default="./dataSet/d_w_15k")

parser.add_argument("--input_dim", type=int, default=150)
parser.add_argument("--output_dim", type=int, default=150)
parser.add_argument("--rel_dim", type=int, default=100)
parser.add_argument("--attr_dim", type=int, default=100)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--head", type=int, default=1)

parser.add_argument("--epoach", type=int, default=1100)
from modelUtil import train


if __name__ == '__main__':
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--maxAttrNum", type=int, default=20)

    args = parser.parse_args()

    train(args)