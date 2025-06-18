import os,sys
b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'model'))
sys.path.insert(0, b_directory)
from repertoire_cls_mul import *
b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'utils'))
sys.path.insert(0, b_directory)
from repertoire_cls_mul_train_util import *

import numpy as np
import pandas as pd
import torch,random
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")

import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_array_path', type=str, required=True)
    parser.add_argument('--train_labels_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--EPOCH', type=int, default=130)
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_path', type=str, default='./model.pt')

    args = parser.parse_args()

    cal_(args.train_array_path,args.train_labels_path,args.seed,
         args.lr,args.EPOCH,args.BATCH_SIZE,args.device,args.save_path)


main()