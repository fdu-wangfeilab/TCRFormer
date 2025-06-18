import os,sys
b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'model'))
sys.path.insert(0, b_directory)
from specific_epitope import *
b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'utils'))
sys.path.insert(0, b_directory)
from specific_epitope_train_util import *

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

    parser.add_argument('--beta_train_emb_path', type=str, required=True)
    parser.add_argument('--beta_train_st_path', type=str, required=True)
    parser.add_argument('--alpha_train_emb_path', type=str, required=True)
    parser.add_argument('--alpha_train_st_path', type=str, required=True)
    parser.add_argument('--train_labels_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--BATCH_SIZE', type=int, default=4)
    parser.add_argument('--EPOCH', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_path', type=str, default='./model.pt')

    args = parser.parse_args()
    cal_(args.beta_train_emb_path,args.beta_train_st_path,args.alpha_train_emb_path,args.alpha_train_st_path,
         args.train_labels_path,args.seed,args.lr,args.weight_decay,args.num_blocks,args.latent_dim,
         args.BATCH_SIZE,args.EPOCH,args.device,args.save_path)


main()