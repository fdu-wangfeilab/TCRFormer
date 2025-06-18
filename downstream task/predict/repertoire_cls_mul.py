from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os,sys
b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'model'))
sys.path.insert(0, b_directory)
from repertoire_cls_mul import *

b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'utils'))
sys.path.insert(0, b_directory)
from repertoire_cls_mul_util import *


import argparse

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataname', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--cuda', type=str, default='cuda:0')

    args = parser.parse_args()

    predict(args.dataname,args.path,args.cuda)

main()