import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'model'))
sys.path.insert(0, b_directory)
from library_cls_bin import *

def calculate_metrics(label, pred):
    
    return [accuracy_score(label, pred),precision_score(label, pred, average='weighted', zero_division=0),
           recall_score(label, pred, average='weighted', zero_division=0),f1_score(label, pred, average='weighted', zero_division=0),
           precision_score(label, pred, average='macro', zero_division=0),recall_score(label, pred, average='macro', zero_division=0),
           f1_score(label, pred, average='macro', zero_division=0)]


def predict(dataname, test_path, cuda='cuda:0'):

    model=torch.load(os.path.join(test_path,dataname,'model.pt'),map_location=cuda)
    model.eval()
    beta=torch.tensor(np.load(os.path.join(test_path,dataname,'test_cdr3.npy')),dtype=torch.float32).to(cuda)
    test_label=np.load(os.path.join(test_path,dataname,'test_labels.npy'))

    preds=model(beta).flatten().cpu().detach().numpy()

    accuracy,weighted_precision,weighted_recall,weighted_f1,macro_precision,macro_recall,macro_f1 = balanced_metrics(test_label,preds)

    print(' accuracy:',accuracy,'weighted:',weighted_precision,weighted_recall,weighted_f1,' macro:',macro_precision,macro_recall,macro_f1)