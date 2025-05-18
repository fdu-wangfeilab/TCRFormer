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

def balanced_metrics(labels, preds):
    auc = np.round(roc_auc_score(labels, preds),3)

    pred_labels = (preds > 0.5).astype(int)
    
    
    accuracy = np.round(accuracy_score(labels, pred_labels),3)
    precision = np.round(precision_score(labels, pred_labels),3)
    recall = np.round(recall_score(labels, pred_labels),3)
    f1 = np.round(f1_score(labels, pred_labels),3)
    
    return auc,accuracy,precision,recall,f1


def predict(dataname, test_path, cuda='cuda:0'):

    model=torch.load(os.path.join(test_path,dataname,'model.pt'),map_location=cuda)
    model.eval()
    beta=torch.tensor(np.load(os.path.join(test_path,dataname,'test_cdr3.npy')),dtype=torch.float32).to(cuda)
    test_label=np.load(os.path.join(test_path,dataname,'test_labels.npy'))

    preds=model(beta).flatten().cpu().detach().numpy()

    auc,accuracy,precision,recall,f1 = balanced_metrics(test_label,preds)


    print('AUC:',auc,' accuracy:',accuracy,' precision:',precision,' recall:',recall,' f1 score:',f1)