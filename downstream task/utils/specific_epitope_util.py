import torch
import numpy as np
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings("ignore")
b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'model'))
sys.path.insert(0, b_directory)
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)
from specific_epitope import *

def predict(epitope, test_path, cuda='cuda:0'):
    roc=[]
    pr=[]
    for i in range(5):
        model=torch.load(os.path.join(test_path,epitope,str(i),'model.pt'),map_location=cuda)
        model.eval()
        beta=torch.tensor(np.load(os.path.join(test_path,epitope,str(i),'beta_test_emb.npy')),dtype=torch.float32).to(cuda)
        beta_st=torch.tensor(np.load(os.path.join(test_path,epitope,str(i),'beta_test_st.npy')),dtype=torch.float32).to(cuda)
        alpha=torch.tensor(np.load(os.path.join(test_path,epitope,str(i),'alpha_test_emb.npy')),dtype=torch.float32).to(cuda)
        alpha_st=torch.tensor(np.load(os.path.join(test_path,epitope,str(i),'alpha_test_st.npy')),dtype=torch.float32).to(cuda)
        test_label=pd.read_csv(os.path.join(test_path,epitope,str(i),'test.csv'))['Label'].to_numpy()

        preds=model(beta,beta_st,alpha,alpha_st).flatten().cpu().detach().numpy()

        roc.append(roc_auc_score(test_label,preds))
        pr.append(average_precision_score(test_label,preds))

    print('AUC-ROC:',np.mean(roc),' AUC-PR:',np.mean(pr))