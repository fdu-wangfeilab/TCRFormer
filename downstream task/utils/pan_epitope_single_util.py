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
from pan_epitope_single import *


def predict(epitope, test_path, cuda='cuda:0', datatype='mixpredtcr'):
    if datatype=='mixpredtcr':
        roc=[]
        pr=[]
        for i in range(5):
            model=torch.load(os.path.join(test_path,'test'+str(i+1), epitope,'model.pt'),map_location=cuda)
            model.eval()
            beta=torch.tensor(np.load(os.path.join(test_path,'test'+str(i+1), epitope,'test_cdr3_emb.npy')),dtype=torch.float32).to(cuda)
            ep_emb=torch.tensor(np.load(os.path.join(test_path,'test'+str(i+1), epitope,'ep_test_emb.npy')),dtype=torch.float32).to(cuda)
            test_label=pd.read_csv(os.path.join(test_path,'test'+str(i+1), epitope,'test.csv'))['Label'].to_numpy()
            
            preds=model(beta,ep_emb).flatten().cpu().detach().numpy()
            
            roc.append(roc_auc_score(test_label,preds))
            pr.append(average_precision_score(test_label,preds))
    
        print('AUC-ROC:',np.mean(roc),' AUC-PR:',np.mean(pr))

    if datatype=='singleindouble':
        roc=[]
        pr=[]
        for i in range(5):
            model=torch.load(os.path.join(test_path,epitope,str(i+1),'model_single.pt'),map_location=cuda)
            beta=torch.tensor(np.load(os.path.join(test_path,epitope,str(i+1),'test_emb_beta.npy')),dtype=torch.float32).to(cuda)
            ep_emb=torch.tensor(np.load(os.path.join(test_path,epitope,str(i+1),'test_emb_ep.npy')),dtype=torch.float32).to(cuda)
            test_label=pd.read_csv(os.path.join(test_path,epitope,str(i+1),'test.csv'))['Target'].to_numpy()
    
            preds=model(beta,ep_emb).flatten().cpu().detach().numpy()
    
            roc.append(roc_auc_score(test_label,preds))
            pr.append(average_precision_score(test_label,preds))
    
        print('AUC-ROC:',np.mean(roc),' AUC-PR:',np.mean(pr))