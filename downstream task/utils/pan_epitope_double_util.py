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
from pan_epitope_double import *

def predict(epitope, test_path, cuda='cuda:0', chain='double'):
    double_roc=[]
    double_pr=[]
    
    for i in range(5):

        beta=torch.tensor(np.load(os.path.join(test_path,epitope,str(i+1),'test_emb_beta.npy')),dtype=torch.float32).to(cuda)
        alpha=torch.tensor(np.load(os.path.join(test_path,epitope,str(i+1),'test_emb_alpha.npy')),dtype=torch.float32).to(cuda)
        ep_emb=torch.tensor(np.load(os.path.join(test_path,epitope,str(i+1),'test_emb_ep.npy')),dtype=torch.float32).to(cuda)
        test_label=pd.read_csv(os.path.join(test_path,epitope,str(i+1),'test.csv'))['Target'].to_numpy()

        
        model=torch.load(os.path.join(test_path,epitope,str(i+1),'model_double.pt'),map_location=cuda)
        model.eval()
        preds=model(cdr3_emb_beta=beta,cdr3_emb_alpha=alpha,epi_emb=ep_emb).flatten().cpu().detach().numpy()
        double_roc.append(roc_auc_score(test_label,preds))
        double_pr.append(average_precision_score(test_label,preds))

        
   
    print('double: AUC-ROC:',np.mean(double_roc),' AUC-PR:',np.mean(double_pr))

