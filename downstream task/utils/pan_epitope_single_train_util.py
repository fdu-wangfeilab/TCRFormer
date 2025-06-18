import numpy as np
import pandas as pd
import torch,random
import sys
import os
import warnings
warnings.filterwarnings("ignore")
b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'model'))
sys.path.insert(0, b_directory)
from pan_epitope_single import *
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings("ignore")

class sx_Dataset(Dataset):
    def __init__(self,data1,data2,data3,):
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3

        self.len = data1.shape[0]
 
    def __getitem__(self, index):
        return self.x1[index],self.x2[index],self.x3[index]
 
    def __len__(self):
        return self.len

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cal_(beta_train_emb_path,ep_train_emb_path,train_health_tcr_emb_path,seed=1,lr=0.0001,EPOCH=20,BATCH_SIZE=64,device='cuda:0',save_path='./model.pt'):
    set_seed(seed)

    beta_train_emb=np.load(beta_train_emb_path)
    ep_train_emb=np.load(ep_train_emb_path)
    train_health_tcr_emb=np.load(train_health_tcr_emb_path)
    
    model=classification_model(tcr_dim=beta_train_emb.shape[-1], pep_dim=ep_train_emb.shape[-1], 
                               )
    model=model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_neg_nums=beta_train_emb.shape[0]
    
    for epoch in range(EPOCH):
        model.train()       
        random.seed(epoch)

        health_nums=random.randint(0,train_health_tcr_emb.shape[0]-train_neg_nums-1)
        dataset = sx_Dataset(beta_train_emb,train_health_tcr_emb[health_nums:health_nums+train_neg_nums],
                             ep_train_emb)
        train_dataloader=DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,drop_last=True)
        for tra_step, (cdr3_pos,Hcdr3_pos,pep) in enumerate(train_dataloader):   
            pep=torch.tensor(pep,dtype=torch.float32).to(device) 
            cdr3_pos=torch.tensor(cdr3_pos,dtype=torch.float32).to(device)    
            Hcdr3_pos=torch.tensor(Hcdr3_pos,dtype=torch.float32).to(device) 

            pos_pred = model(cdr3_pos,pep)
            pos_pred = pos_pred.flatten()
            
            health_pred = model(Hcdr3_pos,pep)
            health_pred = health_pred.flatten()
                  
            pos_loss = F.binary_cross_entropy(pos_pred,torch.ones_like(pos_pred).to(device))
            health_loss = F.binary_cross_entropy(health_pred,torch.zeros_like(health_pred).to(device))
            
            loss = pos_loss+health_loss
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
    torch.save(model,save_path)