import numpy as np
import pandas as pd
import torch,random
import sys
import os
import warnings
warnings.filterwarnings("ignore")
b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'model'))
sys.path.insert(0, b_directory)
from pan_epitope_double import *
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings("ignore")

class sx_Dataset(Dataset):
    def __init__(self,data1,data2,data3,data4):
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.len = data1.shape[0]
 
    def __getitem__(self, index):
        return self.x1[index],self.x2[index],self.x3[index],self.x4[index]
 
    def __len__(self):
        return self.len

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cal_(beta_train_emb_path,alpha_train_emb_path,ep_train_emb_path,train_labels_path,seed=1,lr=0.0001,EPOCH=100,BATCH_SIZE=16, device='cuda:1',save_path='./model.pt'):

    beta_train_emb=np.load(beta_train_emb_path)
    alpha_train_emb=np.load(alpha_train_emb_path)
    ep_train_emb=np.load(ep_train_emb_path)
    train_labels=pd.read_csv(train_labels_path)['Target'].to_numpy()
    
    set_seed(seed)
    dataset = sx_Dataset(beta_train_emb,alpha_train_emb,ep_train_emb,train_labels)
    train_dataloader=DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,drop_last=True)

    model=classification_model(tcr_dim=beta_train_emb.shape[-1], pep_dim=ep_train_emb.shape[-1], 
                               )
    model=model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(EPOCH):
        model.train()       

        for tra_step, (btr,atr,pep,tl) in enumerate(train_dataloader):   
            pep=torch.tensor(pep,dtype=torch.float32).to(device) 
            btr=torch.tensor(btr,dtype=torch.float32).to(device)    
            atr=torch.tensor(atr,dtype=torch.float32).to(device) 
            tl=torch.tensor(tl,dtype=torch.float32).to(device)

            pred = model(btr,atr,pep).flatten()  
            loss = F.binary_cross_entropy(pred,tl)

            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            
    torch.save(model,save_path)

