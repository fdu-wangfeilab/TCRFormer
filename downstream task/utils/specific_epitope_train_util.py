import numpy as np
import pandas as pd
import torch,random
import sys
import os
import warnings
warnings.filterwarnings("ignore")
b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'model'))
sys.path.insert(0, b_directory)
from specific_epitope import *
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings("ignore")

class sx_Dataset(Dataset):
    def __init__(self,data1,data2,data3,data4,data5):
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5

        self.len = data1.shape[0]
        
    def __getitem__(self, index):
        return self.x1[index],self.x2[index],self.x3[index],self.x4[index],self.x5[index]
 
    def __len__(self):
        return self.len

def cal_(beta_train_emb_path,beta_train_st_path,alpha_train_emb_path,alpha_train_st_path,
         train_labels_path,seed=1,lr=0.0001,weight_decay=0.02,
         num_blocks=1,latent_dim=256,BATCH_SIZE=4,epoch=200,device='cuda:0',save_path='./model.pt'):

    beta_train_emb=np.load(beta_train_emb_path)
    beta_train_st=np.load(beta_train_st_path)
    alpha_train_emb=np.load(alpha_train_emb_path)
    alpha_train_st=np.load(alpha_train_st_path)
    train_labels=pd.read_csv(train_labels_path)['Label'].to_numpy()
    
    torch.manual_seed(seed)
    dataset=sx_Dataset(beta_train_emb,beta_train_st,
                       alpha_train_emb,alpha_train_st,train_labels)
    model=classification_model(beta_input_dim=beta_train_emb.shape[-1],beta_mid_dim=beta_train_emb.shape[-2],
                                beta_st_input_dim=beta_train_st.shape[-1],beta_st_mid_dim=beta_train_st.shape[-2],
                                alpha_input_dim=alpha_train_emb.shape[-1],alpha_mid_dim=alpha_train_emb.shape[-2],
                                alpha_st_input_dim=alpha_train_st.shape[-1],alpha_st_mid_dim=alpha_train_st.shape[-2],
                               num_blocks=num_blocks,latent_dim=latent_dim
                              )
    
    model=model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_itm=0
    for es in range(epoch):
        model.train()
        tcr_dataloader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,drop_last=True)
  
        for i,(be,bt,ae,at,labels) in enumerate(tcr_dataloader):
            be=torch.tensor(be,dtype=torch.float32)
            be=be.to(device)
            bt=torch.tensor(bt,dtype=torch.float32)
            bt=bt.to(device)
            
            ae=torch.tensor(ae,dtype=torch.float32)
            ae=be.to(device)
            at=torch.tensor(at,dtype=torch.float32)
            at=at.to(device)
            
            itm_labels=torch.tensor(labels,dtype=torch.float32)
            itm_labels=itm_labels.to(device)

            vl_output = model(be,bt,ae,at).flatten()
            loss_itm = F.binary_cross_entropy(vl_output.squeeze(), itm_labels, reduction='none').mean()

            total_itm+=loss_itm.item()

            optimizer.zero_grad()
            loss_itm.backward()
            optimizer.step()
            
    torch.save(model,save_path)