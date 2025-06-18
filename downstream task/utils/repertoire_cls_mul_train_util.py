import numpy as np
import pandas as pd
import torch,random
import sys
import os
import warnings
warnings.filterwarnings("ignore")
b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'model'))
sys.path.insert(0, b_directory)
from repertoire_cls_mul import *
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings("ignore")

class sx_Dataset(Dataset):
    def __init__(self,data1,data2):
        self.x1 = data1
        self.x2 = data2
        self.len = data1.shape[0]
 
    def __getitem__(self, index):
        return self.x1[index],self.x2[index]
 
    def __len__(self):
        return self.len

def Focal_loss(logits, targets,alpha=1, gamma=2):
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-loss)
    loss = (alpha * (1-pt)**gamma * loss).mean()

    return loss

def cal_(train_array_path,train_labels_path,seed=3,lr=0.0001,EPOCH=130,BATCH_SIZE=64, device='cuda:0', save_path='./model.pt'):
    train_array=np.load(train_array_path)
    train_labels=np.load(train_labels_path)

    torch.manual_seed(seed)
    model=classification_model(tcr_dim=train_array.shape[-1],nums=train_array.shape[-2],class_nums=len(set(train_labels)))
    
    model=model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    labels = np.array(train_labels)
    class_sample_count = np.bincount(labels)  
    weights_per_class = 1. / class_sample_count
    weights = weights_per_class[labels] 
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    for epoch in range(EPOCH):
        model.train()
        train_dataloader= DataLoader(dataset=sx_Dataset(train_array,train_labels),batch_size=BATCH_SIZE,sampler=sampler,num_workers=4,drop_last=True)

        for tra_step, (cdr3,label) in enumerate(train_dataloader):
            cdr3=torch.tensor(cdr3,dtype=torch.float32)
            cdr3=cdr3.to(device)
            
            label=torch.tensor(label,dtype=torch.float32)
            label=label.to(device)
            
            pred = model(cdr3)

            
            loss = Focal_loss(pred,label.long())
            
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

    torch.save(model,'./model.pt')