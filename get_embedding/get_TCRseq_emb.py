import scanpy as sc
import os,sys
import numpy as np
from scipy.sparse import csr_matrix, vstack, save_npz,load_npz
import scipy
import time
import re
import os,copy
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
import anndata as ad
from torch.autograd import Variable
import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union
from functools import reduce
import torch
import numpy as np
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Bernoulli
from tqdm import trange
from math import floor
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import warnings
warnings.filterwarnings("ignore")

def is_whitespaced(seq):
    tok = list(seq)
    spaces = [t for t in tok if t.isspace()]
    if len(spaces) == floor(len(seq) / 2):
        return True
    return False

class sx_Dataset_beta(Dataset):
    def __init__(self,data1,data2,data3,data4):
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.len = data1.shape[0]
 
    def __getitem__(self, index):
        return self.x1[index], self.x2[index],self.x3[index], self.x4[index]
 
    def __len__(self):
        return self.len
    
class sx_Dataset_alpha(Dataset):
    def __init__(self,data1,data2,data3,):
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3

        self.len = data1.shape[0]
 
    def __getitem__(self, index):
        return self.x1[index], self.x2[index],self.x3[index]
 
    def __len__(self):
        return self.len
    
def get_betaseq_emb(read_path,save_path,cdr3_name='TRB_CDR3',
                    v_name='TRBV',
                    d_name='TRBD',
                    j_name='TRBJ',
                    delimiter='\t',device="cuda:0"):
    data=pd.read_csv(read_path,delimiter=delimiter)
    train_cdr3=data[cdr3_name].to_numpy()
    train_v=data[v_name].to_numpy()
    train_d=data[d_name].to_numpy()
    train_j=data[j_name].to_numpy()
    


    tcr_data=pd.Series(train_cdr3)
    aa_tokens = tcr_data.astype(str).apply(lambda x: list(x))
    aa_tks = [s if is_whitespaced(s) else " ".join(list(s)) for s in aa_tokens]
    v_dat=train_v
    d_dat=train_d
    j_dat=train_j
    v_=np.load('../model_save/beta_v_dic.npy')
    d_=np.load('../model_save/beta_d_dic.npy')
    j_=np.load('../model_save/beta_j_dic.npy')
    v_dic = {v_[i]: i for i in range(len(v_))}
    d_dic = {d_[i]: i for i in range(len(d_))}
    j_dic = {j_[i]: i for i in range(len(j_))}
    
    aa_dict = {'.': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
            'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '*': 21}
    a_idx = [[aa_dict[c] for c in s.split()] for s in aa_tks]

    v_data=[]
    d_data=[]
    j_data=[]

    for i in range(len(a_idx)):

        a_idx[i] = a_idx[i] + [aa_dict['.']] * (30 - len(a_idx[i]))
        a_idx[i] = np.array(a_idx[i])

        if type(v_dat[i])==float or v_dat[i] not in v_dic.keys():
            v_data.append(0)
        else:
            v_data.append(v_dic[v_dat[i]])

        if type(d_dat[i])==float or d_dat[i] not in d_dic.keys():
            d_data.append(0)
        else:
            d_data.append(d_dic[d_dat[i]])

        if type(j_dat[i])==float or j_dat[i] not in j_dic.keys():
            j_data.append(0)
        else:
            j_data.append(j_dic[j_dat[i]])

    a_idx=np.vstack(a_idx)
    v_data=np.array(v_data)
    d_data=np.array(d_data)
    j_data=np.array(j_data)
    
    

    SRC_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', 'model'))
    sys.path.insert(0, SRC_DIR)

    model=torch.load('../model_save/model_with_beta.pt')
    
    model=model.to(device)
    dataset=sx_Dataset_beta(a_idx,v_data,d_data,j_data)
    
    BATCH_SIZE=64
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,)
    Return_data=[]
    for index, (aad,v,d,j) in enumerate(train_loader):
        T_data=model.forward(torch.tensor(aad,dtype=torch.float32).to(device).long(),
                  torch.tensor(v,dtype=torch.float32).to(device).long(),
                 torch.tensor(d,dtype=torch.float32).to(device).long(),
                 torch.tensor(j,dtype=torch.float32).to(device).long(),return_encodings=True)
        Return_data.append(T_data.detach().cpu().numpy())
    Return_data=np.concatenate(Return_data, axis=0)
        
    np.save(os.path.join(save_path),Return_data)
        
def get_alphaseq_emb(read_path,save_path,cdr3_name='TRA_CDR3',
                    v_name='TRAV',
                    j_name='TRAJ',
                    delimiter='\t',device="cuda:0"):
    data=pd.read_csv(read_path,delimiter=delimiter)
    train_cdr3=data[cdr3_name].to_numpy()
    train_v=data[v_name].to_numpy()
    train_j=data[j_name].to_numpy()
    


    tcr_data=pd.Series(train_cdr3)
    aa_tokens = tcr_data.astype(str).apply(lambda x: list(x))
    aa_tks = [s if is_whitespaced(s) else " ".join(list(s)) for s in aa_tokens]
    v_dat=train_v
    j_dat=train_j
    v_=np.load('../model_save/alpha_v_dic.npy')
    j_=np.load('../model_save/alpha_j_dic.npy')
    v_dic = {v_[i]: i for i in range(len(v_))}
    j_dic = {j_[i]: i for i in range(len(j_))}
    
    aa_dict = {'.': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
            'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '*': 21}
    a_idx = [[aa_dict[c] for c in s.split()] for s in aa_tks]

    v_data=[]
    j_data=[]

    for i in range(len(a_idx)):

        a_idx[i] = a_idx[i] + [aa_dict['.']] * (30 - len(a_idx[i]))
        a_idx[i] = np.array(a_idx[i])

        if type(v_dat[i])==float or v_dat[i] not in v_dic.keys():
            v_data.append(0)
        else:
            v_data.append(v_dic[v_dat[i]])



        if type(j_dat[i])==float or j_dat[i] not in j_dic.keys():
            j_data.append(0)
        else:
            j_data.append(j_dic[j_dat[i]])

    a_idx=np.vstack(a_idx)
    v_data=np.array(v_data)
    j_data=np.array(j_data)
    
    

    SRC_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', 'model'))
    sys.path.insert(0, SRC_DIR)

    model=torch.load('../model_save/model_with_alpha.pt')
    model=model.to(device)

    dataset=sx_Dataset_alpha(a_idx,v_data,j_data)
    BATCH_SIZE=64
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,)
    Return_data=[]
    for index, (aad,v,j) in enumerate(train_loader):
        T_data=model.forward(torch.tensor(aad,dtype=torch.float32).to(device).long(),
                  torch.tensor(v,dtype=torch.float32).to(device).long(),
                 torch.tensor(j,dtype=torch.float32).to(device).long(),return_encodings=True)
        Return_data.append(T_data.detach().cpu().numpy())
    Return_data=np.concatenate(Return_data, axis=0)
        
    np.save(os.path.join(save_path),Return_data)
   