from igfold import IgFoldRunner
import pandas as pd
import numpy as np
import torch
import os
from tqdm.notebook import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def get_final(stru_list,max_len = 30):
    padded_arrays = []
      
    for arr in stru_list:
        pad_width = ((0, 0), (0, max_len - arr.shape[1]), (0, 0))  
        padded_arr = np.pad(arr, pad_width, mode='constant')  
        padded_arrays.append(padded_arr)

    result_array = np.stack(padded_arrays)
    
    return result_array.squeeze()

def get_betastru_emb(read_path,save_path,cdr3_name='TRB_CDR3',delimiter='\t',padding_max_len = 30):
    igfold = IgFoldRunner()
    data=pd.read_csv(read_path,delimiter=delimiter)[cdr3_name].to_numpy()
    
    train_st_emb=[]
    
    for i in range(data.shape[0]):
        torch.cuda.empty_cache()
        sequences = {
        "H": data[i]}
        emb = igfold.embed(
            sequences=sequences, # Antibody sequences
        )
        torch.cuda.empty_cache()
        train_st_emb.append(emb.structure_embs.detach().cpu().numpy())
    
    train_st_emb=get_final(train_st_emb,padding_max_len)
    
    np.save(os.path.join(save_path),train_st_emb)
    
def get_alphastru_emb(read_path,save_path,cdr3_name='TRA_CDR3',delimiter='\t',padding_max_len = 30):
    igfold = IgFoldRunner()
    data=pd.read_csv(read_path,delimiter=delimiter)[cdr3_name].to_numpy()
    
    train_st_emb=[]
    
    for i in range(data.shape[0]):
        torch.cuda.empty_cache()
        sequences = {
        "L": data[i]}
        emb = igfold.embed(
            sequences=sequences, # Antibody sequences
        )
        torch.cuda.empty_cache()
        train_st_emb.append(emb.structure_embs.detach().cpu().numpy())
    
    train_st_emb=get_final(train_st_emb,padding_max_len)
    
    np.save(os.path.join(save_path),train_st_emb)
