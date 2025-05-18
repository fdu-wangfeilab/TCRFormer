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
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from performer_pytorch import PerformerLM

#config
BATCH_SIZE=64
MASK_PROB = 0.15
REPLACE_PROB = 0.9
RANDOM_TOKEN_PROB = 0
MASK_TOKEN_ID = 21
PAD_TOKEN_ID = 21
MASK_IGNORE_TOKEN_IDS =[0]

# get the random prob matrix and True means smaller than prob threshold
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

# get the mask matrix which cannot be masked
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)      # num of mask of a single sequence in average
    num_tokens = mask.sum(dim=-1, keepdim=True)     # num of pure tokens of each sequence except special tokens
    mask_excess = torch.cat((torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0)))).reshape(mask.size(0),mask.size(-1)).to(device)
    mask_excess = (mask_excess >= (num_tokens * prob).ceil())        # only 15% of pure tokens can be masked
    mask_excess = mask_excess[:, :max_masked]       # get difference between 15% of pure tokens and 15% of all tokens
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)     # rand (0-1) as prob, special token use -1e9
    _, sampled_indices = rand.topk(max_masked, dim=-1)      # get index of topk prob to mask
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)        # delete difference of mask not pure
    new_mask = torch.zeros((batch, seq_len + 1), device=device)     # get (batch, seq_len) shape zero matrix
    new_mask.scatter_(-1, sampled_indices, 1)       # set masks in zero matrix as 1
    return new_mask[:, 1:].bool()       # the final mask, True is mask

def data_mask(data,
    mask_prob = MASK_PROB,
    replace_prob = REPLACE_PROB,
    num_tokens = None,
    random_token_prob = RANDOM_TOKEN_PROB,
    mask_token_id = MASK_TOKEN_ID,
    pad_token_id = PAD_TOKEN_ID,
    mask_ignore_token_ids = MASK_IGNORE_TOKEN_IDS
):
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
    # also do not include these special tokens in the tokens chosen at random
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)   # ignore_token as True, will not be masked later
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)      # get the True/False mask matrix
    # get mask indices
    ## mask_indices = torch.nonzero(mask, as_tuple=True)   # get the index of mask(nonzero value of mask matrix)
    # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
    masked_input = data.clone().detach()
    # if random token probability > 0 for mlm
    if random_token_prob > 0:
        assert num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
        random_token_prob = prob_mask_like(data, random_token_prob)       # get the mask matrix of random token replace
        random_tokens = torch.randint(0, num_tokens, data.shape, device=data.device)     # generate random token matrix with the same shape as input
        random_no_mask = mask_with_tokens(random_tokens, mask_ignore_token_ids)        # not masked matrix for the random token matrix
        random_token_prob &= ~random_no_mask        # get the pure mask matrix of random token replace
        random_indices = torch.nonzero(random_token_prob, as_tuple=True)        # index of random token replace
        masked_input[random_indices] = random_tokens[random_indices]        # replace some tokens by random token
    # [mask] input
    replace_prob = prob_mask_like(data, replace_prob)     # get the mask matrix of token being masked
    masked_input = masked_input.masked_fill(mask * replace_prob, mask_token_id)        # get the data has been masked by mask_token
    # mask out any tokens to padding tokens that were not originally going to be masked
    labels = data.masked_fill(~mask, pad_token_id)        # the label of masked tokens
    return masked_input, labels

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class sx_Dataset(Dataset):
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

def train(train_data,v_data,d_data,j_data,model_save_path,loss_save_path,
          BATCH_SIZE=64,EPOCHS=100,lr=0.00001,weight_decay=0.02,device = torch.device("cuda:0"),warmup_steps=10000,GRADIENT_ACCUMULATION=60,
         max_seq_len=30,dim=1024,depth=12,heads=8,dim_head=128,):
    dataset=sx_Dataset(train_data,v_data,d_data,j_data)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,)
    transformer_model = PerformerLM(V_Beta_size=202,D_Beta_size=11,J_Beta_size=68,max_seq_len=max_seq_len,dim=dim,depth=depth,heads=heads,dim_head=dim_head,)
    optim=torch.optim.AdamW(transformer_model.parameters(),lr=lr,weight_decay=weight_decay)
    
    optim_schedule = ScheduledOptim(optim, 36, n_warmup_steps=warmup_steps)
    transformer_model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN_ID, reduction='mean')
    softmax = nn.Softmax(dim=-1)
    transformer_model.train()
    
    for i in range(1, EPOCHS+1):
        running_loss = 0.0
        cum_acc = 0.0
        loss_save=[]
        auc_save=[]
        tmp_index=[]

        for index, (data,v,d,j) in enumerate(train_loader):
            index += 1
            data=torch.tensor(data,dtype=torch.float32).to(device)
            v=torch.tensor(v.squeeze(),dtype=torch.float32).to(device)
            d=torch.tensor(d.squeeze(),dtype=torch.float32).to(device)
            j=torch.tensor(j.squeeze(),dtype=torch.float32).to(device)

            data, labels = data_mask(data,num_tokens=22)
            logits = transformer_model(data.long(),v.long(),d.long(),j.long())
            loss = loss_fn(logits.transpose(1, 2), labels.long())
            optim_schedule.zero_grad()
            loss.backward()
            optim_schedule.step_and_update_lr()
            running_loss = loss.item()
            #running_loss += loss.item()
            final = softmax(logits)[..., 1:-1]
            final = final.argmax(dim=-1) + 1
            pred_num = (labels != PAD_TOKEN_ID).sum(dim=-1)
            correct_num = ((labels != PAD_TOKEN_ID) * (final == labels)).sum(dim=-1)
            #cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
            cum_acc = torch.true_divide(correct_num, pred_num).mean().item()
            loss_save.append(round(running_loss, 4))
            auc_save.append(round(cum_acc, 4))
            tmp_index.append(int(index))
            if (index+1)%50==0:
                np.savetxt(os.path.join(loss_save_path,str(int(time.time()))+'.txt'),np.concatenate((np.array(tmp_index).reshape(-1,1),np.array(loss_save).reshape(-1,1),np.array(auc_save).reshape(-1,1)),axis=1))
                
                loss_save=[]
                auc_save=[]
                tmp_index=[]
                
        torch.save(transformer_model,model_save_path)       
        