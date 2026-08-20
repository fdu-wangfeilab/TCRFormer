import scanpy as sc
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import TensorDataset, DataLoader


class classification_model(nn.Module):
    def __init__(self, tcr_dim=1024, pep_dim=1024, dim_hidden=256, layers_inter=2, dim_seqlevel=256):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.layers_inter = layers_inter
        self.dim_seqlevel = dim_seqlevel

        self.cdr3_beta_linear = nn.Linear(tcr_dim, dim_hidden)
        self.cdr3_alpha_linear = nn.Linear(tcr_dim, dim_hidden)
        self.pep_linear = nn.Linear(pep_dim, dim_hidden)

        self.gate_conv = nn.Conv2d(dim_hidden*2, dim_hidden, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.inter_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_hidden),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_hidden),
                nn.ReLU(),
            )
        ])

        self.seqlevel_outlyer = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(dim_seqlevel, 1),
            nn.Sigmoid()
        )

    def forward(self, cdr3_emb_beta, cdr3_emb_alpha, epi_emb, addition=None):
        cdr3_beta_emb = self.cdr3_beta_linear(cdr3_emb_beta)
        cdr3_beta_feat = cdr3_beta_emb.transpose(1, 2)
        
        cdr3_alpha_emb = self.cdr3_alpha_linear(cdr3_emb_alpha)
        cdr3_alpha_feat = cdr3_alpha_emb.transpose(1, 2)
        
        epi_emb = self.pep_linear(epi_emb)
        epi_feat = epi_emb.transpose(1, 2)
        len_epi = epi_emb.shape[1]
        
        cdr3_beta_feat_mat = cdr3_beta_feat.unsqueeze(3).repeat([1, 1, 1, len_epi])
        cdr3_alpha_feat_mat = cdr3_alpha_feat.unsqueeze(3).repeat([1, 1, 1, len_epi])
        
        combined = torch.cat([cdr3_beta_feat_mat, cdr3_alpha_feat_mat], dim=1)
        gate = self.sigmoid(self.gate_conv(combined))
        fused_feat_mat = gate * cdr3_beta_feat_mat + (1 - gate) * cdr3_alpha_feat_mat

        epi_feat_mat = epi_feat.unsqueeze(2).repeat([1, 1, fused_feat_mat.shape[2], 1])

        inter_map = fused_feat_mat * epi_feat_mat
        
        for i in range(self.layers_inter):
            inter_map = self.inter_layers[i](inter_map)
            

        seqlevel_out = self.seqlevel_outlyer(inter_map)
        
        return seqlevel_out

def get_result(labels,predictions):    
    return roc_auc_score(labels,predictions)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

import pandas as pd
import numpy as np
import os
import random

def load_embeddings_and_labels(csv_path, base_dir='./train/', sample_size=None, negative_ratio=1, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)

    df = pd.read_csv(csv_path)
    for col in ['Va', 'Ja', 'Vb', 'Jb']:
        df[col] = df[col].astype(str)
        

    tcr_cols = ['CDR3a_extended', 'Va', 'Ja', 'CDR3b_extended', 'Vb', 'Jb']
    unique_tcrs = df[tcr_cols].drop_duplicates().values.tolist()
    tcr_pool = [tuple(x) for x in unique_tcrs]

    positive_pairs = set()
    for _, row in df.iterrows():
        pos_tuple = (row['Peptide'], row['CDR3a_extended'], row['Va'], row['Ja'], 
                     row['CDR3b_extended'], row['Vb'], row['Jb'])
        positive_pairs.add(pos_tuple)

    if sample_size is not None and sample_size < len(df):
        process_df = df.sample(n=sample_size, random_state=random_seed)
    else:
        process_df = df

    ep_embs, alpha_embs, beta_embs, labels = [], [], [], []


    def get_alpha_emb(cdr3a, va, ja):
        filename = f"{cdr3a}_{va.replace('/', '')}{ja}.npy"
        return np.load(os.path.join(base_dir, 'alpha', filename))

    def get_beta_emb(cdr3b, vb, jb):
        filename = f"{cdr3b}_{vb.replace('/', '')}{jb}.npy"
        return np.load(os.path.join(base_dir, 'beta', filename))

    def get_ep_emb(peptide):
        filename = f"{peptide}.npy"
        return np.load(os.path.join(base_dir, 'ep', filename))

    for _, row in process_df.iterrows():
        peptide = row['Peptide']
        cdr3a, va, ja = row['CDR3a_extended'], row['Va'], row['Ja']
        cdr3b, vb, jb = row['CDR3b_extended'], row['Vb'], row['Jb']


        try:
            cur_ep_emb = get_ep_emb(peptide)
            cur_a_emb = get_alpha_emb(cdr3a, va, ja)
            cur_b_emb = get_beta_emb(cdr3b, vb, jb)
            
            ep_embs.append(cur_ep_emb)
            alpha_embs.append(cur_a_emb)
            beta_embs.append(cur_b_emb)
            labels.append(1)
        except FileNotFoundError as e:

            continue 


        negative_count = 0
 
        max_attempts = negative_ratio * 10 
        attempts = 0

        while negative_count < negative_ratio and attempts < max_attempts:
            attempts += 1

            sampled_tcr = random.choice(tcr_pool)
            s_cdr3a, s_va, s_ja, s_cdr3b, s_vb, s_jb = sampled_tcr


            sampled_pair = (peptide, s_cdr3a, s_va, s_ja, s_cdr3b, s_vb, s_jb)
            if sampled_pair in positive_pairs:
                continue

            try:
                s_a_emb = get_alpha_emb(s_cdr3a, s_va, s_ja)
                s_b_emb = get_beta_emb(s_cdr3b, s_vb, s_jb)

                ep_embs.append(cur_ep_emb) 
                alpha_embs.append(s_a_emb) 
                beta_embs.append(s_b_emb)  
                labels.append(0)

                negative_count += 1
            except FileNotFoundError:
                continue

    ep_embs = np.stack(ep_embs)
    alpha_embs = np.stack(alpha_embs)
    beta_embs = np.stack(beta_embs)
    labels = np.array(labels, dtype=np.int32)

    return ep_embs, beta_embs, alpha_embs, labels


def run(train_csv = '/mnt/sdc/tyh/TCRFormer/IMM25/train_v1.0.csv', base_dir = '/mnt/sdc/tyh/TCRFormer/IMM25/train_emb/', sample_size = 10000, tcr_dim = 1024, pep_dim=1024, device='cuda:0', BATCH_SIZE= 1536, lr=0.001, weight_decay=0.1, EPOCH = 20, seed = 1):
    
    set_seed(seed)

    model=classification_model(tcr_dim=tcr_dim, pep_dim=pep_dim, 
                           )
    model=model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(EPOCH):
        model.train()
        ep_train_emb,beta_train_emb,alpha_train_emb,train_labels = load_embeddings_and_labels(csv_path = train_csv, base_dir= base_dir, sample_size = sample_size, random_seed=epoch)
    
        dataset = sx_Dataset(beta_train_emb,alpha_train_emb,ep_train_emb,train_labels)
    
        train_dataloader=DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,drop_last=True)
    
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
            torch.save(model,os.path.dirname(train_csv)+'/model.pt')
            
