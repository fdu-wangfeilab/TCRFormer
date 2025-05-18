import torch
from torch import nn
from torch.nn import functional as F

class classification_model(nn.Module):
    def __init__(self,tcr_dim=1024,pep_dim=1024,dim_hidden=256,layers_inter=2,dim_seqlevel=256,inter_type='mul',):
        self.dim_hidden = dim_hidden 
        self.layers_inter = layers_inter 
        self.dim_seqlevel = dim_seqlevel 
        super().__init__()

        self.cdr3_linear = nn.Linear(tcr_dim, dim_hidden)
        self.pep_linear = nn.Linear(pep_dim, dim_hidden)

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


    def forward(self, cdr3_emb,epi_emb, addition=None):
        len_cdr3, len_epi = cdr3_emb.shape[1], epi_emb.shape[1]
        
        cdr3_emb=self.cdr3_linear(cdr3_emb)
        epi_emb=self.pep_linear(epi_emb)

        cdr3_feat = cdr3_emb.transpose(1, 2)
        epi_feat = epi_emb.transpose(1, 2)

        cdr3_feat_mat = cdr3_feat.unsqueeze(3).repeat([1, 1, 1, len_epi])  # batch_size, dim_hidden, len_cdr3, len_epi
        epi_feat_mat = epi_feat.unsqueeze(2).repeat([1, 1, len_cdr3, 1])  # batch_size, dim_hidden, len_cdr3, len_epi

        inter_map = cdr3_feat_mat * epi_feat_mat
            
        for i in range(self.layers_inter):
            inter_map = self.inter_layers[i](inter_map)

        seqlevel_out = self.seqlevel_outlyer(inter_map)
        
        return seqlevel_out