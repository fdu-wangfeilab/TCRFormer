import torch
from torch import nn
from torch.nn import functional as F

class classification_model(nn.Module):
    def __init__(self, tcr_dim=1024, pep_dim=1024, dim_hidden=256, layers_inter=2, dim_seqlevel=256, inter_type='mul'):
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