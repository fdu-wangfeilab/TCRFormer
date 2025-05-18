import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class AttentionBlock(nn.Module):
    def __init__(
        self,
        feature_dim=2000,
        mlp_ratio=4.0,
        num_heads=8,
        norm_type="bn",
        affine=True,
        expand_dim=32,
        **kwargs,
    ) -> None:
        super().__init__()

        if norm_type == "bn":
            self.norm1 = nn.BatchNorm1d(feature_dim, affine=affine, eps=1e-6)
            self.norm2 = nn.BatchNorm1d(feature_dim, affine=affine, eps=1e-6)
        elif norm_type == "ln":
            self.norm1 = nn.LayerNorm(expand_dim, elementwise_affine=affine, eps=1e-6)
            self.norm2 = nn.LayerNorm(expand_dim, elementwise_affine=affine, eps=1e-6)

        self.attn = Attention(expand_dim, num_heads=num_heads, qkv_bias=True, **kwargs)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        mlp_hidden_dim = int(expand_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=expand_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
    
    def forward(self, x, c=None):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    

class classification_model(nn.Module):
    
    def __init__(self,
                 beta_input_dim = 1024,beta_mid_dim = 20,
                 beta_st_input_dim = 1024,beta_st_mid_dim = 20,
                 alpha_input_dim = 1024,alpha_mid_dim = 20,
                 alpha_st_input_dim = 1024,alpha_st_mid_dim = 20,
                 dropout_prob=0.5,num_blocks=8,latent_dim=256,emb_dim=64):
        super().__init__()
        
        self.exp_linear=nn.Linear(beta_st_input_dim, latent_dim)
        self.low_linear=nn.Linear(beta_input_dim, latent_dim)
        

        self.class_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        
        self.blks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blks.append(
                AttentionBlock(
                    feature_dim=beta_mid_dim*4+1,
                    expand_dim=latent_dim,
                    mlp_ratio=4.0,
                )
            )
        
        self.dense = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim,50),
            nn.ReLU(),
            nn.Linear(50,1),
            nn.Sigmoid()
        )
        
    def forward(self, beta,beta_st,alpha,alpha_st):
        
        beta=self.low_linear(beta)
        alpha=self.low_linear(alpha)
        
        beta_st=self.exp_linear(beta_st)
        alpha_st=self.exp_linear(alpha_st)
        
        
        class_tokens = self.class_token.repeat(beta.shape[0], 1, 1)
        x=torch.cat([class_tokens, beta,beta_st,alpha,alpha_st], dim=1)
        
        for blk in self.blks:
            x = blk(x)

        result = x[:,0,:].reshape(beta.shape[0],-1)
        
        return self.dense(result)