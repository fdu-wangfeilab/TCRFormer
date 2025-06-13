import torch
from torch import nn
from torch.nn import functional as F

class get_msg(nn.Module):
    def __init__(self,input_dim = 1024,mid_dim = 20,dropout_prob=0.5,num_blocks=12,latent_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        
        self.dropout = nn.Dropout(dropout_prob)
        self.itm_head_1 = nn.Linear(input_dim*mid_dim, input_dim)
        self.itm_head_2 = nn.Linear(input_dim , latent_dim)

    def forward(self, x):
 
        result = self.itm_head_1(x)
        result = self.dropout(result)
        result = self.itm_head_2(result)

        return result

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        self.fc_out = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Reshape input
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        
        # Split the queries, keys, and values into multiple heads
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        
        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / (self.dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention to the values
        out = torch.matmul(attention, values).permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        out = out.view(batch_size, seq_len, self.dim)
        
        # Final linear layer
        out = self.fc_out(out)
        return out

class MultiLayerSelfAttention(nn.Module):
    def __init__(self, dim, num_layers=2, num_heads=8):
        super(MultiLayerSelfAttention, self).__init__()
        self.num_layers = num_layers
        self.attention_layers = nn.ModuleList([
            SelfAttention(dim, num_heads) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.attention_layers:
            x = layer(x)
        return x


class classification_model(nn.Module):
    def __init__(self,
                 tcr_dim =1024,nums=30,
                 latent_dim=256,dropout_prob=0.5,
                 num_blocks=2,num_heads=8,class_nums=8):
        super().__init__()
        
        self.get_beta = get_msg(input_dim=tcr_dim,mid_dim=nums,latent_dim=latent_dim)
        self.class_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        
        self.self_attention = MultiLayerSelfAttention(latent_dim, num_blocks, num_heads)
        
                
        self.dense = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(latent_dim,20),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(20,class_nums),
        )
    
    def forward(self, cdr3):
        bs,top_nums,_,_=cdr3.shape
        cdr3 = cdr3.view(bs,top_nums,-1)
        cdr3 = self.get_beta(cdr3)
        
        class_tokens = self.class_token.repeat(bs, 1, 1)
        x=torch.cat([class_tokens, cdr3], dim=1)
        x = self.self_attention(x)
        result = x[:,0,:].reshape(bs,-1)
        
        
        
        return self.dense(result)