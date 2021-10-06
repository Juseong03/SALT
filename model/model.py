import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
from einops import rearrange
from .salt import SALT

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ff_encodings(x,B):
    x_proj = (2. * np.pi * x.unsqueeze(-1)) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


#mlp
class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class EmbNN(nn.Module):
    def __init__(self,dims,):
        super(EmbNN, self).__init__()

        self.fc = nn.Linear(dims[0], dims[1], bias = False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dims[1], dims[1], bias = False)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.act(x)
        x = self.fc2(x) + x 
        x = self.softmax(x)
        
        return x


# main class
class Model(nn.Module):
    def __init__(
        self,
        *,
        cat_idxs,
        con_idxs,
        cat_dim,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        gmlp_act = None,
        num_special_tokens = 1,
        
        ff_dropout = 0.,
        lastmlp_dropout = 0.,
        cont_embeddings = 'MLP',
        scalingfactor = 10,
        attentiontype = 'col',
        share_mode = 'avg',
        output_mode = 'mul',
        share = True    
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens        
        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(cat_dim)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        self.dim = dim
        self.dim_head = dim_head
        
        self.emb_dim = emb_dim
        self.cat_dim = cat_dim
        self.cat_idxs = cat_idxs
        self.con_idxs = con_idxs
        
        self.nfeats = self.num_categories + num_continuous

        h = self.total_tokens + int(5  * len(self.con_idxs))
        self.embedding = nn.Embedding(h, emb_dim)

        # Continuous Embeddings
        self.Con_Embedding = nn.ModuleList([])
        for _ in range(len(self.con_idxs)):
            self.Con_Embedding.append(
                ConEmbedding(
                    emb_dim=int(emb_dim),
                    dis_dim=h,
                ).to(DEVICE)
            )
        
        self.salt = SALT(
            input_size = self.nfeats + 1,
            emb_dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads = heads,
            forward_expansion=4,
            dropout=ff_dropout,
            gate_act = None,
            share = share,
        )
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Linear(emb_dim * 2, dim_out),
        )
    def forward(self, x):
        
        # categorical
        cat_x = x[:, self.cat_idxs]
        cat_x = cat_x.type(torch.long)

        # Continuous
        con_x = x[:, self.con_idxs]
        con_x = con_x.type(torch.float)
        con_x_T = con_x.transpose(1, 0)

        cat = 0
        con = 0
        x_emb = []
        for i in range(self.nfeats):
            if i in self.cat_idxs:
                cat_tmp = cat_x[:,cat] + sum(self.cat_dim[:cat])
                x_emb.append(self.embedding(cat_tmp))
                cat += 1
            elif i in self.con_idxs:
                con_tmp = con_x_T[con]
                con_tmp = con_tmp.reshape(len(con_tmp), 1)
                con_tmp  = self.Con_Embedding[con](con_tmp, mode="embedding")
                con_tmp = torch.matmul(con_tmp, self.embedding.weight)
                x_emb.append(con_tmp)
                con += 1
        
        x_cls = torch.zeros(len(x),1).type(torch.long)
        x_emb.append(self.embedding(x_cls).flatten(1))
        x_emb = torch.stack(x_emb, dim=1)
        
        # SALT
        output = self.salt(x_emb)

        return self.mlp(output[:,-1,:].flatten(1))

