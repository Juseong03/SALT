import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from torch.autograd import Function
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)
    
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)
    
class SelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        heads = 8,
        dim_head = 16,
    ):        
        super(SelfAttention, self).__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(emb_dim, inner_dim * 3, bias = False)
        
        self.to_out = nn.Linear(inner_dim, emb_dim)
        

    def forward(self, x, share=None):
        b, n, d = x.shape
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        
        energy = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        if share != None:
            energy = energy + share
            
        attention = torch.softmax(energy, dim=-1)
        
#         if share != None:
#             attention = (attention + share)/2
            
        out = einsum('b h i j, b h j d -> b h i d', attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)

        return self.to_out(out), energy


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_size, 
        emb_dim,
        heads,
        dim_head,
        dropout,
        forward_expansion = 4,
    ):

        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(emb_dim, heads, dim_head)
        self.feed_forward  = FeedForward(emb_dim, dropout = dropout)
        
        self.attention2 = SelfAttention(input_size, heads, dim_head)
        self.feed_forward2  = FeedForward(input_size, dropout = dropout)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(input_size)
        self.norm4 = nn.LayerNorm(input_size)
        
    def forward(self, x, share=None, share2 = None):
        
        attention, share = self.attention(x, share)
        x = attention + x
        x = self.norm1(x)
        
        forward = self.feed_forward(x)
        x = forward + x
        x = self.norm2(x)
        
        
        x = x.transpose(-2, -1)
        attention, share2 = self.attention2(x, share2)
        x = attention + x
        x = self.norm3(x)
        
        forward = self.feed_forward2(x)
        x = forward + x
        x = self.norm4(x)
        
        x = x.transpose(-2, -1)
        
        return x, share, share2
    

class Attention(nn.Module):
    def __init__(
        self, dim, inner_dim, out_dim, heads
    ):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.scale = inner_dim ** -0.5
        
        self.to_out = nn.Linear(inner_dim, out_dim)
        
    def forward(self, x, share=None):
        
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        energy = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        b, i, j = energy.shape
        energy = energy.reshape(b, 1, i, j)
        energy = energy.repeat(1, self.heads, 1, 1)    
        
        if share != None :    
            energy = energy + share
            
        attention = torch.softmax(energy, dim=-1)
        
#         if share != None :
#             attention = (attention + share)/2
            
        b, n, d = v.shape
        v = v.reshape(b, 1, n, d)
        v = v.repeat(1, self.heads, 1, 1)

        out = einsum('b h i j, b h j d -> b h i d', attention, v)

        out = self.to_out(out)
        return out, energy



class LinearBlock(nn.Module):
    def __init__(
        self,
        input_size,
        emb_dim,
        head,
        dim_head,
        forward_expansion,
        dropout,
        gate_act,
    ):
        super(LinearBlock, self).__init__()
                
        self.feature_wise = Linear_Layer(    
            dim = emb_dim,
            dim_head = dim_head,
            heads = head,
            dropout = dropout,
            gate_act = gate_act
        )
        
        self.dimension_wise = Linear_Layer(    
            dim = input_size,
            dim_head = dim_head,
            heads = head,
            dropout = dropout,
            gate_act = gate_act
        )
        
    def forward(self, x, share=None, share2 = None):        
        
        
        x, share = self.feature_wise(x, share)
        
        x, share2 = self.dimension_wise(x.transpose(-2, -1), share2)
        x = x.transpose(-2, -1)
        
        return x, share, share2

class Linear_Layer(torch.nn.Module):
    def __init__(self, 
                 dim,
                 dim_head,
                 heads,
                 dropout = 0.1,
                 gate_act = None,
                ):
        
        super(Linear_Layer, self).__init__()
        self.heads = heads
        
        self.norm = nn.LayerNorm(dim)
        self.attention = Attention(dim, dim_head, dim_head, heads)

        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_head * heads * 2),
            nn.GELU()
        )
        
        self.gating = GatingUnit(
            heads = heads,
            dim = dim_head,
            gate_act = gate_act,
        )
                
        self.proj_out = nn.Sequential(
            nn.Linear(dim_head * heads, dim),
        )

    def forward(self, x, share = None):
        
        attn, share = self.attention(x, share)
        norm_x = self.norm(x)
        out = self.proj_in(norm_x)
        
        out = rearrange(out, 'b n (h d) -> b h n d', h = self.heads)
        out = self.gating(out, attn)
        out = rearrange(out, 'b h n d -> b n (h d)', h = self.heads)
        
        out = self.proj_out(out)
        x = out + x
        
        return x, share
        
class GatingUnit(nn.Module):
    def __init__(self, heads, dim, gate_act = None):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Conv2d(heads,heads, 1)
        nn.init.constant_(self.gate.bias, 1.0)
        nn.init.constant_(self.gate.weight, 0.0)
        
        self.act = nn.Identity() if gate_act is None else gate_act
        
    def forward(self, x, attn = None):

        res, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)
        gate = self.gate(gate)
        
        return self.act(gate + attn) * res


class SALT(nn.Module):
    def __init__(
        self,
        input_size,
        emb_dim,
        depth,
        heads,
        dim_head,
        dropout,
        forward_expansion,
        gate_act,
        share = 'share',
        share_mode = 'avg',
        output_mode = 'mul'
    ):
        super(SALT, self).__init__()
        self.depth = depth
        self.Transformer = nn.ModuleList(
            [
                TransformerBlock(
                    input_size = input_size,
                    emb_dim=emb_dim,
                    heads=heads,
                    dim_head = dim_head,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(depth)
            ]
        )

        self.Linears = nn.ModuleList(
            [
                LinearBlock(
                    input_size=input_size,
                    emb_dim=emb_dim,
                    head = heads,
                    dim_head = dim_head,
                    forward_expansion=forward_expansion,
                    dropout = dropout,
                    gate_act = gate_act,
                )        
                for _ in range(depth)
            ]
        )
        
        self.share = share
        self.share_mode = share_mode
        self.output_mode = output_mode
        self.scale1 = input_size ** -0.5
        self.scale2 = emb_dim ** -0.5
        if self.share_mode == 'layer' :
            self.share1_layer = nn.Linear(input_size * 2, input_size)
            self.share2_layer = nn.Linear(emb_dim * 2, emb_dim)
        else : 
            self.share1_layer = nn.Linear(input_size, input_size)
            self.share2_layer = nn.Linear(emb_dim, emb_dim)
            
        if self.output_mode == 'cat' :
            self.output_layer = nn.Linear(emb_dim * 2, emb_dim)
        else : 
            self.output_layer = nn.Linear(emb_dim , emb_dim)
            
    def forward(self, x_categ, x_cont = None):
        if x_cont is None : 
            x = x_categ
        else : 
            x = torch.cat((x_categ, x_cont), dim = 1)
        
        x_att = x.clone()
        x_mlp = x.clone()
        share = None
        share2 = None
        # Depth
        for layer in range(0, self.depth):
            if self.share == 'share':
                x_att, share_mlp, share_mlp2 = self.Transformer[layer](x_att, share, share2)
                x_mlp, share_att, share_att2 = self.Linears[layer](x_mlp, share, share2)
                if self.share_mode == 'avg' :
                    share = (share_att + share_mlp)/2
                    share2 = (share_att2 + share_mlp2)/2
                
                elif self.share_mode == 'avg_linear' :
                    share = (share_att + share_mlp)/2
                    share2 = (share_att2 + share_mlp2)/2
                    share = self.share1_layer(share)
                    share2 = self.share2_layer(share2)
                    
                elif self.share_mode == 'add' :
                    share = (share_att + share_mlp)
                    share2 = (share_att2 + share_mlp2)
                
                elif self.share_mode == 'add_linear' :
                    share = (share_att + share_mlp)
                    share2 = (share_att2 + share_mlp2)
                    share = self.share1_layer(share)
                    share2 = self.share2_layer(share2)
                
                elif self.share_mode == 'mul' :
                    share = (share_att * share_mlp) * self.scale1
                    share2 = (share_att2 * share_mlp2) * self.scale2
                
                elif self.share_mode == 'mul_linear' :
                    share = (share_att * share_mlp) * self.scale1
                    share2 = (share_att2 * share_mlp2) * self.scale2
                    share = self.share1_layer(share)
                    share2 = self.share2_layer(share2)
                
                elif self.share_mode == 'matmul' :
                    share = torch.matmul(share_att, share_mlp) * self.scale1
                    share2 = torch.matmul(share_att2, share_mlp2) * self.scale2
                
                elif self.share_mode == 'layer' :
                    share = torch.cat((share_att, share_mlp), dim = -1)
                    share2 = torch.cat((share_att2, share_mlp2), dim = -1)
                    share = self.share1_layer(share)
                    share2 = self.share2_layer(share2)
            else :
                x_att, share_mlp, share_mlp2 = self.Transformer[layer](x_att)
                x_mlp, share_att, share_att2 = self.Linears[layer](x_mlp)
                

        if self.output_mode == 'mul' :
            out = torch.mul(x_att, x_mlp)
        elif self.output_mode == 'add' :
            out = torch.add(x_att, x_mlp)
        elif self.output_mode == 'cat' :
            out = torch.cat((x_att, x_mlp), dim = -1)
            out = self.output_layer(out)
        elif self.output_mode == 'mul_linear' :
            out = torch.mul(x_att, x_mlp)
            out = self.output_layer(out)
        elif self.output_mode == 'add_linear' :
            out = torch.add(x_att, x_mlp)
            out = self.output_layer(out)
        
        return out