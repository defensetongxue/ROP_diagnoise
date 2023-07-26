from einops import rearrange,repeat
import torch.nn.functional as F

import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size,  embed_dim, depth, heads, mlp_dim,  channels = 3, dim_head = 12, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.patch_length=int(image_size/patch_size)
        num_patches = int(self.patch_length ** 2)
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.patch_embedding=PatchEmbedding(in_channels=channels,patch_size=patch_size,embed_dim= embed_dim )
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches ,embed_dim))
        self.patch_to_embedding = nn.Linear(patch_dim, embed_dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, img, mask = None):
        x=self.patch_embedding(img)

        # x += self.pos_embedding # [batch_size, num_patches, embed_dim]
        x = self.dropout(x) 

        x = self.transformer(x, mask) # [batch_size, num_patches, embed_dim]


        x = self.mlp_head(x)
        x=x.reshape(-1,self.patch_length,self.patch_length)
        return x
    
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)  # shape = [batch_size, embed_dim, num_patches ** 0.5, num_patches ** 0.5]
        x = x.flatten(2)  # shape = [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # shape = [batch_size, num_patches, embed_dim]
        return x
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        # dim=128,depth=12，heads=8，dim_head=64,mlp_dim=128
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads # 64 x 8
        self.heads = heads # 8
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads 
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
    # dim=128,hidden_dim=128
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)