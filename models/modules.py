import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from .build import MODELS
from utils import misc
from knn_cuda import KNN
from .serialization import Point
from utils.logger import *
import random
import math

class Token_Embed(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        if in_c == 3:
            self.first_conv = nn.Sequential(
                nn.Conv1d(in_c, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 256, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, out_c, 1)
            )

        else:
            self.first_conv = nn.Sequential(
                nn.Conv1d(in_c, in_c, 1),
                nn.BatchNorm1d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_c, in_c, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(in_c * 2, out_c, 1),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_c, out_c, 1)
            )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.out_c)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # mask = mask * float('-inf') 
            mask = mask * - 100000.0
            attn = attn + mask.unsqueeze(1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        
    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos, rec_layer=[12], shuffle=False):
        if shuffle:
            perm = torch.randperm(x.size(1))
            x = x[:, perm]
            pos = pos[:, perm]
        
        ret = []
        for i, block in enumerate(self.blocks):
            if i+1 in rec_layer:
                x = block(x + pos)
                ret.append(x)
            else:
                x = block(x + pos)

        if shuffle:
            for i in range(len(ret)):
                ret[i] = ret[i][:, perm]

        return ret[0] if len(ret) == 1 else ret


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

    def forward(self, x, pos, return_token_num, shuffle=False):
        if shuffle:
            visible_token_num = x.size(1)-return_token_num
            perm = torch.randperm(visible_token_num)
            x[:, :visible_token_num] = x[:, :visible_token_num][:, perm]
            pos[:, :visible_token_num] = pos[:, :visible_token_num][:, perm]

        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel

        return x

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class PatchDivider(nn.Module):
    def __init__(self, patch_size, grid_size=0.02, order=["hilbert-xyz"]):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.order = order

    def serialization(self, pts):
        scaled_coord = pts / self.grid_size
        grid_coord = torch.floor(scaled_coord).to(torch.int64)
        min_coord = grid_coord.min(dim=1, keepdim=True)[0]
        grid_coord = grid_coord - min_coord

        batch_idx = torch.arange(0, pts.shape[0], 1.0).unsqueeze(1).repeat(1, pts.shape[1]).to(torch.int64).to(pts.device)

        point_dict = {'batch': batch_idx.flatten(), 'grid_coord': grid_coord.flatten(0, 1)}
        point_dict = Point(**point_dict)

        if isinstance(self.order, list):
            order = [random.choice(self.order)]
        else:
            order = [self.order]
        
        point_dict.serialization(order=order)

        orders = point_dict.serialized_order
        inverses = point_dict.serialized_inverse

        return orders, inverses
    
    def find_closest_point_to_center(self, patches):
        '''
            patches: B L P 3
            -----------------
            output: B L 3
        '''
        # Calculate the center of each patch
        centers = patches.mean(dim=2)  # B L 3

        # Calculate the distance of each point to the center
        distances = torch.norm(patches - centers.unsqueeze(2), dim=3)  # B L P

        # Find the index of the closest point to the center for each patch
        closest_point_indices = distances.argmin(dim=2)  # B L

        # Gather the closest points using the indices
        B, L, P, _ = patches.size()
        closest_points = patches[torch.arange(B).unsqueeze(1), torch.arange(L).unsqueeze(0), closest_point_indices]  # B L 3
        return closest_points

    def forward(self, pts, virtual_center=True):
        '''
            pts: B N 3
            -----------------
            output: B N 3
        '''
        B, N, _ = pts.shape
        orders, _ = self.serialization(pts)
        reordered_pts = pts.flatten(0, 1)[orders].reshape(B, N, -1).contiguous()
        patches = torch.stack(torch.split(reordered_pts, self.patch_size, dim=1), dim=1)

        B, L, P, _ = patches.shape
        if virtual_center:
            centers = torch.mean(patches, dim=2)
        else:
            centers = self.find_closest_point_to_center(patches)
        centers = centers.reshape(B, L, 3)
        patches = patches - centers.unsqueeze(2)
        
        return patches, centers