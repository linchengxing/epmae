import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from .build import MODELS
import random
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from .modules import *

class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.mask_ratio 
        self.encoder_depth = config.encoder_depth
        self.drop_path_rate = config.drop_path_rate
        self.encoder_num_heads = config.encoder_num_heads 
        # embedding
        self.encoder_dim =  config.encoder_dim
        
        self.shuffle = config.shuffle

        self.base_encoder = Token_Embed(in_c=3, out_c=self.encoder_dim)

        self.momentum_target = config.get('momentum_target', False)
        if self.momentum_target:
            self.momentum_encoder = Token_Embed(in_c=3, out_c=self.encoder_dim)
            for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
                param_m.data.copy_(param_b.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

        self.mask_type = config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.encoder_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.encoder_depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.encoder_dim,
            depth = self.encoder_depth,
            drop_path_rate = dpr,
            num_heads = self.encoder_num_heads,
        )

        self.rec_layer = config.rec_layer
        self.decoder_dim = config.decoder_dim
        self.proj_layer = nn.ModuleList([
            nn.Linear(self.encoder_dim, self.decoder_dim)
            for i in range(len(self.rec_layer)) 
        ])
        self.fuse_layer = nn.Linear(len(self.rec_layer), 1, bias=False)

        self.norm = nn.LayerNorm(self.encoder_dim)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def forward(self, neighborhood, center, eval = False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = eval) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug = eval)

        batch_size, _, group_size, _ = neighborhood.size()

        x_vis = self.base_encoder(neighborhood[~bool_masked_pos].reshape(batch_size, -1, group_size, 3))  #  B G C

        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        if eval:
            x_vis = self.blocks(x_vis, pos, shuffle=self.shuffle)
            x_vis = self.norm(x_vis)
        else:
            x_vis = self.blocks(x_vis, pos, rec_layer=self.rec_layer, shuffle=self.shuffle)
            if len(self.rec_layer) == 1:
                x_vis = self.norm(x_vis)
            else:
                x_vis = [self.proj_layer[i](self.norm(x_vis[i])) for i in range(len(x_vis))]
                x_vis = torch.stack(x_vis, dim=-1)
                w = F.softmax(self.fuse_layer.weight.squeeze(0), dim=0)
                x_vis = torch.sum(w * x_vis, dim=-1)

        return x_vis, bool_masked_pos
            

@MODELS.register_module()
class Point_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger ='Point_MAE')
        self.config = config
        self.MAE_encoder = MaskTransformer(config)

        self.drop_path_rate = config.drop_path_rate

        self.encoder_dim = config.encoder_dim
        self.decoder_dim = config.decoder_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        trunc_normal_(self.mask_token, std=.02)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.decoder_dim)
        )

        self.decoder_depth = config.decoder_depth
        self.decoder_num_heads = config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.decoder_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        self.shuffle = config.shuffle
        self.patch_size = config.patch_size
        self.grid_size = config.grid_size
        self.order = config.order
        self.patch_divider = PatchDivider(patch_size=self.patch_size, grid_size=self.grid_size, order=self.order)

        self.search_size = config.patch_selection.search_size
        self.pred_ratio = config.patch_selection.pred_ratio

        # prediction head
        self.momentum_target = config.get('momentum_target', False)
        if self.momentum_target:
            self.predictor = nn.Linear(self.decoder_dim, self.encoder_dim)
        else:
            self.predictor = nn.Conv1d(self.decoder_dim, 3*self.patch_size, 1)

        # loss
        self.loss_type = config.loss_type
        if self.loss_type == 'smoothl1':
            self.beta = config.beta
        self.build_loss_func()
        self.apply(self._init_weights)

    def build_loss_func(self):
        if self.loss_type == 'smoothl1':
            self.loss_func = nn.SmoothL1Loss(beta=self.beta)
        elif self.loss_type == 'l2' or self.loss_type == 'mse':
            self.loss_func = nn.MSELoss()
        elif self.loss_type == 'l1':
            self.loss_func = nn.L1Loss()
        elif self.loss_type == 'cdl1':
            self.loss_func = ChamferDistanceL1().cuda()
        elif self.loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()

    def get_gt_mask(self, mask):
        # Create the convolution kernel
        kernel_size = 2 * self.search_size + 1
        kernel = torch.ones((1, 1, kernel_size), dtype=torch.float32, device=mask.device)
        mask_float = mask.unsqueeze(1).float()  # B N 1
        counts = F.conv1d(mask_float, kernel, padding=self.search_size)
        
        center_counts = counts.squeeze(1) - mask_float.squeeze(1)
        center_counts = center_counts + 1 # to distinguish from the visible position
        center_counts = center_counts * mask.bool()
        
        return center_counts.int()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, eval = False):
        neighborhood, center = self.patch_divider(pts)

        _, N1, _ = center.shape

        x_vis, mask = self.MAE_encoder(neighborhood, center, eval=eval)
        _, N2, _ = x_vis.shape

        if eval:
            return x_vis.mean(1) + x_vis.max(1)[0]
        
        N = N1 - N2
        gt_mask = self.get_gt_mask(mask)
        gt_mask_index = torch.argsort(gt_mask, dim=1, descending=True)[:, :N]
        end_index = int(N * self.pred_ratio)
        gt_mask_index = gt_mask_index[:, :end_index]
        B, N, _, _ = neighborhood.size()
        gt_mask_final = torch.zeros((B, N), dtype=torch.bool, device=gt_mask_index.device)
        gt_mask_final.scatter_(1, gt_mask_index, True)

        B,_,C = x_vis.shape # B VIS C
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[gt_mask_final]).reshape(B, -1, C)

        _,N,_ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N, shuffle=self.shuffle)

        if self.momentum_target:
            B, _, C = x_rec.shape
            rbds = self.predictor(x_rec).reshape(-1, C)
            B, _, N, _ = neighborhood.size()
            with torch.no_grad():
                gts = self.MAE_encoder.momentum_encoder(neighborhood[gt_mask_final].reshape(B, -1, N, 3)).reshape(-1, C)
        else:
            B, M, C = x_rec.shape
            rbds = self.predictor(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B*M, -1, 3)
            gts = neighborhood[gt_mask_final].reshape(B*M, -1, 3)

        loss1 = self.loss_func(rbds, gts.detach())
        return loss1

# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.encoder_dim = config.encoder_dim
        self.drop_path_rate = config.drop_path_rate
        
        self.patch_size = config.patch_size
        self.group_method = config.group_method
        if self.group_method == "KNN":
            self.sequence_length = config.sequence_length
            self.patch_divider = Group(num_group=self.sequence_length, group_size=self.patch_size)
        elif self.group_method == "Serialization":
            self.grid_size = config.grid_size
            self.order = config.order
            self.patch_divider = PatchDivider(patch_size=self.patch_size, grid_size=self.grid_size, order=self.order)
        else:
            raise NotImplementedError

        self.base_encoder = Token_Embed(in_c=3, out_c=self.encoder_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.encoder_dim)
        )

        self.pooling_type = config.pooling_type
        self.use_cls_token = config.use_cls_token
        self.use_max_pooling = config.use_max_pooling
        self.use_mean_pooling = config.use_mean_pooling

        if self.pooling_type == 'sum':
            HEAD_CHAANNEL = 1
        elif self.pooling_type == 'concat':
            HEAD_CHAANNEL = 0
            if self.use_cls_token:
                HEAD_CHAANNEL += 1
            if self.use_max_pooling:
                HEAD_CHAANNEL += 1
            if self.use_mean_pooling:
                HEAD_CHAANNEL += 1
        else:
            raise NotImplementedError

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.encoder_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.encoder_depth = config.encoder_depth
        self.encoder_num_heads = config.encoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.encoder_depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.encoder_dim,
            depth=self.encoder_depth,
            drop_path_rate=dpr,
            num_heads=self.encoder_num_heads,
        )

        self.norm = nn.LayerNorm(self.encoder_dim)

        self.cls_dim = config.cls_dim
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.encoder_dim * HEAD_CHAANNEL, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        if config.get('smooth') is None:
            self.loss_ce = nn.CrossEntropyLoss()
        else:
            self.loss_ce = nn.CrossEntropyLoss(label_smoothing=config.smooth)

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path, logger):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger=logger)
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger=logger
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger=logger)
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger=logger
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger=logger)
        else:
            print_log('Training from scratch!!!', logger=logger)
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def pooling(self, x):
        features = []

        if self.use_cls_token:
            features.append(x[:, 0])
            x = x[:, 1:]

        if self.use_max_pooling:
            features.append(x.max(1)[0])
        
        if self.use_mean_pooling:
            features.append(x.mean(1))

        return torch.cat(features, dim=-1) if self.pooling_type == 'concat' else sum(features)

    def forward(self, pts):
        neighborhood, center = self.patch_divider(pts)

        group_input_tokens = self.base_encoder(neighborhood)  # B G N
        pos = self.pos_embed(center)

        if self.use_cls_token:
            cls_token = self.cls_token.expand(center.size(0), -1, -1)
            cls_pos = self.cls_pos.expand(center.size(0), -1, -1)
            group_input_tokens = torch.cat([cls_token, group_input_tokens], dim=1)
            pos = torch.cat([cls_pos, pos], dim=1)

        # transformer
        x = self.blocks(group_input_tokens, pos)
        x = self.norm(x)
        feat = self.pooling(x)
        ret = self.cls_head_finetune(feat)
        return ret
    