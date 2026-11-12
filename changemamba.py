from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

#from utils import misc
try:
    import misc
except Exception:
    misc = None
try:
    from checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
except Exception:
    get_missing_parameters_message = get_unexpected_parameters_message = None
try:
    from logger import *
except Exception:
    def print_log(*args, **kwargs):
        print(*args)
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

try:
    from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
except Exception:
    ChamferDistanceL1 = ChamferDistanceL2 = None
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception:
    Mamba = None
from block_scan import Block

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

try:
    from knn_cuda import KNN
except Exception:
    KNN = None
try:
    from torch_cluster import knn as torch_cluster_knn
except Exception:
    torch_cluster_knn = None
try:
    from build import MODELS
except Exception:
    MODELS = None
try:
    from serialization import Point
except Exception:
    Point = None
import os
import sys
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception:
    Mamba = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
import os.path as osp
from datetime import datetime
import argparse
import pickle

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR
try:
    import torch_geometric
except Exception:
    torch_geometric = None
try:
    from change_dataset import ChangeDataset, MyDataLoader, ChangeDataset_synthetic
except Exception:
    ChangeDataset = MyDataLoader = ChangeDataset_synthetic = None
try:
    from transforms import NormalizeScale, SamplePoints
except Exception:
    NormalizeScale = SamplePoints = None
try:
    from metric import ConfusionMatrix
except Exception:
    ConfusionMatrix = None
try:
    from imbalanced_sampler import ImbalancedDatasetSampler
except Exception:
    ImbalancedDatasetSampler = None
try:
    from pointnet2 import MLP
except Exception:
    MLP = None

try:
    from torch_geometric.nn import DynamicEdgeConv, global_max_pool, global_mean_pool, avg_pool_x
    from torch_geometric.utils import to_dense_batch
except Exception:
    DynamicEdgeConv = global_max_pool = global_mean_pool = avg_pool_x = None
    to_dense_batch = None

try:
    from utils import ktprint, set_logger, check_dirs
except Exception:
    ktprint = set_logger = check_dirs = None

import torch.nn as nn
import numpy as np



NUM_CLASS = 5


def _patch_pointcept_hilbert_for_old_torch():
    """Pointcept Hilbert serialization uses dtype-view unsupported by torch 1.10."""
    try:
        import importlib

        hilbert_module = importlib.import_module(
            "pointcept.models.utils.serialization.hilbert"
        )
        default_module = importlib.import_module(
            "pointcept.models.utils.serialization.default"
        )
    except Exception:
        return
    if getattr(hilbert_module, "_codex_old_torch_patch", False):
        return

    def encode_compat(locs, num_dims, num_bits):
        orig_shape = locs.shape
        if orig_shape[-1] != num_dims:
            raise ValueError(
                f"Expected locs last dimension {num_dims}, got {orig_shape[-1]}."
            )
        if num_dims * num_bits > 63:
            raise ValueError(
                f"Cannot encode {num_dims} dims x {num_bits} bits into int64."
            )

        locs_flat = locs.long().reshape(-1, num_dims)
        shifts = torch.arange(
            num_bits - 1,
            -1,
            -1,
            device=locs.device,
            dtype=torch.long,
        )
        gray = ((locs_flat.unsqueeze(-1) >> shifts) & 1).bool()

        for bit in range(0, num_bits):
            for dim in range(0, num_dims):
                mask = gray[:, dim, bit]
                gray[:, 0, bit + 1 :] = torch.logical_xor(
                    gray[:, 0, bit + 1 :], mask[:, None]
                )
                to_flip = torch.logical_and(
                    torch.logical_not(mask[:, None]).repeat(
                        1, gray.shape[2] - bit - 1
                    ),
                    torch.logical_xor(
                        gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]
                    ),
                )
                gray[:, dim, bit + 1 :] = torch.logical_xor(
                    gray[:, dim, bit + 1 :], to_flip
                )
                gray[:, 0, bit + 1 :] = torch.logical_xor(
                    gray[:, 0, bit + 1 :], to_flip
                )

        gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))
        hh_bin = hilbert_module.gray2binary(gray)
        powers = torch.arange(
            hh_bin.shape[-1] - 1,
            -1,
            -1,
            device=locs.device,
            dtype=torch.long,
        )
        code = (hh_bin.long() << powers).sum(dim=-1)
        return code.reshape(orig_shape[:-1])

    hilbert_module.encode = encode_compat
    default_module.hilbert_encode_ = encode_compat
    default_module.hilbert_encode = (
        lambda grid_coord, depth=16: encode_compat(
            grid_coord, num_dims=3, num_bits=depth
        )
    )
    hilbert_module._codex_old_torch_patch = True


#This encoder works by:
#Computing local features from neighboring points.
#Extracting a global summary of the neighborhood.
#Merging both local and global context.
#Reducing features into a single embedding for each group.
#It resembles the architecture of PointNet, which uses MLP and max pooling operations to summarize unordered point sets.

class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel, input_channels=6):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True) if KNN is not None else None

    def _sample_centers(self, xyz):
        try:
            return misc.fps(xyz, self.num_group)
        except Exception:
            num_points = xyz.size(1)
            if num_points < self.num_group:
                raise ValueError(
                    f"Point cloud has {num_points} points, fewer than num_group={self.num_group}."
                )
            center_idx = torch.linspace(
                0, num_points - 1, steps=self.num_group, device=xyz.device
            ).long()
            return xyz[:, center_idx, :]

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = self._sample_centers(xyz)  # B G 3
        # knn to get the neighborhood
        # idx = knn_query(xyz, center, self.group_size)  # B G M
        if self.knn is not None:
            _, idx = self.knn(xyz, center)  # B G M
        else:
            dist = torch.cdist(center, xyz)
            idx = dist.topk(k=self.group_size, dim=-1, largest=False)[1]
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        feature_dim = xyz.size(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, feature_dim).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def serialization(pos, feat=None, x_res=None, order="z", layers_outputs=[], grid_size=0.02):
    bs, n_p, _ = pos.size()
    if not isinstance(order, list):
        order = [order]

    scaled_coord = pos / grid_size
    grid_coord = torch.floor(scaled_coord).to(torch.int64)
    min_coord = grid_coord.min(dim=1, keepdim=True)[0]
    grid_coord = grid_coord - min_coord

    batch_idx = torch.arange(0, pos.shape[0], 1.0).unsqueeze(1).repeat(1, pos.shape[1]).to(torch.int64).to(pos.device)

    point_dict = {'batch': batch_idx.flatten(), 'grid_coord': grid_coord.flatten(0, 1), }
    point_dict = Point(**point_dict)
    point_dict.serialization(order=order)

    order = point_dict.serialized_order
    inverse_order = point_dict.serialized_inverse

    pos = pos.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if feat is not None:
        feat = feat.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if x_res is not None:
        x_res = x_res.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()

    for i in range(len(layers_outputs)):
        layers_outputs[i] = layers_outputs[i].flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    return pos, order, inverse_order, feat, x_res
        

def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):

    if ssm_cfg is None:
        ssm_cfg = {}
    if Mamba is None:
        raise ImportError(
            "mamba_ssm is required for PointMamba models. Use the pointmamba "
            "environment for PointMamba, or --model_arch ptv3_match for PTv3."
        )
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out: int = 0.,
            drop_path=0.,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_out = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids + pos

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, inference_params=inference_params
            )
            hidden_states = self.drop_out(hidden_states)

        hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))

        return hidden_states
      

class PointMambaScan(nn.Module):
    def __init__(self, num_group=128, group_size=16, input_channels=6):
        super().__init__()
        if num_group < 1:
            raise ValueError(f"num_group must be >= 1, got {num_group}.")
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}.")
        

        self.trans_dim = 384
        self.depth = 12
        self.cls_dim = 5

        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = 384
        self.use_cls_token = False
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims, input_channels=input_channels)

        
        drop_path = 0.1 
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.depth)]
        self.rms_norm = False 
        self.drop_out = 0
        self.max_head = True 
        self.avg_head = False 

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out=self.drop_out,
                                 drop_path=dpr)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1
        if self.max_head and self.avg_head:
            self.HEAD_CHANEL += 1

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        self.OrderScale_gamma_1, self.OrderScale_beta_1 = init_OrderScale(self.trans_dim)
        self.OrderScale_gamma_2, self.OrderScale_beta_2 = init_OrderScale(self.trans_dim)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    
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

    def _encode_tokens(self, pts):
        neighborhood, center = self.group_divider(pts)
       
        group_input_tokens = self.encoder(neighborhood)  # B G N
        center_xyz = center[:, :, :3]
        center_time = center[:, :, 6:7] if center.size(-1) > 6 else None
        pos = self.pos_embed(center_xyz)  # B G C
      
       #You have B samples in a batch,Each sample contains G groups, and Each group is represented by C features.
        # # reordering strategy
        _, _, _, group_input_tokens_forward, pos_forward = serialization_func(center_xyz, group_input_tokens, pos,
                                                                              'hilbert')
        _, _, _, group_input_tokens_backward, pos_backward = serialization_func(center_xyz, group_input_tokens, pos,
                                                                                'hilbert-trans')
        group_input_tokens_forward = apply_OrderScale(group_input_tokens_forward,
                                                      self.OrderScale_gamma_1, self.OrderScale_beta_1)
        group_input_tokens_backward = apply_OrderScale(group_input_tokens_backward,
                                                       self.OrderScale_gamma_2, self.OrderScale_beta_2)

        center_forward, _, _, _, _ = serialization_func(center_xyz, None, None, 'hilbert')
        center_backward, _, _, _, _ = serialization_func(center_xyz, None, None, 'hilbert-trans')
        if center_time is not None:
            _, _, _, center_time_forward, _ = serialization_func(center_xyz, center_time, None, 'hilbert')
            _, _, _, center_time_backward, _ = serialization_func(center_xyz, center_time, None, 'hilbert-trans')
            token_times = torch.cat([center_time_forward, center_time_backward], dim=1)
        else:
            token_times = None
        
        pos = torch.cat([pos_forward, pos_backward], dim=1)
        group_input_tokens = torch.cat([group_input_tokens_forward, group_input_tokens_backward], dim=1)
        token_centers = torch.cat([center_forward, center_backward], dim=1)

       
        x = group_input_tokens
        x = self.blocks(x, pos)
        return x, token_centers, token_times

    def forward(self, pts, return_token_info=False):
        x, token_centers, token_times = self._encode_tokens(pts)
        # x = self.norm(x)
        if self.use_cls_token:
            cls_token = x[:, -1, :]
            concat_f = torch.cat([cls_token, x.mean(1)], dim=1)
        else:
            concat_f = None
            if self.avg_head:
                if concat_f is None:
                    concat_f = x[:, :].mean(1)
                else:
                    concat_f = torch.cat([concat_f, x[:, :].mean(1)], dim=1)
            if self.max_head:
                if concat_f is None:
                    concat_f = x[:, :].max(1)[0]
                    # concat_f = x[:, -1]
                else:
                    concat_f = torch.cat([concat_f, x[:, :].max(1)[0]], dim=1)
                    # concat_f = torch.cat([concat_f, x[:, -1]], dim=1)
            if concat_f is None:
                assert False
        # concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)

        if return_token_info:
            return {
                "global_feat": concat_f,
                "token_features": x,
                "token_centers": token_centers,
                "token_times": token_times,
            }
        return concat_f



class Net_mamba_point(torch.nn.Module):
    def __init__(
        self,
        NUM_CLASS=3,
        signed_diff=False,
        token_match_k=16,
        token_match_tau_feat=0.1,
        token_match_tau_geo=1.0,
        token_match_detach_scores=False,
        num_group=128,
        group_size=16,
        use_residual_token_mamba=False,
        residual_token_mamba_depth=2,
        use_change_aware_token_fusion=False,
        use_token_auxiliary_loss=False,
        use_explicit_change_decomposition=False,
        use_existence_aware_fusion=False,
        use_existence_auxiliary_loss=False,
        use_direction_auxiliary_loss=False,
        use_binary_change_auxiliary_loss=False,
        use_changed_direction_auxiliary_loss=False,
        use_temporal_embedding=False,
        use_temporal_4class_auxiliary_loss=False,
        use_null_correspondence=False,
        null_match_init=-2.0,
    ) -> None:
        super().__init__()
        if token_match_k < 1:
            raise ValueError(f"token_match_k must be >= 1, got {token_match_k}.")
        if token_match_tau_feat <= 0:
            raise ValueError(
                f"token_match_tau_feat must be > 0, got {token_match_tau_feat}."
            )
        if token_match_tau_geo <= 0:
            raise ValueError(
                f"token_match_tau_geo must be > 0, got {token_match_tau_geo}."
            )
        if residual_token_mamba_depth < 1:
            raise ValueError(
                f"residual_token_mamba_depth must be >= 1, got {residual_token_mamba_depth}."
            )
        self.mamba = PointMambaScan(num_group=num_group, group_size=group_size)
        self.signed_diff = signed_diff
        self.num_group = num_group
        self.group_size = group_size
        self.token_match_k = token_match_k
        self.token_match_tau_feat = token_match_tau_feat
        self.token_match_tau_geo = token_match_tau_geo
        self.token_match_detach_scores = token_match_detach_scores
        self.use_residual_token_mamba = use_residual_token_mamba
        self.use_change_aware_token_fusion = use_change_aware_token_fusion
        self.use_token_auxiliary_loss = use_token_auxiliary_loss
        self.use_explicit_change_decomposition = use_explicit_change_decomposition
        self.use_existence_aware_fusion = use_existence_aware_fusion
        self.use_existence_auxiliary_loss = use_existence_auxiliary_loss
        self.use_direction_auxiliary_loss = use_direction_auxiliary_loss
        self.use_binary_change_auxiliary_loss = use_binary_change_auxiliary_loss
        self.use_changed_direction_auxiliary_loss = use_changed_direction_auxiliary_loss
        self.use_temporal_embedding = use_temporal_embedding
        self.use_temporal_4class_auxiliary_loss = use_temporal_4class_auxiliary_loss
        self.use_null_correspondence = use_null_correspondence
        self.residual_token_mamba_depth = residual_token_mamba_depth
        self.time_embed = (
            nn.Parameter(torch.zeros(2, 384))
            if use_temporal_embedding
            else None
        )
        self.null_match_logit = (
            nn.Parameter(torch.tensor(float(null_match_init)))
            if use_null_correspondence
            else None
        )
        self.residual_token_mamba = (
            MixerModel(
                d_model=384,
                n_layer=residual_token_mamba_depth,
                rms_norm=False,
                drop_out=0,
                drop_path=0.0,
            )
            if use_residual_token_mamba
            else None
        )
        fusion_evidence_dim = 4 if use_existence_aware_fusion else 0
        fusion_in_dim = 384 * 5 + fusion_evidence_dim
        self.change_token_fusion = None
        self.change_token_fusion_t0 = (
            nn.Sequential(
                nn.Linear(fusion_in_dim, 384),
                nn.LayerNorm(384),
                nn.GELU(),
                nn.Linear(384, 384),
            )
            if use_change_aware_token_fusion
            else None
        )
        self.change_token_fusion_t1 = (
            nn.Sequential(
                nn.Linear(fusion_in_dim, 384),
                nn.LayerNorm(384),
                nn.GELU(),
                nn.Linear(384, 384),
            )
            if use_change_aware_token_fusion
            else None
        )
        self.temporal_interaction = (
            nn.Sequential(
                nn.Linear(384 * 5, 384),
                nn.LayerNorm(384),
                nn.GELU(),
                nn.Linear(384, 384),
            )
            if use_explicit_change_decomposition
            else None
        )
        self.invariant_branch = (
            nn.Sequential(
                nn.Linear(384 * 5, 384),
                nn.LayerNorm(384),
                nn.GELU(),
                nn.Linear(384, 384),
            )
            if use_explicit_change_decomposition
            else None
        )
        self.change_branch = (
            nn.Sequential(
                nn.Linear(384 * 5, 384),
                nn.LayerNorm(384),
                nn.GELU(),
                nn.Linear(384, 384),
            )
            if use_explicit_change_decomposition
            else None
        )
        self.token_aux_head = (
            nn.Sequential(
                nn.Linear(384 if use_explicit_change_decomposition else 384 * 5, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, NUM_CLASS),
            )
            if use_token_auxiliary_loss
            else None
        )
        self.existence_aux_head_t0 = (
            nn.Sequential(
                nn.Linear(384, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 1),
            )
            if use_existence_auxiliary_loss
            else None
        )
        self.existence_aux_head_t1 = (
            nn.Sequential(
                nn.Linear(384, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 1),
            )
            if use_existence_auxiliary_loss
            else None
        )
        head_feature_blocks = 5
        if use_change_aware_token_fusion:
            head_feature_blocks += 2
        if use_explicit_change_decomposition:
            # Project invariant and change representations from both times.
            head_feature_blocks += 4
        head_in_dim = 384 * head_feature_blocks + 12
        self.point_head = nn.Sequential(
            nn.Linear(head_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, NUM_CLASS),
        )
        self.direction_aux_head_t0 = (
            nn.Sequential(
                nn.Linear(head_in_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 2),
            )
            if use_direction_auxiliary_loss
            else None
        )
        self.direction_aux_head_t1 = (
            nn.Sequential(
                nn.Linear(head_in_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 2),
            )
            if use_direction_auxiliary_loss
            else None
        )
        self.binary_change_head = (
            nn.Sequential(
                nn.Linear(head_in_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 2),
            )
            if use_binary_change_auxiliary_loss
            else None
        )
        self.changed_direction_head = (
            nn.Sequential(
                nn.Linear(head_in_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 2),
            )
            if use_changed_direction_auxiliary_loss
            else None
        )
        self.temporal_4class_head = (
            nn.Sequential(
                nn.Linear(head_in_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 4),
            )
            if use_temporal_4class_auxiliary_loss
            else None
        )
    def _project_tokens_to_points(self, point_coord, token_centers, token_features):
        """
        Soft assignment from tokens to points
        point_coord:  B N 3
        token_centers: B G 3
        token_features: B G C
        """

        # B N G
        dist = torch.cdist(point_coord, token_centers)

        # convert distance to similarity
        weight = torch.softmax(-dist, dim=-1)  # B N G

        # weighted sum of token features
        point_feat = torch.matmul(weight, token_features)  # B N C

        return point_feat

    def _token_soft_match(
        self,
        query_centers,
        key_centers,
        query_tokens,
        key_tokens,
        return_info=False,
    ):
        """
        Cross-temporal token soft matching.
        For each query token, reconstruct an aligned token feature from key-frame
        token candidates using geometry-guided feature similarity.
        """
        k = min(self.token_match_k, key_centers.size(1))
        batch_size, num_tokens, _ = query_centers.shape
        channels = key_tokens.size(-1)
        dist = torch.cdist(query_centers, key_centers)  # B Gq Gk
        knn_dist, knn_idx = dist.topk(k=k, dim=-1, largest=False)  # B Gq k

        key_tokens_expanded = key_tokens.unsqueeze(1).expand(
            batch_size, num_tokens, -1, channels
        )
        local_key_tokens = torch.gather(
            key_tokens_expanded,
            dim=2,
            index=knn_idx.unsqueeze(-1).expand(-1, -1, -1, channels),
        )
        query_tokens_norm = F.normalize(query_tokens, dim=-1)
        local_key_norm = F.normalize(local_key_tokens, dim=-1)
        feat_score = (query_tokens_norm.unsqueeze(2) * local_key_norm).sum(dim=-1)
        geo_score = -knn_dist / max(self.token_match_tau_geo, 1e-6)
        score = feat_score / max(self.token_match_tau_feat, 1e-6) + geo_score
        if self.use_null_correspondence:
            null_score = self.null_match_logit.expand(batch_size, num_tokens, 1)
            score_with_null = torch.cat([score, null_score], dim=-1)
            weight_with_null = torch.softmax(score_with_null, dim=-1)
            weight = weight_with_null[:, :, :k]
            null_prob = weight_with_null[:, :, k:].contiguous()
        else:
            weight = torch.softmax(score, dim=-1)
            null_prob = torch.zeros(
                batch_size, num_tokens, 1, device=score.device, dtype=score.dtype
            )
        if self.token_match_detach_scores:
            weight = weight.detach()
            null_prob = null_prob.detach()
        aligned_tokens = (weight.unsqueeze(-1) * local_key_tokens).sum(dim=2)
        if not return_info:
            return aligned_tokens

        confidence = weight.max(dim=-1, keepdim=True)[0]
        if self.use_null_correspondence:
            entropy_weight = torch.cat([weight, null_prob], dim=-1)
            entropy_norm = math.log(max(k + 1, 2))
        else:
            entropy_weight = weight
            entropy_norm = math.log(max(k, 2))
        entropy = -(
            entropy_weight * torch.log(entropy_weight.clamp_min(1e-8))
        ).sum(dim=-1, keepdim=True)
        entropy = entropy / entropy_norm
        geo_residual = (weight * knn_dist).sum(dim=-1, keepdim=True)
        return {
            "aligned": aligned_tokens,
            "confidence": confidence,
            "entropy": entropy,
            "geo_residual": geo_residual,
            "null_prob": null_prob,
        }

    def _refine_token_residual(self, token_centers, token_residual):
        if self.residual_token_mamba is None:
            return token_residual
        pos = self.mamba.pos_embed(token_centers[:, :, :3])
        return self.residual_token_mamba(token_residual, pos)

    def _change_aware_token_fusion(
        self,
        query_tokens,
        aligned_key_tokens,
        match_info=None,
        direction="t0",
    ):
        if self.change_token_fusion_t0 is None or self.change_token_fusion_t1 is None:
            return None
        signed_diff = aligned_key_tokens - query_tokens
        abs_diff = torch.abs(signed_diff)
        prod = query_tokens * aligned_key_tokens
        features = [
            query_tokens,
            aligned_key_tokens,
            abs_diff,
            signed_diff,
            prod,
        ]
        if self.use_existence_aware_fusion:
            if match_info is None:
                raise ValueError("existence-aware fusion requires match_info.")
            features.extend(
                [
                    match_info["confidence"],
                    match_info["entropy"],
                    match_info["geo_residual"],
                    match_info["null_prob"],
                ]
            )
        fusion_input = torch.cat(features, dim=-1)
        if self.change_token_fusion_t0 is not None and direction == "t0":
            return self.change_token_fusion_t0(fusion_input)
        if self.change_token_fusion_t1 is not None and direction == "t1":
            return self.change_token_fusion_t1(fusion_input)
        raise ValueError(f"unknown change-fusion direction: {direction}")

    def _build_token_pair_features(self, query_tokens, aligned_key_tokens):
        signed_diff = aligned_key_tokens - query_tokens
        abs_diff = torch.abs(signed_diff)
        prod = query_tokens * aligned_key_tokens
        return torch.cat(
            [
                query_tokens,
                aligned_key_tokens,
                abs_diff,
                signed_diff,
                prod,
            ],
            dim=-1,
        )

    def _cross_temporal_interact(self, query_tokens, aligned_key_tokens):
        if self.temporal_interaction is None:
            return query_tokens
        return query_tokens + self.temporal_interaction(
            self._build_token_pair_features(query_tokens, aligned_key_tokens)
        )

    def _decompose_change_information(self, query_tokens, aligned_key_tokens):
        if self.invariant_branch is None or self.change_branch is None:
            return None, None
        pair_features = self._build_token_pair_features(query_tokens, aligned_key_tokens)
        invariant_tokens = self.invariant_branch(pair_features)
        change_tokens = self.change_branch(pair_features)
        return invariant_tokens, change_tokens


    def _build_mamba_inputs(self, data):
        if isinstance(data, dict):
            coord0 = data["coord0"]
            feat0 = data["feat0"]
            coord1 = data["coord1"]
            feat1 = data["feat1"]

            if coord0.dim() == 2:
                coord0 = coord0.unsqueeze(0)
                feat0 = feat0.unsqueeze(0)
                coord1 = coord1.unsqueeze(0)
                feat1 = feat1.unsqueeze(0)

            x_dense0 = torch.cat([coord0, feat0], dim=-1).contiguous()
            x_dense1 = torch.cat([coord1, feat1], dim=-1).contiguous()
            return coord0, coord1, x_dense0, x_dense1

        x_dense0, _ = to_dense_batch(data.x, data.batch)
        x_dense1, _ = to_dense_batch(data.x2, data.batch2)
        return x_dense0[:, :, :3], x_dense1[:, :, :3], x_dense0, x_dense1

    
   
    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_mamba_inputs(data)

        t0_info = self.mamba(x_dense0, return_token_info=True)
        t1_info = self.mamba(x_dense1, return_token_info=True)
        t0_tokens = t0_info["token_features"]
        t1_tokens = t1_info["token_features"]
        t0_centers = t0_info["token_centers"]
        t1_centers = t1_info["token_centers"]
        if self.time_embed is not None:
            t0_tokens = t0_tokens + self.time_embed[0].view(1, 1, -1)
            t1_tokens = t1_tokens + self.time_embed[1].view(1, 1, -1)

        point_feat_t1 = self._project_tokens_to_points(
            coord1, t1_centers, t1_tokens
        )
        point_feat_t0 = self._project_tokens_to_points(
            coord0, t0_centers, t0_tokens
        )

        if self.signed_diff:
            point_feat_diff = point_feat_t1 - point_feat_t0
        else:
            point_feat_diff = torch.abs(point_feat_t1 - point_feat_t0)

        match_t0_to_t1 = self._token_soft_match(
            t0_centers,
            t1_centers,
            t0_tokens,
            t1_tokens,
            return_info=self.use_existence_aware_fusion or self.use_existence_auxiliary_loss,
        )
        match_t1_to_t0 = self._token_soft_match(
            t1_centers,
            t0_centers,
            t1_tokens,
            t0_tokens,
            return_info=self.use_existence_aware_fusion or self.use_existence_auxiliary_loss,
        )
        if isinstance(match_t0_to_t1, dict):
            aligned_t1_tokens_for_t0 = match_t0_to_t1["aligned"]
        else:
            aligned_t1_tokens_for_t0 = match_t0_to_t1
            match_t0_to_t1 = None

        if isinstance(match_t1_to_t0, dict):
            aligned_t0_tokens_for_t1 = match_t1_to_t0["aligned"]
        else:
            aligned_t0_tokens_for_t1 = match_t1_to_t0
            match_t1_to_t0 = None

        interacted_t0_tokens = self._cross_temporal_interact(
            t0_tokens, aligned_t1_tokens_for_t0
        )
        interacted_t1_tokens = self._cross_temporal_interact(
            t1_tokens, aligned_t0_tokens_for_t1
        )

        invariant_t0 = invariant_t1 = None
        change_t0 = change_t1 = None
        if self.use_explicit_change_decomposition:
            aligned_interacted_t1_for_t0 = self._token_soft_match(
                t0_centers, t1_centers, interacted_t0_tokens, interacted_t1_tokens
            )
            aligned_interacted_t0_for_t1 = self._token_soft_match(
                t1_centers, t0_centers, interacted_t1_tokens, interacted_t0_tokens
            )
            invariant_t0, change_t0 = self._decompose_change_information(
                interacted_t0_tokens, aligned_interacted_t1_for_t0
            )
            invariant_t1, change_t1 = self._decompose_change_information(
                interacted_t1_tokens, aligned_interacted_t0_for_t1
            )

        if self.use_residual_token_mamba:
            token_residual_t0 = torch.abs(t0_tokens - aligned_t1_tokens_for_t0)
            token_residual_t1 = torch.abs(t1_tokens - aligned_t0_tokens_for_t1)
            token_residual_t0 = self._refine_token_residual(t0_centers, token_residual_t0)
            token_residual_t1 = self._refine_token_residual(t1_centers, token_residual_t1)
            recon_diff_t0 = self._project_tokens_to_points(
                coord0, t0_centers, token_residual_t0
            )
            recon_diff_t1 = self._project_tokens_to_points(
                coord1, t1_centers, token_residual_t1
            )
        else:
            aligned_t1_for_t0 = self._project_tokens_to_points(
                coord0, t0_centers, aligned_t1_tokens_for_t0
            )
            aligned_t0_for_t1 = self._project_tokens_to_points(
                coord1, t1_centers, aligned_t0_tokens_for_t1
            )
            if self.signed_diff:
                recon_diff_t0 = point_feat_t0 - aligned_t1_for_t0
                recon_diff_t1 = point_feat_t1 - aligned_t0_for_t1
            else:
                recon_diff_t0 = torch.abs(point_feat_t0 - aligned_t1_for_t0)
                recon_diff_t1 = torch.abs(point_feat_t1 - aligned_t0_for_t1)

        point_features = [
            point_feat_t0,
            point_feat_t1,
            point_feat_diff,
            recon_diff_t0,
            recon_diff_t1,
        ]

        change_token_t0 = change_token_t1 = None
        if self.use_change_aware_token_fusion:
            change_token_t0 = self._change_aware_token_fusion(
                interacted_t0_tokens,
                aligned_t1_tokens_for_t0,
                match_info=match_t0_to_t1,
                direction="t0",
            )
            change_token_t1 = self._change_aware_token_fusion(
                interacted_t1_tokens,
                aligned_t0_tokens_for_t1,
                match_info=match_t1_to_t0,
                direction="t1",
            )
            point_features.extend(
                [
                    self._project_tokens_to_points(coord0, t0_centers, change_token_t0),
                    self._project_tokens_to_points(coord1, t1_centers, change_token_t1),
                ]
            )

        if self.use_explicit_change_decomposition:
            point_features.extend(
                [
                    self._project_tokens_to_points(coord0, t0_centers, invariant_t0),
                    self._project_tokens_to_points(coord1, t1_centers, invariant_t1),
                    self._project_tokens_to_points(coord0, t0_centers, change_t0),
                    self._project_tokens_to_points(coord1, t1_centers, change_t1),
                ]
            )

        point_input = torch.cat(
            point_features + [x_dense0, x_dense1],
            dim=-1,
        )
        out = self.point_head(point_input)
        needs_auxiliary_outputs = (
            self.token_aux_head is not None
            or self.use_explicit_change_decomposition
            or self.use_existence_auxiliary_loss
            or self.use_existence_aware_fusion
            or self.use_direction_auxiliary_loss
            or self.use_binary_change_auxiliary_loss
            or self.use_changed_direction_auxiliary_loss
            or self.use_temporal_4class_auxiliary_loss
        )
        if not needs_auxiliary_outputs:
            return out

        if self.use_explicit_change_decomposition:
            token_aux_features_t0 = change_t0
            token_aux_features_t1 = change_t1
        else:
            token_aux_features_t0 = self._build_token_pair_features(
                t0_tokens, aligned_t1_tokens_for_t0
            )
            token_aux_features_t1 = self._build_token_pair_features(
                t1_tokens, aligned_t0_tokens_for_t1
            )

        outputs = {
            "point_logits": out,
            "token_centers_t0": t0_centers,
            "token_centers_t1": t1_centers,
        }
        if self.direction_aux_head_t0 is not None and self.direction_aux_head_t1 is not None:
            outputs.update(
                {
                    "direction_logits_t0": self.direction_aux_head_t0(point_input),
                    "direction_logits_t1": self.direction_aux_head_t1(point_input),
                }
            )
        if self.binary_change_head is not None:
            outputs["binary_change_logits"] = self.binary_change_head(point_input)
        if self.changed_direction_head is not None:
            outputs["changed_direction_logits"] = self.changed_direction_head(point_input)
        if self.temporal_4class_head is not None:
            outputs["temporal_4class_logits"] = self.temporal_4class_head(point_input)
        if match_t0_to_t1 is not None and match_t1_to_t0 is not None:
            outputs.update(
                {
                    "match_confidence_t0": match_t0_to_t1["confidence"],
                    "match_confidence_t1": match_t1_to_t0["confidence"],
                    "match_entropy_t0": match_t0_to_t1["entropy"],
                    "match_entropy_t1": match_t1_to_t0["entropy"],
                    "match_geo_residual_t0": match_t0_to_t1["geo_residual"],
                    "match_geo_residual_t1": match_t1_to_t0["geo_residual"],
                    "match_null_prob_t0": match_t0_to_t1["null_prob"],
                    "match_null_prob_t1": match_t1_to_t0["null_prob"],
                }
            )
        if self.existence_aux_head_t0 is not None and self.existence_aux_head_t1 is not None:
            existence_input_t0 = change_t0 if change_t0 is not None else change_token_t0
            existence_input_t1 = change_t1 if change_t1 is not None else change_token_t1
            if existence_input_t0 is None:
                existence_input_t0 = interacted_t0_tokens
            if existence_input_t1 is None:
                existence_input_t1 = interacted_t1_tokens
            outputs.update(
                {
                    "existence_logits_t0": self.existence_aux_head_t0(existence_input_t0).squeeze(-1),
                    "existence_logits_t1": self.existence_aux_head_t1(existence_input_t1).squeeze(-1),
                }
            )
        if self.token_aux_head is not None:
            outputs.update(
                {
                    "token_logits_t0": self.token_aux_head(token_aux_features_t0),
                    "token_logits_t1": self.token_aux_head(token_aux_features_t1),
                }
            )
        if self.use_explicit_change_decomposition:
            outputs.update(
                {
                    "invariant_tokens_t0": invariant_t0,
                    "invariant_tokens_t1": invariant_t1,
                    "change_tokens_t0": change_t0,
                    "change_tokens_t1": change_t1,
                }
            )
        return outputs
        


class SimpleDirectionFusionMambaPoint(Net_mamba_point):
    """
    Clean model entry for the main experiment.

    Pipeline:
      PointMamba token encoding
      -> bidirectional local soft matching
      -> direction-aware change fusion
      -> token-to-point projection
      -> point-level 3-class prediction
    """

    def __init__(
        self,
        num_classes=3,
        token_match_k=16,
        token_match_tau_feat=0.1,
        token_match_tau_geo=1.0,
        token_match_detach_scores=True,
        num_group=128,
        group_size=16,
        signed_diff=True,
        use_binary_change_auxiliary_loss=False,
        use_changed_direction_auxiliary_loss=False,
        use_temporal_embedding=False,
        use_temporal_4class_auxiliary_loss=False,
    ) -> None:
        super().__init__(
            NUM_CLASS=num_classes,
            signed_diff=signed_diff,
            token_match_k=token_match_k,
            token_match_tau_feat=token_match_tau_feat,
            token_match_tau_geo=token_match_tau_geo,
            token_match_detach_scores=token_match_detach_scores,
            num_group=num_group,
            group_size=group_size,
            use_residual_token_mamba=False,
            use_change_aware_token_fusion=True,
            use_token_auxiliary_loss=False,
            use_explicit_change_decomposition=False,
            use_existence_aware_fusion=False,
            use_existence_auxiliary_loss=False,
            use_direction_auxiliary_loss=False,
            use_binary_change_auxiliary_loss=use_binary_change_auxiliary_loss,
            use_changed_direction_auxiliary_loss=use_changed_direction_auxiliary_loss,
            use_temporal_embedding=use_temporal_embedding,
            use_temporal_4class_auxiliary_loss=use_temporal_4class_auxiliary_loss,
            use_null_correspondence=False,
        )


class NoDirectionFusionMambaPoint(Net_mamba_point):
    """
    Ablation model with shared, no-direction-aware change fusion.

    Pipeline:
      PointMamba token encoding
      -> bidirectional local soft matching
      -> shared change fusion for both directions
      -> token-to-point projection
      -> point-level 3-class prediction
    """

    def __init__(
        self,
        num_classes=3,
        token_match_k=16,
        token_match_tau_feat=0.1,
        token_match_tau_geo=1.0,
        token_match_detach_scores=True,
        num_group=128,
        group_size=16,
        signed_diff=True,
    ) -> None:
        super().__init__(
            NUM_CLASS=num_classes,
            signed_diff=signed_diff,
            token_match_k=token_match_k,
            token_match_tau_feat=token_match_tau_feat,
            token_match_tau_geo=token_match_tau_geo,
            token_match_detach_scores=token_match_detach_scores,
            num_group=num_group,
            group_size=group_size,
            use_residual_token_mamba=False,
            use_change_aware_token_fusion=True,
            use_token_auxiliary_loss=False,
            use_explicit_change_decomposition=False,
            use_existence_aware_fusion=False,
            use_existence_auxiliary_loss=False,
            use_direction_auxiliary_loss=False,
            use_binary_change_auxiliary_loss=False,
            use_changed_direction_auxiliary_loss=False,
            use_temporal_embedding=False,
            use_temporal_4class_auxiliary_loss=False,
            use_null_correspondence=False,
        )
        # Share the same fusion function for T0->T1 and T1->T0 so this ablation
        # removes direction-specific fusion parameters while preserving the same
        # feature dimensionality as the direction-aware baseline.
        shared_fusion = self.change_token_fusion_t0
        self.change_token_fusion_t0 = shared_fusion
        self.change_token_fusion_t1 = shared_fusion


class PTV3DirectionFusionPoint(nn.Module):
    """
    Point Transformer V3 encoder ablation using the ME-CPT/Pointcept backbone.

    Pipeline:
      PTv3 point encoding for T0/T1 separately
      -> bidirectional geometric local soft matching at point level
      -> direction-aware point fusion
      -> point-level 3-class prediction
    """

    def __init__(
        self,
        num_classes=3,
        match_k=16,
        match_tau_feat=0.1,
        match_tau_geo=1.0,
        signed_diff=True,
        direction_aware=True,
        ptv3_order=("z", "z-trans", "hilbert", "hilbert-trans"),
        ptv3_patch_size=512,
        use_binary_change_auxiliary_loss=False,
        use_removal_like_auxiliary_loss=False,
        me_cpt_root=osp.join(osp.dirname(osp.abspath(__file__)), "vendor", "ME-CPT"),
    ) -> None:
        super().__init__()
        if match_k < 1:
            raise ValueError(f"match_k must be >= 1, got {match_k}.")
        if match_tau_feat <= 0:
            raise ValueError(f"match_tau_feat must be > 0, got {match_tau_feat}.")
        if match_tau_geo <= 0:
            raise ValueError(f"match_tau_geo must be > 0, got {match_tau_geo}.")
        if me_cpt_root not in sys.path:
            sys.path.insert(0, me_cpt_root)
        try:
            from pointcept.models.point_transformer_v3.ptv3_cd_multi import (
                PointTransformerV3,
            )
            _patch_pointcept_hilbert_for_old_torch()
        except Exception as exc:
            raise ImportError(
                "Could not import ME-CPT PointTransformerV3. Make sure you run "
                "inside an environment with ME-CPT/Pointcept dependencies "
                "(spconv, torch_scatter, pointops) installed."
            ) from exc

        enc_patch_size = tuple([int(ptv3_patch_size)] * 5)
        dec_patch_size = tuple([int(ptv3_patch_size)] * 4)
        self.encoder = PointTransformerV3(
            in_channels=7,
            order=tuple(ptv3_order),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(64, 64, 64, 128, 256),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=enc_patch_size,
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=dec_patch_size,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            shuffle_orders=False,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=False,
            upcast_attention=True,
            upcast_softmax=True,
            cls_mode=False,
            pdnorm_bn=False,
            pdnorm_ln=False,
        )
        channels = 64
        self.match_k = match_k
        self.match_tau_feat = match_tau_feat
        self.match_tau_geo = match_tau_geo
        self.signed_diff = signed_diff
        self.direction_aware = direction_aware
        self.use_binary_change_auxiliary_loss = use_binary_change_auxiliary_loss
        self.use_removal_like_auxiliary_loss = use_removal_like_auxiliary_loss
        self._keep_encoder_eval = False
        fusion_in_dim = channels * 5
        self.change_fusion_t0 = nn.Sequential(
            nn.Linear(fusion_in_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.change_fusion_t1 = nn.Sequential(
            nn.Linear(fusion_in_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        if not direction_aware:
            self.change_fusion_t1 = self.change_fusion_t0
        head_feature_blocks = 7 if direction_aware else 6
        head_in_dim = channels * head_feature_blocks + 12
        self.point_head = nn.Sequential(
            nn.Linear(head_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        self.binary_change_head = self._make_binary_change_head(head_in_dim)
        self.removal_like_head = self._make_removal_like_head(head_in_dim)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "_keep_encoder_eval", False):
            self.encoder.eval()
        return self

    @staticmethod
    def _build_dense_inputs(data):
        coord0 = data["coord0"]
        feat0 = data["feat0"]
        coord1 = data["coord1"]
        feat1 = data["feat1"]
        if coord0.dim() == 2:
            coord0 = coord0.unsqueeze(0)
            feat0 = feat0.unsqueeze(0)
            coord1 = coord1.unsqueeze(0)
            feat1 = feat1.unsqueeze(0)
        x_dense0 = torch.cat([coord0, feat0], dim=-1).contiguous()
        x_dense1 = torch.cat([coord1, feat1], dim=-1).contiguous()
        return coord0, coord1, x_dense0, x_dense1

    def _make_binary_change_head(self, input_dim):
        if not self.use_binary_change_auxiliary_loss:
            return None
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def _make_removal_like_head(self, input_dim):
        if not self.use_removal_like_auxiliary_loss:
            return None
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def _format_point_output(self, point_input, extra_outputs=None):
        point_logits = self.point_head(point_input)
        has_auxiliary_head = (
            self.binary_change_head is not None or self.removal_like_head is not None
        )
        if not has_auxiliary_head and not extra_outputs:
            return point_logits
        output = {"point_logits": point_logits}
        if self.binary_change_head is not None:
            output["binary_change_logits"] = self.binary_change_head(point_input)
        if self.removal_like_head is not None:
            output["removal_like_logits"] = self.removal_like_head(point_input)
        if extra_outputs:
            output.update(extra_outputs)
        return output

    def _encode_points(self, coord, x_dense, time_value):
        batch_size, num_points, _ = coord.shape
        flat_coord = coord.reshape(batch_size * num_points, 3).contiguous()
        time_feat = torch.full(
            (batch_size, num_points, 1),
            float(time_value),
            dtype=x_dense.dtype,
            device=x_dense.device,
        )
        x_with_time = torch.cat([x_dense, time_feat], dim=-1)
        flat_feat = x_with_time.reshape(
            batch_size * num_points, x_with_time.size(-1)
        ).contiguous()
        batch = torch.arange(batch_size, device=coord.device).repeat_interleave(num_points)
        offset = torch.arange(
            1,
            batch_size + 1,
            device=coord.device,
            dtype=torch.long,
        ) * num_points
        point = self.encoder(
            {
                "coord": flat_coord,
                "feat": flat_feat,
                "batch": batch,
                "offset": offset,
                "grid_size": 0.03,
            }
        )
        return point.feat.reshape(batch_size, num_points, -1).contiguous()

    def _point_soft_match(self, query_coord, key_coord, query_feat, key_feat):
        return self._point_soft_match_info(
            query_coord, key_coord, query_feat, key_feat
        )["aligned"]

    def _point_soft_match_info(self, query_coord, key_coord, query_feat, key_feat):
        k = min(self.match_k, key_coord.size(1))
        batch_size, num_query, _ = query_coord.shape
        num_key = key_coord.size(1)

        if getattr(self, "use_scalable_knn", False):
            if torch_cluster_knn is None:
                raise RuntimeError(
                    "Scalable matching requires torch_cluster.knn in the active environment."
                )
            knn_indices = []
            knn_distances = []
            for batch_id in range(batch_size):
                edge_index = torch_cluster_knn(
                    key_coord[batch_id].contiguous(),
                    query_coord[batch_id].contiguous(),
                    k,
                )
                query_index, key_index = edge_index[0], edge_index[1]
                order = torch.argsort(query_index)
                key_index = key_index[order].reshape(num_query, k)
                local_key_coord = key_coord[batch_id][key_index]
                distance = torch.norm(
                    query_coord[batch_id].unsqueeze(1) - local_key_coord,
                    dim=-1,
                )
                knn_indices.append(key_index)
                knn_distances.append(distance)
            knn_idx = torch.stack(knn_indices, dim=0)
            knn_dist = torch.stack(knn_distances, dim=0)
            batch_idx = torch.arange(batch_size, device=key_feat.device).view(
                batch_size, 1, 1
            )
            aligned_parts = []
            weight_parts = []
            score_parts = []
            query_chunk_size = int(
                getattr(self, "match_query_chunk_size", 4096)
            )
            detach_scores = bool(getattr(self, "match_detach_scores", False))
            for start in range(0, num_query, query_chunk_size):
                end = min(start + query_chunk_size, num_query)
                local_idx = knn_idx[:, start:end]
                local_key_feat = key_feat[batch_idx, local_idx]
                if detach_scores:
                    with torch.no_grad():
                        feat_score = (
                            F.normalize(
                                query_feat[:, start:end].detach(), dim=-1
                            ).unsqueeze(2)
                            * F.normalize(local_key_feat.detach(), dim=-1)
                        ).sum(dim=-1)
                        geo_score = (
                            -knn_dist[:, start:end]
                            / max(self.match_tau_geo, 1e-6)
                        )
                        score = (
                            feat_score / max(self.match_tau_feat, 1e-6)
                            + geo_score
                        )
                        weight = torch.softmax(score, dim=-1)
                else:
                    feat_score = (
                        F.normalize(query_feat[:, start:end], dim=-1).unsqueeze(2)
                        * F.normalize(local_key_feat, dim=-1)
                    ).sum(dim=-1)
                    geo_score = (
                        -knn_dist[:, start:end]
                        / max(self.match_tau_geo, 1e-6)
                    )
                    score = (
                        feat_score / max(self.match_tau_feat, 1e-6) + geo_score
                    )
                    weight = torch.softmax(score, dim=-1)
                aligned_parts.append(
                    (weight.unsqueeze(-1) * local_key_feat).sum(dim=2)
                )
                weight_parts.append(weight)
                score_parts.append(score)
            aligned = torch.cat(aligned_parts, dim=1)
            weight = torch.cat(weight_parts, dim=1)
            score = torch.cat(score_parts, dim=1)
            return {
                "aligned": aligned,
                "weight": weight,
                "knn_idx": knn_idx,
                "knn_dist": knn_dist,
                "score": score,
            }

        # Preserve the original path for common 4096-point experiments. At
        # larger resolutions, query chunking computes the exact same top-k
        # neighbors without materializing an N x N distance matrix.
        max_full_distance_elements = 4096 * 4096
        chunk_size = num_query
        if num_query * num_key > max_full_distance_elements:
            chunk_size = min(num_query, 512)

        aligned_parts = []
        weight_parts = []
        index_parts = []
        distance_parts = []
        score_parts = []
        normalized_key_feat = F.normalize(key_feat, dim=-1)
        batch_idx = torch.arange(batch_size, device=key_feat.device).view(
            batch_size, 1, 1
        )
        for start in range(0, num_query, chunk_size):
            end = min(start + chunk_size, num_query)
            dist = torch.cdist(query_coord[:, start:end], key_coord)
            knn_dist, knn_idx = dist.topk(k=k, dim=-1, largest=False)
            local_key_feat = key_feat[batch_idx, knn_idx]
            local_key_feat_normalized = normalized_key_feat[batch_idx, knn_idx]
            feat_score = (
                F.normalize(query_feat[:, start:end], dim=-1).unsqueeze(2)
                * local_key_feat_normalized
            ).sum(dim=-1)
            geo_score = -knn_dist / max(self.match_tau_geo, 1e-6)
            score = feat_score / max(self.match_tau_feat, 1e-6) + geo_score
            weight = torch.softmax(score, dim=-1)
            aligned = (weight.unsqueeze(-1) * local_key_feat).sum(dim=2)
            aligned_parts.append(aligned)
            weight_parts.append(weight)
            index_parts.append(knn_idx)
            distance_parts.append(knn_dist)
            score_parts.append(score)

        aligned = torch.cat(aligned_parts, dim=1)
        weight = torch.cat(weight_parts, dim=1)
        knn_idx = torch.cat(index_parts, dim=1)
        knn_dist = torch.cat(distance_parts, dim=1)
        score = torch.cat(score_parts, dim=1)
        return {
            "aligned": aligned,
            "weight": weight,
            "knn_idx": knn_idx,
            "knn_dist": knn_dist,
            "score": score,
        }

    @staticmethod
    def _pair_features(query_feat, aligned_feat):
        signed_diff = aligned_feat - query_feat
        abs_diff = torch.abs(signed_diff)
        prod = query_feat * aligned_feat
        return torch.cat(
            [query_feat, aligned_feat, abs_diff, signed_diff, prod],
            dim=-1,
        )

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_dense_inputs(data)
        feat_t0 = self._encode_points(coord0, x_dense0, time_value=0.0)
        feat_t1 = self._encode_points(coord1, x_dense1, time_value=1.0)

        aligned_t1_for_t0 = self._point_soft_match(coord0, coord1, feat_t0, feat_t1)
        aligned_t0_for_t1 = self._point_soft_match(coord1, coord0, feat_t1, feat_t0)

        if self.signed_diff:
            point_feat_diff = feat_t1 - feat_t0
            recon_diff_t0 = feat_t0 - aligned_t1_for_t0
            recon_diff_t1 = feat_t1 - aligned_t0_for_t1
        else:
            point_feat_diff = torch.abs(feat_t1 - feat_t0)
            recon_diff_t0 = torch.abs(feat_t0 - aligned_t1_for_t0)
            recon_diff_t1 = torch.abs(feat_t1 - aligned_t0_for_t1)

        change_t0 = self.change_fusion_t0(
            self._pair_features(feat_t0, aligned_t1_for_t0)
        )
        change_t1 = self.change_fusion_t1(
            self._pair_features(feat_t1, aligned_t0_for_t1)
        )
        change_features = [change_t0, change_t1]
        if not self.direction_aware:
            change_features = [0.5 * (change_t0 + change_t1)]
        point_input = torch.cat(
            [
                feat_t0,
                feat_t1,
                point_feat_diff,
                recon_diff_t0,
                recon_diff_t1,
                *change_features,
                x_dense0,
                x_dense1,
            ],
            dim=-1,
        )
        return self._format_point_output(point_input)


class PTV3SiameseConcatPoint(PTV3DirectionFusionPoint):
    """Siamese PTv3 baseline without cross-temporal correspondence."""

    def __init__(
        self,
        num_classes=3,
        match_k=8,
        match_tau_feat=0.1,
        match_tau_geo=1.0,
        signed_diff=True,
        ptv3_order=("z", "z-trans"),
        ptv3_patch_size=512,
        me_cpt_root=osp.join(osp.dirname(osp.abspath(__file__)), "vendor", "ME-CPT"),
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            match_k=match_k,
            match_tau_feat=match_tau_feat,
            match_tau_geo=match_tau_geo,
            signed_diff=signed_diff,
            direction_aware=False,
            ptv3_order=ptv3_order,
            ptv3_patch_size=ptv3_patch_size,
            me_cpt_root=me_cpt_root,
        )
        channels = 64
        head_in_dim = channels * 4 + 12
        self.point_head = nn.Sequential(
            nn.Linear(head_in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.binary_change_head = None
        self.removal_like_head = None

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_dense_inputs(data)
        feat_t0 = self._encode_points(coord0, x_dense0, time_value=0.0)
        feat_t1 = self._encode_points(coord1, x_dense1, time_value=1.0)
        direct_diff = feat_t1 - feat_t0
        if not self.signed_diff:
            direct_diff = direct_diff.abs()
        point_input = torch.cat(
            [
                feat_t0,
                feat_t1,
                direct_diff,
                direct_diff.abs(),
                x_dense0,
                x_dense1,
            ],
            dim=-1,
        )
        return self.point_head(point_input)


class PTV3MatchingEvidencePoint(PTV3DirectionFusionPoint):
    """Bidirectional XYZ matching baseline without a learned relation branch."""

    def __init__(
        self,
        num_classes=3,
        match_k=8,
        match_tau_feat=0.1,
        match_tau_geo=1.0,
        signed_diff=True,
        ptv3_order=("z", "z-trans"),
        ptv3_patch_size=512,
        me_cpt_root=osp.join(osp.dirname(osp.abspath(__file__)), "vendor", "ME-CPT"),
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            match_k=match_k,
            match_tau_feat=match_tau_feat,
            match_tau_geo=match_tau_geo,
            signed_diff=signed_diff,
            direction_aware=False,
            ptv3_order=ptv3_order,
            ptv3_patch_size=ptv3_patch_size,
            me_cpt_root=me_cpt_root,
        )
        channels = 64
        head_in_dim = channels * 7 + 12
        self.point_head = nn.Sequential(
            nn.Linear(head_in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.binary_change_head = None
        self.removal_like_head = None

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_dense_inputs(data)
        feat_t0 = self._encode_points(coord0, x_dense0, time_value=0.0)
        feat_t1 = self._encode_points(coord1, x_dense1, time_value=1.0)

        aligned_t1_for_t0 = self._point_soft_match(
            coord0, coord1, feat_t0, feat_t1
        )
        aligned_t0_for_t1 = self._point_soft_match(
            coord1, coord0, feat_t1, feat_t0
        )
        residual_t0 = feat_t0 - aligned_t1_for_t0
        residual_t1 = feat_t1 - aligned_t0_for_t1
        direct_diff = feat_t1 - feat_t0
        if not self.signed_diff:
            direct_diff = direct_diff.abs()
            residual_t0 = residual_t0.abs()
            residual_t1 = residual_t1.abs()

        point_input = torch.cat(
            [
                feat_t0,
                feat_t1,
                direct_diff.abs(),
                residual_t0,
                residual_t1,
                residual_t0.abs(),
                residual_t1.abs(),
                x_dense0,
                x_dense1,
            ],
            dim=-1,
        )
        return self.point_head(point_input)


class PTV3RelationFusionPoint(PTV3DirectionFusionPoint):
    """
    Class-agnostic direction-conditioned cross-temporal relation fusion.

    The fusion module is independent of dataset label taxonomy. It decomposes
    bidirectional correspondence evidence into:
      1) symmetric consistency,
      2) directional discrepancy,
      3) local structural variation.
    A dataset-specific prediction head maps the universal relation feature to
    the required number of output classes.
    """

    def __init__(
        self,
        num_classes=3,
        match_k=16,
        match_tau_feat=0.1,
        match_tau_geo=1.0,
        signed_diff=True,
        ptv3_order=("z", "z-trans", "hilbert", "hilbert-trans"),
        ptv3_patch_size=512,
        use_binary_change_auxiliary_loss=False,
        use_removal_like_auxiliary_loss=False,
        me_cpt_root=osp.join(osp.dirname(osp.abspath(__file__)), "vendor", "ME-CPT"),
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            match_k=match_k,
            match_tau_feat=match_tau_feat,
            match_tau_geo=match_tau_geo,
            signed_diff=signed_diff,
            direction_aware=False,
            ptv3_order=ptv3_order,
            ptv3_patch_size=ptv3_patch_size,
            use_binary_change_auxiliary_loss=use_binary_change_auxiliary_loss,
            use_removal_like_auxiliary_loss=use_removal_like_auxiliary_loss,
            me_cpt_root=me_cpt_root,
        )
        channels = 64
        self.symmetric_encoder = nn.Sequential(
            nn.Linear(channels * 4, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.directional_encoder = nn.Sequential(
            nn.Linear(channels * 4, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.structural_encoder = nn.Sequential(
            nn.Linear(channels * 4 + 4, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.relation_gate = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.GELU(),
            nn.Linear(channels, 3),
        )
        self.relation_refine = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        head_in_dim = channels * 4 + 12
        self.point_head = nn.Sequential(
            nn.Linear(head_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        self.binary_change_head = self._make_binary_change_head(head_in_dim)
        self.removal_like_head = self._make_removal_like_head(head_in_dim)

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_dense_inputs(data)
        feat_t0 = self._encode_points(coord0, x_dense0, time_value=0.0)
        feat_t1 = self._encode_points(coord1, x_dense1, time_value=1.0)

        aligned_t1_for_t0 = self._point_soft_match(coord0, coord1, feat_t0, feat_t1)
        aligned_t0_for_t1 = self._point_soft_match(coord1, coord0, feat_t1, feat_t0)

        diff_0 = feat_t0 - aligned_t1_for_t0
        diff_1 = feat_t1 - aligned_t0_for_t1
        abs_diff_0 = diff_0.abs()
        abs_diff_1 = diff_1.abs()

        symmetric_input = torch.cat(
            [
                0.5 * (feat_t0 + feat_t1),
                torch.abs(feat_t0 - feat_t1),
                0.5 * (abs_diff_0 + abs_diff_1),
                feat_t0 * feat_t1,
            ],
            dim=-1,
        )
        symmetric_relation = self.symmetric_encoder(symmetric_input)

        directional_input = torch.cat(
            [
                diff_0,
                diff_1,
                diff_0 - diff_1,
                torch.abs(diff_0 - diff_1),
            ],
            dim=-1,
        )
        directional_relation = self.directional_encoder(directional_input)

        local_var0 = torch.abs(feat_t0 - aligned_t1_for_t0)
        local_var1 = torch.abs(feat_t1 - aligned_t0_for_t1)
        geo_residual = coord1 - coord0
        geo_distance = geo_residual.norm(dim=-1, keepdim=True)
        structural_input = torch.cat(
            [
                local_var0,
                local_var1,
                torch.abs(local_var0 - local_var1),
                0.5 * (local_var0 + local_var1),
                geo_residual,
                geo_distance,
            ],
            dim=-1,
        )
        structural_relation = self.structural_encoder(structural_input)

        relation_stack = torch.stack(
            [symmetric_relation, directional_relation, structural_relation],
            dim=-2,
        )
        gate_input = torch.cat(
            [symmetric_relation, directional_relation, structural_relation],
            dim=-1,
        )
        gate = torch.softmax(self.relation_gate(gate_input), dim=-1).unsqueeze(-1)
        universal_relation = (gate * relation_stack).sum(dim=-2)
        universal_relation = universal_relation + self.relation_refine(gate_input)

        if self.signed_diff:
            point_feat_diff = feat_t1 - feat_t0
        else:
            point_feat_diff = torch.abs(feat_t1 - feat_t0)
        point_input = torch.cat(
            [
                feat_t0,
                feat_t1,
                point_feat_diff,
                universal_relation,
                x_dense0,
                x_dense1,
            ],
            dim=-1,
        )
        return self._format_point_output(point_input)


class PTV3RelationEncoderPoint(PTV3DirectionFusionPoint):
    """
    Unified class-agnostic cross-temporal relation encoder.

    Instead of splitting evidence into fixed handcrafted branches, this module
    constructs generic correspondence cues and lets a lightweight encoder learn
    the relation representation. The final classifier still receives rich
    point-level temporal and correspondence context for precise localization.
    """

    def __init__(
        self,
        num_classes=3,
        match_k=16,
        match_tau_feat=0.1,
        match_tau_geo=1.0,
        signed_diff=True,
        ptv3_order=("z", "z-trans", "hilbert", "hilbert-trans"),
        ptv3_patch_size=512,
        use_binary_change_auxiliary_loss=False,
        use_removal_like_auxiliary_loss=False,
        me_cpt_root=osp.join(osp.dirname(osp.abspath(__file__)), "vendor", "ME-CPT"),
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            match_k=match_k,
            match_tau_feat=match_tau_feat,
            match_tau_geo=match_tau_geo,
            signed_diff=signed_diff,
            direction_aware=False,
            ptv3_order=ptv3_order,
            ptv3_patch_size=ptv3_patch_size,
            use_binary_change_auxiliary_loss=use_binary_change_auxiliary_loss,
            use_removal_like_auxiliary_loss=use_removal_like_auxiliary_loss,
            me_cpt_root=me_cpt_root,
        )
        channels = 64
        cue_dim = channels * 11 + 4
        self.relation_encoder = nn.Sequential(
            nn.Linear(cue_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.relation_gate = nn.Sequential(
            nn.Linear(cue_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        head_in_dim = channels * 8 + 12
        self.point_head = nn.Sequential(
            nn.Linear(head_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        self.binary_change_head = self._make_binary_change_head(head_in_dim)
        self.removal_like_head = self._make_removal_like_head(head_in_dim)

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_dense_inputs(data)
        feat_t0 = self._encode_points(coord0, x_dense0, time_value=0.0)
        feat_t1 = self._encode_points(coord1, x_dense1, time_value=1.0)

        aligned_t1_for_t0 = self._point_soft_match(coord0, coord1, feat_t0, feat_t1)
        aligned_t0_for_t1 = self._point_soft_match(coord1, coord0, feat_t1, feat_t0)

        residual_t0 = feat_t0 - aligned_t1_for_t0
        residual_t1 = feat_t1 - aligned_t0_for_t1
        direct_diff = feat_t1 - feat_t0 if self.signed_diff else torch.abs(feat_t1 - feat_t0)
        geo_residual = coord1 - coord0
        geo_distance = geo_residual.norm(dim=-1, keepdim=True)
        relation_cues = torch.cat(
            [
                feat_t0,
                feat_t1,
                aligned_t1_for_t0,
                aligned_t0_for_t1,
                residual_t0,
                residual_t1,
                residual_t0.abs(),
                residual_t1.abs(),
                direct_diff,
                direct_diff.abs(),
                feat_t0 * feat_t1,
                geo_residual,
                geo_distance,
            ],
            dim=-1,
        )
        relation = self.relation_gate(relation_cues) * self.relation_encoder(relation_cues)
        point_input = torch.cat(
            [
                feat_t0,
                feat_t1,
                direct_diff.abs(),
                residual_t0,
                residual_t1,
                residual_t0.abs(),
                residual_t1.abs(),
                relation,
                x_dense0,
                x_dense1,
            ],
            dim=-1,
        )
        return self._format_point_output(point_input)


class PTV3RelationAttentionPoint(PTV3DirectionFusionPoint):
    """
    Correspondence-prior relation attention ablation.

    Pipeline:
      PTv3 point encoding for T0/T1 separately
      -> bidirectional local soft correspondence prior
      -> gated relation-attention feature update
      -> shared no-direction change representation
      -> dataset-specific point prediction

    Matching is used to guide feature learning through an adaptive update,
    instead of being only concatenated as late classifier evidence.
    """

    def __init__(
        self,
        num_classes=3,
        match_k=16,
        match_tau_feat=0.1,
        match_tau_geo=1.0,
        signed_diff=True,
        ptv3_order=("z", "z-trans", "hilbert", "hilbert-trans"),
        ptv3_patch_size=512,
        use_binary_change_auxiliary_loss=False,
        use_removal_like_auxiliary_loss=False,
        use_semantic_auxiliary_loss=False,
        semantic_num_classes=4,
        use_symmetric_xyz_relation=True,
        use_prior_guided_interaction=True,
        me_cpt_root=osp.join(osp.dirname(osp.abspath(__file__)), "vendor", "ME-CPT"),
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            match_k=match_k,
            match_tau_feat=match_tau_feat,
            match_tau_geo=match_tau_geo,
            signed_diff=signed_diff,
            direction_aware=False,
            ptv3_order=ptv3_order,
            ptv3_patch_size=ptv3_patch_size,
            use_binary_change_auxiliary_loss=use_binary_change_auxiliary_loss,
            use_removal_like_auxiliary_loss=use_removal_like_auxiliary_loss,
            me_cpt_root=me_cpt_root,
        )
        channels = 64
        self.use_symmetric_xyz_relation = use_symmetric_xyz_relation
        self.use_prior_guided_interaction = use_prior_guided_interaction
        self.match_prior_dim = self.match_k + 3
        relation_in_dim = channels * 5
        if self.use_prior_guided_interaction:
            relation_in_dim += self.match_prior_dim
        self.relation_message = nn.Sequential(
            nn.Linear(relation_in_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels, channels),
        )
        self.relation_gate = nn.Sequential(
            nn.Linear(relation_in_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.update_norm = nn.LayerNorm(channels)
        relation_blocks = 6 if self.use_symmetric_xyz_relation else 5
        head_in_dim = channels * relation_blocks + 12
        self.point_head = nn.Sequential(
            nn.Linear(head_in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.binary_change_head = self._make_binary_change_head(head_in_dim)
        self.removal_like_head = self._make_removal_like_head(head_in_dim)
        self.semantic_head = None
        if use_semantic_auxiliary_loss:
            self.semantic_head = nn.Sequential(
                nn.Linear(channels, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, semantic_num_classes),
            )

    def _pad_match_values(self, values):
        if values.size(-1) == self.match_k:
            return values
        pad = self.match_k - values.size(-1)
        if pad <= 0:
            return values[..., : self.match_k]
        return F.pad(values, (0, pad), value=0.0)

    def _build_match_prior(self, match_info):
        weight = self._pad_match_values(match_info["weight"])
        knn_dist = self._pad_match_values(match_info["knn_dist"])
        confidence = weight.max(dim=-1, keepdim=True).values
        denom = math.log(max(weight.size(-1), 2))
        entropy = -(weight * torch.log(weight.clamp_min(1e-8))).sum(
            dim=-1,
            keepdim=True,
        ) / denom
        expected_distance = (weight * knn_dist).sum(dim=-1, keepdim=True)
        # Ablate explicit prior information without changing model capacity or
        # the soft correspondence used to align features.
        mode = getattr(self, "correspondence_prior", "full")
        valid_modes = (
            "confidence",
            "distance",
            "confidence_entropy",
            "confidence_distance",
            "confidence_entropy_distance",
            "full",
        )
        if mode not in valid_modes:
            raise ValueError(f"Unknown correspondence prior: {mode}")
        if mode != "full":
            weight = torch.zeros_like(weight)
        if mode == "distance":
            confidence = torch.zeros_like(confidence)
        if mode not in ("confidence_entropy", "confidence_entropy_distance", "full"):
            entropy = torch.zeros_like(entropy)
        if mode not in ("distance", "confidence_distance", "confidence_entropy_distance", "full"):
            expected_distance = torch.zeros_like(expected_distance)
        return torch.cat([weight, confidence, entropy, expected_distance], dim=-1)

    def _relation_attention_update(self, query_feat, match_info):
        aligned_feat = match_info["aligned"]
        relation_features = [self._pair_features(query_feat, aligned_feat)]
        if self.use_prior_guided_interaction:
            relation_features.append(self._build_match_prior(match_info))
        relation_input = torch.cat(relation_features, dim=-1)
        message = self.relation_message(relation_input)
        gate = self.relation_gate(relation_input)
        return self.update_norm(query_feat + gate * message)

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_dense_inputs(data)
        feat_t0 = self._encode_points(coord0, x_dense0, time_value=0.0)
        feat_t1 = self._encode_points(coord1, x_dense1, time_value=1.0)

        match_t1_for_t0 = self._point_soft_match_info(coord0, coord1, feat_t0, feat_t1)
        match_t0_for_t1 = self._point_soft_match_info(coord1, coord0, feat_t1, feat_t0)
        aligned_t1_for_t0 = match_t1_for_t0["aligned"]
        aligned_t0_for_t1 = match_t0_for_t1["aligned"]

        updated_t0 = self._relation_attention_update(feat_t0, match_t1_for_t0)
        updated_t1 = self._relation_attention_update(feat_t1, match_t0_for_t1)

        if self.signed_diff:
            updated_diff = updated_t1 - updated_t0
            residual_t0 = updated_t0 - aligned_t1_for_t0
            residual_t1 = updated_t1 - aligned_t0_for_t1
        else:
            updated_diff = torch.abs(updated_t1 - updated_t0)
            residual_t0 = torch.abs(updated_t0 - aligned_t1_for_t0)
            residual_t1 = torch.abs(updated_t1 - aligned_t0_for_t1)

        relation_residual = 0.5 * (
            (
                F.normalize(updated_t0, dim=-1)
                - F.normalize(aligned_t1_for_t0, dim=-1)
            )
            .norm(dim=-1, keepdim=True)
            + (
                F.normalize(updated_t1, dim=-1)
                - F.normalize(aligned_t0_for_t1, dim=-1)
            )
            .norm(dim=-1, keepdim=True)
        )
        point_features = [
            updated_t0,
            updated_t1,
            updated_diff,
            residual_t0,
            residual_t1,
        ]
        if self.use_symmetric_xyz_relation:
            change_t0 = self.change_fusion_t0(
                self._pair_features(updated_t0, aligned_t1_for_t0)
            )
            change_t1 = self.change_fusion_t1(
                self._pair_features(updated_t1, aligned_t0_for_t1)
            )
            point_features.append(0.5 * (change_t0 + change_t1))
        point_features.extend([x_dense0, x_dense1])
        point_input = torch.cat(point_features, dim=-1)
        extra_outputs = {"relation_residual": relation_residual}
        if self.semantic_head is not None:
            extra_outputs["semantic_logits_t0"] = self.semantic_head(updated_t0)
            extra_outputs["semantic_logits_t1"] = self.semantic_head(updated_t1)
        return self._format_point_output(point_input, extra_outputs=extra_outputs)


class PTV3XYXYZRelationAttentionPoint(PTV3RelationAttentionPoint):
    """Semantic-free complementary XYZ/XY relation model.

    ``use_height_cues`` provides a controlled XYZ+XY versus
    XYZ+XY+height ablation. ``use_adaptive_relation_fusion`` switches
    between correspondence-conditioned gating and a fixed concat-MLP
    fusion for a direct adaptive-fusion ablation.
    """

    def __init__(
        self,
        num_classes=3,
        match_k=8,
        match_tau_feat=0.1,
        match_tau_geo=1.0,
        signed_diff=True,
        ptv3_order=("z", "z-trans"),
        ptv3_patch_size=512,
        use_binary_change_auxiliary_loss=False,
        use_removal_like_auxiliary_loss=False,
        use_height_cues=True,
        use_adaptive_relation_fusion=True,
        me_cpt_root=osp.join(osp.dirname(osp.abspath(__file__)), "vendor", "ME-CPT"),
        correspondence_prior="full",
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            match_k=match_k,
            match_tau_feat=match_tau_feat,
            match_tau_geo=match_tau_geo,
            signed_diff=signed_diff,
            ptv3_order=ptv3_order,
            ptv3_patch_size=ptv3_patch_size,
            use_binary_change_auxiliary_loss=use_binary_change_auxiliary_loss,
            use_removal_like_auxiliary_loss=use_removal_like_auxiliary_loss,
            use_semantic_auxiliary_loss=False,
            me_cpt_root=me_cpt_root,
        )
        channels = 64
        prior_channels = 32
        self.use_height_cues = use_height_cues
        self.correspondence_prior = correspondence_prior
        self.use_adaptive_relation_fusion = use_adaptive_relation_fusion
        self.xy_relation_encoder = nn.Sequential(
            nn.Linear(channels * 5, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.xy_direction_fusion = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        if self.use_adaptive_relation_fusion:
            self.dual_prior_encoder = nn.Sequential(
                nn.Linear(self.match_prior_dim, prior_channels),
                nn.LayerNorm(prior_channels),
                nn.GELU(),
            )
            self.xy_xyz_prior_fusion = nn.Sequential(
                nn.Linear(prior_channels * 4 + 4, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, prior_channels),
            )
            self.xy_xyz_message = nn.Sequential(
                nn.Linear(channels + 4, channels),
                nn.LayerNorm(channels),
                nn.GELU(),
                nn.Linear(channels, channels),
            )
            self.xy_xyz_gate = nn.Sequential(
                nn.Linear(channels * 2 + prior_channels + 4, channels),
                nn.LayerNorm(channels),
                nn.GELU(),
                nn.Linear(channels, channels),
                nn.Sigmoid(),
            )
            self.xy_xyz_norm = nn.LayerNorm(channels)
        else:
            self.fixed_xy_xyz_fusion = nn.Sequential(
                nn.Linear(channels * 2 + 4, channels),
                nn.LayerNorm(channels),
                nn.GELU(),
                nn.Linear(channels, channels),
                nn.LayerNorm(channels),
            )

    @staticmethod
    def _align_xy_value(key_value, match_info):
        batch_size = key_value.size(0)
        batch_idx = torch.arange(
            batch_size,
            device=key_value.device,
        ).view(batch_size, 1, 1)
        local_value = key_value[batch_idx, match_info["knn_idx"]]
        return (match_info["weight"].unsqueeze(-1) * local_value).sum(dim=2)

    @staticmethod
    def _height_cues(coord0, coord1, xy_match_t1_for_t0, xy_match_t0_for_t1):
        aligned_z1 = PTV3XYXYZRelationAttentionPoint._align_xy_value(
            coord1[..., 2:3], xy_match_t1_for_t0
        )
        aligned_z0 = PTV3XYXYZRelationAttentionPoint._align_xy_value(
            coord0[..., 2:3], xy_match_t0_for_t1
        )
        dz_t0 = aligned_z1 - coord0[..., 2:3]
        dz_t1 = coord1[..., 2:3] - aligned_z0
        return torch.cat(
            [
                torch.sign(dz_t0) * torch.log1p(torch.abs(dz_t0)),
                torch.log1p(torch.abs(dz_t0)),
                torch.sign(dz_t1) * torch.log1p(torch.abs(dz_t1)),
                torch.log1p(torch.abs(dz_t1)),
            ],
            dim=-1,
        )

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_dense_inputs(data)
        feat_t0 = self._encode_points(coord0, x_dense0, time_value=0.0)
        feat_t1 = self._encode_points(coord1, x_dense1, time_value=1.0)

        xyz_match_t1_for_t0 = self._point_soft_match_info(
            coord0, coord1, feat_t0, feat_t1
        )
        xyz_match_t0_for_t1 = self._point_soft_match_info(
            coord1, coord0, feat_t1, feat_t0
        )
        xy_match_t1_for_t0 = self._point_soft_match_info(
            coord0[..., :2], coord1[..., :2], feat_t0, feat_t1
        )
        xy_match_t0_for_t1 = self._point_soft_match_info(
            coord1[..., :2], coord0[..., :2], feat_t1, feat_t0
        )

        aligned_t1_for_t0 = xyz_match_t1_for_t0["aligned"]
        aligned_t0_for_t1 = xyz_match_t0_for_t1["aligned"]
        updated_t0 = self._relation_attention_update(feat_t0, xyz_match_t1_for_t0)
        updated_t1 = self._relation_attention_update(feat_t1, xyz_match_t0_for_t1)

        if self.signed_diff:
            updated_diff = updated_t1 - updated_t0
            residual_t0 = updated_t0 - aligned_t1_for_t0
            residual_t1 = updated_t1 - aligned_t0_for_t1
        else:
            updated_diff = torch.abs(updated_t1 - updated_t0)
            residual_t0 = torch.abs(updated_t0 - aligned_t1_for_t0)
            residual_t1 = torch.abs(updated_t1 - aligned_t0_for_t1)

        change_t0 = self.change_fusion_t0(
            self._pair_features(updated_t0, aligned_t1_for_t0)
        )
        change_t1 = self.change_fusion_t1(
            self._pair_features(updated_t1, aligned_t0_for_t1)
        )
        change_repr = 0.5 * (change_t0 + change_t1)

        aligned_xy_t1 = self._align_xy_value(updated_t1, xy_match_t1_for_t0)
        aligned_xy_t0 = self._align_xy_value(updated_t0, xy_match_t0_for_t1)
        xy_relation_t0 = self.xy_relation_encoder(
            self._pair_features(updated_t0, aligned_xy_t1)
        )
        xy_relation_t1 = self.xy_relation_encoder(
            self._pair_features(updated_t1, aligned_xy_t0)
        )
        xy_relation = self.xy_direction_fusion(
            torch.cat([xy_relation_t0, xy_relation_t1], dim=-1)
        )
        if self.use_height_cues:
            height_cues = self._height_cues(
                coord0,
                coord1,
                xy_match_t1_for_t0,
                xy_match_t0_for_t1,
            )
        else:
            height_cues = coord0.new_zeros(
                coord0.size(0),
                coord0.size(1),
                4,
            )
        if self.use_adaptive_relation_fusion:
            prior_embedding = self.xy_xyz_prior_fusion(
                torch.cat(
                    [
                        self.dual_prior_encoder(
                            self._build_match_prior(xyz_match_t1_for_t0)
                        ),
                        self.dual_prior_encoder(
                            self._build_match_prior(xyz_match_t0_for_t1)
                        ),
                        self.dual_prior_encoder(
                            self._build_match_prior(xy_match_t1_for_t0)
                        ),
                        self.dual_prior_encoder(
                            self._build_match_prior(xy_match_t0_for_t1)
                        ),
                        height_cues,
                    ],
                    dim=-1,
                )
            )
            dual_gate = self.xy_xyz_gate(
                torch.cat(
                    [change_repr, xy_relation, prior_embedding, height_cues],
                    dim=-1,
                )
            )
            dual_message = self.xy_xyz_message(
                torch.cat([xy_relation, height_cues], dim=-1)
            )
            fused_change_repr = self.xy_xyz_norm(
                change_repr + dual_gate * dual_message
            )
        else:
            dual_gate = None
            fused_change_repr = self.fixed_xy_xyz_fusion(
                torch.cat([change_repr, xy_relation, height_cues], dim=-1)
            )

        relation_residual = 0.5 * (
            (
                F.normalize(updated_t0, dim=-1)
                - F.normalize(aligned_t1_for_t0, dim=-1)
            ).norm(dim=-1, keepdim=True)
            + (
                F.normalize(updated_t1, dim=-1)
                - F.normalize(aligned_t0_for_t1, dim=-1)
            ).norm(dim=-1, keepdim=True)
        )
        point_input = torch.cat(
            [
                updated_t0,
                updated_t1,
                updated_diff,
                residual_t0,
                residual_t1,
                fused_change_repr,
                x_dense0,
                x_dense1,
            ],
            dim=-1,
        )
        extra_outputs = {"relation_residual": relation_residual}
        if dual_gate is not None:
            extra_outputs["xy_xyz_gate"] = dual_gate
        return self._format_point_output(point_input, extra_outputs=extra_outputs)


class PTV3SemanticRelationAttentionPoint(PTV3RelationAttentionPoint):
    """Relation attention with lightweight semantic-conditioned change fusion."""

    def __init__(
        self,
        num_classes=7,
        match_k=8,
        match_tau_feat=0.1,
        match_tau_geo=1.0,
        signed_diff=True,
        ptv3_order=("z", "z-trans", "hilbert", "hilbert-trans"),
        ptv3_patch_size=1024,
        use_binary_change_auxiliary_loss=False,
        use_removal_like_auxiliary_loss=False,
        semantic_num_classes=4,
        use_xy_xyz_prior=False,
        use_directional_semantic_transition=False,
        use_ordered_semantic_probability_transition=False,
        use_residual_ordered_semantic_transition=False,
        use_adaptive_semantic_spatial_fusion=True,
        prior_injection_mode="p5_full",
        me_cpt_root=osp.join(osp.dirname(osp.abspath(__file__)), "vendor", "ME-CPT"),
    ) -> None:
        valid_prior_modes = {
            "p1_no_prior",
            "p2_late_concat",
            "p3_interaction_only",
            "p4_fusion_only",
            "p5_full",
        }
        if prior_injection_mode not in valid_prior_modes:
            raise ValueError(
                f"Unknown prior injection mode: {prior_injection_mode}. "
                f"Expected one of {sorted(valid_prior_modes)}."
            )
        use_prior_guided_interaction = prior_injection_mode in {
            "p3_interaction_only",
            "p5_full",
        }
        super().__init__(
            num_classes=num_classes,
            match_k=match_k,
            match_tau_feat=match_tau_feat,
            match_tau_geo=match_tau_geo,
            signed_diff=signed_diff,
            ptv3_order=ptv3_order,
            ptv3_patch_size=ptv3_patch_size,
            use_binary_change_auxiliary_loss=use_binary_change_auxiliary_loss,
            use_removal_like_auxiliary_loss=use_removal_like_auxiliary_loss,
            use_semantic_auxiliary_loss=False,
            semantic_num_classes=semantic_num_classes,
            use_prior_guided_interaction=use_prior_guided_interaction,
            me_cpt_root=me_cpt_root,
        )
        channels = 64
        semantic_channels = 32
        prior_channels = 32
        self.prior_injection_mode = prior_injection_mode
        self.use_prior_conditioned_fusion = prior_injection_mode in {
            "p4_fusion_only",
            "p5_full",
        }
        self.use_late_prior_concatenation = (
            prior_injection_mode == "p2_late_concat"
        )
        self.requires_prior_embedding = (
            self.use_prior_conditioned_fusion
            or self.use_late_prior_concatenation
        )
        self.use_xy_xyz_prior = use_xy_xyz_prior
        self.use_adaptive_semantic_spatial_fusion = (
            use_adaptive_semantic_spatial_fusion
        )
        if (
            not self.use_adaptive_semantic_spatial_fusion
            and not self.use_xy_xyz_prior
        ):
            raise ValueError(
                "Fixed semantic-spatial concatenation requires XYZ/XY relations."
            )
        self.semantic_feature_head = nn.Sequential(
            nn.Linear(channels, semantic_channels),
            nn.LayerNorm(semantic_channels),
            nn.GELU(),
        )
        self.semantic_classifier = nn.Linear(
            semantic_channels,
            semantic_num_classes,
        )
        self.semantic_relation = nn.Sequential(
            nn.Linear(semantic_channels * 3, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        if self.use_adaptive_semantic_spatial_fusion:
            self.semantic_prior_encoder = nn.Sequential(
                nn.Linear(self.match_prior_dim, prior_channels),
                nn.LayerNorm(prior_channels),
                nn.GELU(),
            )
            self.semantic_change_message = nn.Sequential(
                nn.Linear(channels, channels),
                nn.LayerNorm(channels),
                nn.GELU(),
                nn.Linear(channels, channels),
            )
            self.semantic_change_gate = nn.Sequential(
                nn.Linear(channels * 2 + prior_channels, channels),
                nn.LayerNorm(channels),
                nn.GELU(),
                nn.Linear(channels, channels),
                nn.Sigmoid(),
            )
            self.semantic_change_norm = nn.LayerNorm(channels)
        self.use_ordered_semantic_probability_transition = (
            use_ordered_semantic_probability_transition
        )
        self.use_residual_ordered_semantic_transition = (
            use_residual_ordered_semantic_transition
        )
        if (
            self.use_residual_ordered_semantic_transition
            and not self.use_ordered_semantic_probability_transition
        ):
            raise ValueError(
                "Residual ordered transition requires ordered semantic probability transition."
            )
        if self.use_ordered_semantic_probability_transition and not use_xy_xyz_prior:
            raise ValueError(
                "Ordered semantic probability transition requires XY correspondence."
            )
        if (
            self.use_ordered_semantic_probability_transition
            and not self.use_adaptive_semantic_spatial_fusion
        ):
            raise ValueError(
                "Ordered semantic transition requires adaptive semantic-spatial fusion."
            )
        self.use_directional_semantic_transition = (
            use_directional_semantic_transition
        )
        if use_directional_semantic_transition:
            self.directional_semantic_relation = nn.Sequential(
                nn.Linear(semantic_channels * 5, channels),
                nn.LayerNorm(channels),
                nn.GELU(),
                nn.Linear(channels, channels),
            )
            self.semantic_direction_fusion = nn.Sequential(
                nn.Linear(channels * 2, channels),
                nn.LayerNorm(channels),
                nn.GELU(),
                nn.Linear(channels, channels),
            )
        if use_xy_xyz_prior:
            self.xy_relation_encoder = nn.Sequential(
                nn.Linear(channels * 5, channels),
                nn.LayerNorm(channels),
                nn.GELU(),
                nn.Linear(channels, channels),
            )
            self.xy_direction_fusion = nn.Sequential(
                nn.Linear(channels * 2, channels),
                nn.LayerNorm(channels),
                nn.GELU(),
                nn.Linear(channels, channels),
            )
            if self.use_adaptive_semantic_spatial_fusion:
                dual_prior_in_dim = 32 * 4 + 4
                if self.requires_prior_embedding:
                    self.xy_xyz_prior_encoder = nn.Sequential(
                        nn.Linear(dual_prior_in_dim, 64),
                        nn.LayerNorm(64),
                        nn.GELU(),
                        nn.Linear(64, 32),
                    )
                self.xy_xyz_change_message = nn.Sequential(
                    nn.Linear(channels * 2 + 4, channels),
                    nn.LayerNorm(channels),
                    nn.GELU(),
                    nn.Linear(channels, channels),
                )
                ordered_transition_channels = (
                    semantic_channels
                    if (
                        use_ordered_semantic_probability_transition
                        and not use_residual_ordered_semantic_transition
                    )
                    else 0
                )
                fusion_prior_channels = (
                    prior_channels if self.use_prior_conditioned_fusion else 0
                )
                self.xy_xyz_change_gate = nn.Sequential(
                    nn.Linear(
                        channels * 3
                        + ordered_transition_channels
                        + fusion_prior_channels
                        + 4,
                        channels,
                    ),
                    nn.LayerNorm(channels),
                    nn.GELU(),
                    nn.Linear(channels, channels),
                    nn.Sigmoid(),
                )
                self.xy_xyz_change_norm = nn.LayerNorm(channels)
                if self.use_late_prior_concatenation:
                    self.late_prior_fusion = nn.Sequential(
                        nn.Linear(channels + prior_channels, channels),
                        nn.LayerNorm(channels),
                        nn.GELU(),
                        nn.Linear(channels, channels),
                    )
            if use_ordered_semantic_probability_transition:
                self.ordered_semantic_transition_encoder = nn.Sequential(
                    nn.Linear(semantic_num_classes * 3, semantic_channels),
                    nn.LayerNorm(semantic_channels),
                    nn.GELU(),
                    nn.Linear(semantic_channels, semantic_channels),
                )
            if use_residual_ordered_semantic_transition:
                self.ordered_transition_residual_adapter = nn.Sequential(
                    nn.Linear(semantic_channels, channels),
                    nn.LayerNorm(channels),
                    nn.GELU(),
                    nn.Linear(channels, channels),
                )
                self.ordered_transition_residual_gate = nn.Sequential(
                    nn.Linear(
                        channels + semantic_channels + prior_channels,
                        channels,
                    ),
                    nn.LayerNorm(channels),
                    nn.GELU(),
                    nn.Linear(channels, channels),
                    nn.Sigmoid(),
                )
                self.ordered_transition_residual_scale = nn.Parameter(
                    torch.zeros(1)
                )
        if not self.use_adaptive_semantic_spatial_fusion:
            self.fixed_semantic_spatial_fusion = nn.Sequential(
                nn.Linear(channels * 3 + 4, channels),
                nn.LayerNorm(channels),
                nn.GELU(),
                nn.Linear(channels, channels),
            )

    @staticmethod
    def _align_embedding(key_embedding, match_info):
        batch_size = key_embedding.size(0)
        batch_idx = torch.arange(
            batch_size,
            device=key_embedding.device,
        ).view(batch_size, 1, 1)
        local_embedding = key_embedding[batch_idx, match_info["knn_idx"]]
        return (
            match_info["weight"].unsqueeze(-1) * local_embedding
        ).sum(dim=2)

    def _semantic_transition(self, query_embedding, aligned_embedding):
        return self.semantic_relation(
            torch.cat(
                [
                    query_embedding,
                    aligned_embedding,
                    torch.abs(query_embedding - aligned_embedding),
                ],
                dim=-1,
            )
        )

    def _directional_semantic_transition(
        self,
        query_embedding,
        aligned_embedding,
    ):
        signed_difference = aligned_embedding - query_embedding
        return self.directional_semantic_relation(
            torch.cat(
                [
                    query_embedding,
                    aligned_embedding,
                    signed_difference,
                    torch.abs(signed_difference),
                    query_embedding * aligned_embedding,
                ],
                dim=-1,
            )
        )

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_dense_inputs(data)
        feat_t0 = self._encode_points(coord0, x_dense0, time_value=0.0)
        feat_t1 = self._encode_points(coord1, x_dense1, time_value=1.0)

        match_t1_for_t0 = self._point_soft_match_info(
            coord0, coord1, feat_t0, feat_t1
        )
        match_t0_for_t1 = self._point_soft_match_info(
            coord1, coord0, feat_t1, feat_t0
        )
        xy_match_t1_for_t0 = None
        xy_match_t0_for_t1 = None
        if self.use_xy_xyz_prior:
            xy_match_t1_for_t0 = self._point_soft_match_info(
                coord0[..., :2], coord1[..., :2], feat_t0, feat_t1
            )
            xy_match_t0_for_t1 = self._point_soft_match_info(
                coord1[..., :2], coord0[..., :2], feat_t1, feat_t0
            )
        aligned_t1_for_t0 = match_t1_for_t0["aligned"]
        aligned_t0_for_t1 = match_t0_for_t1["aligned"]
        updated_t0 = self._relation_attention_update(feat_t0, match_t1_for_t0)
        updated_t1 = self._relation_attention_update(feat_t1, match_t0_for_t1)

        semantic_t0 = self.semantic_feature_head(updated_t0)
        semantic_t1 = self.semantic_feature_head(updated_t1)
        semantic_logits_t0 = self.semantic_classifier(semantic_t0)
        semantic_logits_t1 = self.semantic_classifier(semantic_t1)
        aligned_semantic_t1 = self._align_embedding(
            semantic_t1, match_t1_for_t0
        )
        aligned_semantic_t0 = self._align_embedding(
            semantic_t0, match_t0_for_t1
        )
        if self.use_directional_semantic_transition:
            semantic_relation_t0_to_t1 = self._directional_semantic_transition(
                semantic_t0,
                aligned_semantic_t1,
            )
            semantic_relation_t1_to_t0 = self._directional_semantic_transition(
                semantic_t1,
                aligned_semantic_t0,
            )
            semantic_relation = self.semantic_direction_fusion(
                torch.cat(
                    [
                        semantic_relation_t0_to_t1,
                        semantic_relation_t1_to_t0,
                    ],
                    dim=-1,
                )
            )
        else:
            semantic_relation = 0.5 * (
                self._semantic_transition(semantic_t0, aligned_semantic_t1)
                + self._semantic_transition(semantic_t1, aligned_semantic_t0)
            )

        if self.signed_diff:
            updated_diff = updated_t1 - updated_t0
            residual_t0 = updated_t0 - aligned_t1_for_t0
            residual_t1 = updated_t1 - aligned_t0_for_t1
        else:
            updated_diff = torch.abs(updated_t1 - updated_t0)
            residual_t0 = torch.abs(updated_t0 - aligned_t1_for_t0)
            residual_t1 = torch.abs(updated_t1 - aligned_t0_for_t1)

        change_t0 = self.change_fusion_t0(
            self._pair_features(updated_t0, aligned_t1_for_t0)
        )
        change_t1 = self.change_fusion_t1(
            self._pair_features(updated_t1, aligned_t0_for_t1)
        )
        change_repr = 0.5 * (change_t0 + change_t1)
        xyz_prior_t0 = None
        xyz_prior_t1 = None
        if (
            self.use_adaptive_semantic_spatial_fusion
            and self.requires_prior_embedding
        ):
            xyz_prior_t0 = self.semantic_prior_encoder(
                self._build_match_prior(match_t1_for_t0)
            )
            xyz_prior_t1 = self.semantic_prior_encoder(
                self._build_match_prior(match_t0_for_t1)
            )
        ordered_semantic_transition = None
        ordered_transition_residual_gate = None
        if self.use_xy_xyz_prior:
            aligned_xy_t1 = self._align_embedding(
                updated_t1, xy_match_t1_for_t0
            )
            aligned_xy_t0 = self._align_embedding(
                updated_t0, xy_match_t0_for_t1
            )
            xy_relation_t0 = self.xy_relation_encoder(
                self._pair_features(updated_t0, aligned_xy_t1)
            )
            xy_relation_t1 = self.xy_relation_encoder(
                self._pair_features(updated_t1, aligned_xy_t0)
            )
            xy_relation = self.xy_direction_fusion(
                torch.cat([xy_relation_t0, xy_relation_t1], dim=-1)
            )
            if self.use_ordered_semantic_probability_transition:
                semantic_probability_t0 = torch.softmax(
                    semantic_logits_t0, dim=-1
                )
                semantic_probability_t1 = torch.softmax(
                    semantic_logits_t1, dim=-1
                )
                if self.use_residual_ordered_semantic_transition:
                    semantic_probability_t0 = semantic_probability_t0.detach()
                    semantic_probability_t1 = semantic_probability_t1.detach()
                aligned_probability_t0 = self._align_embedding(
                    semantic_probability_t0, xy_match_t0_for_t1
                )
                ordered_semantic_transition = (
                    self.ordered_semantic_transition_encoder(
                        torch.cat(
                            [
                                aligned_probability_t0,
                                semantic_probability_t1,
                                semantic_probability_t1
                                - aligned_probability_t0,
                            ],
                            dim=-1,
                        )
                    )
                )
            aligned_xy_z1 = self._align_embedding(
                coord1[..., 2:3], xy_match_t1_for_t0
            )
            aligned_xy_z0 = self._align_embedding(
                coord0[..., 2:3], xy_match_t0_for_t1
            )
            dz_t0 = aligned_xy_z1 - coord0[..., 2:3]
            dz_t1 = coord1[..., 2:3] - aligned_xy_z0
            signed_log_dz_t0 = torch.sign(dz_t0) * torch.log1p(torch.abs(dz_t0))
            signed_log_dz_t1 = torch.sign(dz_t1) * torch.log1p(torch.abs(dz_t1))
            height_cues = torch.cat(
                [
                    signed_log_dz_t0,
                    torch.log1p(torch.abs(dz_t0)),
                    signed_log_dz_t1,
                    torch.log1p(torch.abs(dz_t1)),
                ],
                dim=-1,
            )
            semantic_gate = None
            fused_change_repr = None
            if self.use_adaptive_semantic_spatial_fusion:
                prior_embedding = None
                if self.requires_prior_embedding:
                    xy_prior_t0 = self.semantic_prior_encoder(
                        self._build_match_prior(xy_match_t1_for_t0)
                    )
                    xy_prior_t1 = self.semantic_prior_encoder(
                        self._build_match_prior(xy_match_t0_for_t1)
                    )
                    prior_embedding = self.xy_xyz_prior_encoder(
                        torch.cat(
                            [
                                xyz_prior_t0,
                                xyz_prior_t1,
                                xy_prior_t0,
                                xy_prior_t1,
                                height_cues,
                            ],
                            dim=-1,
                        )
                    )
                gate_features = [
                    change_repr,
                    semantic_relation,
                    xy_relation,
                ]
                message_features = [semantic_relation, xy_relation]
                if (
                    ordered_semantic_transition is not None
                    and not self.use_residual_ordered_semantic_transition
                ):
                    gate_features.append(ordered_semantic_transition)
                if self.use_prior_conditioned_fusion:
                    gate_features.append(prior_embedding)
                gate_features.append(height_cues)
                message_features.append(height_cues)
                semantic_gate = self.xy_xyz_change_gate(
                    torch.cat(gate_features, dim=-1)
                )
                semantic_message = self.xy_xyz_change_message(
                    torch.cat(message_features, dim=-1)
                )
                base_change_repr = self.xy_xyz_change_norm(
                    change_repr + semantic_gate * semantic_message
                )
                fused_change_repr = base_change_repr
                if self.use_late_prior_concatenation:
                    fused_change_repr = self.late_prior_fusion(
                        torch.cat(
                            [fused_change_repr, prior_embedding],
                            dim=-1,
                        )
                    )
                if self.use_residual_ordered_semantic_transition:
                    ordered_transition_residual_gate = (
                        self.ordered_transition_residual_gate(
                            torch.cat(
                                [
                                    base_change_repr,
                                    ordered_semantic_transition,
                                    prior_embedding,
                                ],
                                dim=-1,
                            )
                        )
                    )
                    ordered_transition_delta = (
                        self.ordered_transition_residual_adapter(
                            ordered_semantic_transition
                        )
                    )
                    residual_scale = torch.tanh(
                        self.ordered_transition_residual_scale
                    )
                    fused_change_repr = (
                        base_change_repr
                        + residual_scale
                        * ordered_transition_residual_gate
                        * ordered_transition_delta
                    )
        else:
            prior_embedding = 0.5 * (xyz_prior_t0 + xyz_prior_t1)
            semantic_gate = self.semantic_change_gate(
                torch.cat(
                    [change_repr, semantic_relation, prior_embedding],
                    dim=-1,
                )
            )
            fused_change_repr = self.semantic_change_norm(
                change_repr
                + semantic_gate * self.semantic_change_message(semantic_relation)
            )

        relation_residual = 0.5 * (
            (
                F.normalize(updated_t0, dim=-1)
                - F.normalize(aligned_t1_for_t0, dim=-1)
            ).norm(dim=-1, keepdim=True)
            + (
                F.normalize(updated_t1, dim=-1)
                - F.normalize(aligned_t0_for_t1, dim=-1)
            ).norm(dim=-1, keepdim=True)
        )
        if self.use_adaptive_semantic_spatial_fusion:
            point_features = [
                updated_t0,
                updated_t1,
                updated_diff,
                residual_t0,
                residual_t1,
                fused_change_repr,
                x_dense0,
                x_dense1,
            ]
        else:
            fixed_change_repr = self.fixed_semantic_spatial_fusion(
                torch.cat(
                    [
                        change_repr,
                        semantic_relation,
                        xy_relation,
                        height_cues,
                    ],
                    dim=-1,
                )
            )
            point_features = [
                updated_t0,
                updated_t1,
                updated_diff,
                residual_t0,
                residual_t1,
                fixed_change_repr,
                x_dense0,
                x_dense1,
            ]
        point_input = torch.cat(point_features, dim=-1)
        return self._format_point_output(
            point_input,
            extra_outputs={
                "relation_residual": relation_residual,
                "semantic_logits_t0": semantic_logits_t0,
                "semantic_logits_t1": semantic_logits_t1,
                "semantic_change_gate": semantic_gate,
                "ordered_semantic_transition": ordered_semantic_transition,
                "ordered_transition_residual_gate": (
                    ordered_transition_residual_gate
                ),
                "ordered_transition_residual_scale": (
                    torch.tanh(self.ordered_transition_residual_scale)
                    if self.use_residual_ordered_semantic_transition
                    else None
                ),
            },
        )


class PTV3T1AnchoredSemanticXYXYZRelationAttentionPoint(
    PTV3SemanticRelationAttentionPoint
):
    """Independent-sampling variant whose complete prediction graph is T1-anchored."""

    def __init__(
        self,
        num_classes=7,
        match_k=8,
        match_tau_feat=0.1,
        match_tau_geo=1.0,
        signed_diff=True,
        ptv3_order=("z", "z-trans", "hilbert", "hilbert-trans"),
        ptv3_patch_size=1024,
        use_binary_change_auxiliary_loss=False,
        use_removal_like_auxiliary_loss=False,
        semantic_num_classes=4,
        use_scalable_knn=False,
        match_detach_scores=True,
        match_query_chunk_size=4096,
        use_t1_ordered_semantic_transition=False,
        use_local_vertical_profile=False,
        me_cpt_root=osp.join(osp.dirname(osp.abspath(__file__)), "vendor", "ME-CPT"),
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            match_k=match_k,
            match_tau_feat=match_tau_feat,
            match_tau_geo=match_tau_geo,
            signed_diff=signed_diff,
            ptv3_order=ptv3_order,
            ptv3_patch_size=ptv3_patch_size,
            use_binary_change_auxiliary_loss=use_binary_change_auxiliary_loss,
            use_removal_like_auxiliary_loss=use_removal_like_auxiliary_loss,
            semantic_num_classes=semantic_num_classes,
            use_xy_xyz_prior=True,
            use_adaptive_semantic_spatial_fusion=True,
            me_cpt_root=me_cpt_root,
        )
        self.use_scalable_knn = bool(use_scalable_knn)
        self.match_detach_scores = bool(match_detach_scores)
        self.match_query_chunk_size = max(int(match_query_chunk_size), 1)
        self.use_t1_ordered_semantic_transition = bool(
            use_t1_ordered_semantic_transition
        )
        self.use_local_vertical_profile = bool(use_local_vertical_profile)
        if self.use_local_vertical_profile:
            channels = 64
            prior_channels = 32
            profile_dim = 8
            self.local_vertical_profile_encoder = nn.Sequential(
                nn.Linear(profile_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Linear(32, channels),
            )
            self.local_vertical_profile_gate = nn.Sequential(
                nn.Linear(channels * 2 + prior_channels, channels),
                nn.LayerNorm(channels),
                nn.GELU(),
                nn.Linear(channels, channels),
                nn.Sigmoid(),
            )
            # Preserve the warm-start prediction exactly until training learns
            # that the local vertical profile provides a useful correction.
            self.local_vertical_profile_scale = nn.Parameter(torch.zeros(1))
        if self.use_t1_ordered_semantic_transition:
            channels = 64
            semantic_channels = 32
            prior_channels = 32
            self.t1_ordered_transition_encoder = nn.Sequential(
                nn.Linear(semantic_channels * 4, semantic_channels),
                nn.LayerNorm(semantic_channels),
                nn.GELU(),
                nn.Linear(semantic_channels, semantic_channels),
            )
            self.t1_ordered_transition_adapter = nn.Sequential(
                nn.Linear(semantic_channels, channels),
                nn.LayerNorm(channels),
                nn.GELU(),
                nn.Linear(channels, channels),
            )
            self.t1_ordered_transition_gate = nn.Sequential(
                nn.Linear(
                    channels + semantic_channels + prior_channels,
                    channels,
                ),
                nn.LayerNorm(channels),
                nn.GELU(),
                nn.Linear(channels, channels),
                nn.Sigmoid(),
            )
            # A zero scale preserves the warm-start baseline exactly. It first
            # learns whether the ordered residual is useful before allowing
            # the new message and gate branches to alter the representation.
            self.t1_ordered_transition_scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _xy_vertical_statistics(query_coord, key_coord, match_info):
        """Weighted vertical mean residual, spread, and range for XY candidates."""
        batch_size = key_coord.size(0)
        batch_idx = torch.arange(
            batch_size,
            device=key_coord.device,
        ).view(batch_size, 1, 1)
        local_z = key_coord[..., 2:3][batch_idx, match_info["knn_idx"]]
        weight = match_info["weight"].unsqueeze(-1)
        mean_z = (weight * local_z).sum(dim=2)
        variance_z = (weight * (local_z - mean_z.unsqueeze(2)).square()).sum(
            dim=2
        )
        std_z = torch.sqrt(variance_z.clamp_min(1e-8))
        range_z = local_z.amax(dim=2) - local_z.amin(dim=2)
        mean_residual = mean_z - query_coord[..., 2:3]
        return mean_residual, std_z, range_z

    @staticmethod
    def _signed_log_pair(value):
        magnitude = torch.log1p(value.abs())
        return torch.cat([torch.sign(value) * magnitude, magnitude], dim=-1)

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_dense_inputs(data)
        time_value0 = float(data.get("_time_value0", 0.0))
        time_value1 = float(data.get("_time_value1", 1.0))
        feat_t0 = self._encode_points(
            coord0, x_dense0, time_value=time_value0
        )
        feat_t1 = self._encode_points(
            coord1, x_dense1, time_value=time_value1
        )

        xyz_t1_for_t0 = self._point_soft_match_info(
            coord0, coord1, feat_t0, feat_t1
        )
        xyz_t0_for_t1 = self._point_soft_match_info(
            coord1, coord0, feat_t1, feat_t0
        )
        xy_t1_for_t0 = self._point_soft_match_info(
            coord0[..., :2], coord1[..., :2], feat_t0, feat_t1
        )
        xy_t0_for_t1 = self._point_soft_match_info(
            coord1[..., :2], coord0[..., :2], feat_t1, feat_t0
        )

        updated_t0 = self._relation_attention_update(feat_t0, xyz_t1_for_t0)
        updated_t1 = self._relation_attention_update(feat_t1, xyz_t0_for_t1)
        updated_t0_on_t1 = self._align_embedding(updated_t0, xyz_t0_for_t1)

        semantic_t0 = self.semantic_feature_head(updated_t0)
        semantic_t1 = self.semantic_feature_head(updated_t1)
        semantic_logits_t0 = self.semantic_classifier(semantic_t0)
        semantic_logits_t1 = self.semantic_classifier(semantic_t1)
        semantic_t1_on_t0 = self._align_embedding(semantic_t1, xyz_t1_for_t0)
        semantic_t0_on_t1 = self._align_embedding(semantic_t0, xyz_t0_for_t1)
        semantic_relation_t0 = self._semantic_transition(
            semantic_t0, semantic_t1_on_t0
        )
        semantic_relation_t1 = self._semantic_transition(
            semantic_t1, semantic_t0_on_t1
        )
        semantic_relation = 0.5 * (
            self._align_embedding(semantic_relation_t0, xyz_t0_for_t1)
            + semantic_relation_t1
        )

        aligned_t1_for_t0 = xyz_t1_for_t0["aligned"]
        aligned_t0_for_t1 = xyz_t0_for_t1["aligned"]
        residual_t0 = updated_t0 - aligned_t1_for_t0
        residual_t1 = updated_t1 - aligned_t0_for_t1
        residual_t0_on_t1 = self._align_embedding(residual_t0, xyz_t0_for_t1)
        updated_diff = updated_t1 - updated_t0_on_t1
        if not self.signed_diff:
            residual_t0_on_t1 = residual_t0_on_t1.abs()
            residual_t1 = residual_t1.abs()
            updated_diff = updated_diff.abs()

        change_t0 = self.change_fusion_t0(
            self._pair_features(updated_t0, aligned_t1_for_t0)
        )
        change_t1 = self.change_fusion_t1(
            self._pair_features(updated_t1, aligned_t0_for_t1)
        )
        change_repr = 0.5 * (
            self._align_embedding(change_t0, xyz_t0_for_t1) + change_t1
        )

        aligned_xy_t1_for_t0 = self._align_embedding(updated_t1, xy_t1_for_t0)
        aligned_xy_t0_for_t1 = self._align_embedding(updated_t0, xy_t0_for_t1)
        xy_relation_t0 = self.xy_relation_encoder(
            self._pair_features(updated_t0, aligned_xy_t1_for_t0)
        )
        xy_relation_t1 = self.xy_relation_encoder(
            self._pair_features(updated_t1, aligned_xy_t0_for_t1)
        )
        xy_relation = self.xy_direction_fusion(
            torch.cat(
                [
                    self._align_embedding(xy_relation_t0, xy_t0_for_t1),
                    xy_relation_t1,
                ],
                dim=-1,
            )
        )

        dz_t0, std_t1_on_t0, range_t1_on_t0 = self._xy_vertical_statistics(
            coord0,
            coord1,
            xy_t1_for_t0,
        )
        mean_t0_minus_t1, std_t0_on_t1, range_t0_on_t1 = (
            self._xy_vertical_statistics(
                coord1,
                coord0,
                xy_t0_for_t1,
            )
        )
        dz_t1 = -mean_t0_minus_t1
        height_t0 = self._signed_log_pair(dz_t0)
        height_t1 = self._signed_log_pair(dz_t1)
        height_cues = torch.cat(
            [self._align_embedding(height_t0, xy_t0_for_t1), height_t1],
            dim=-1,
        )
        local_vertical_profile = None
        if self.use_local_vertical_profile:
            std_t1_on_t1 = self._align_embedding(
                std_t1_on_t0,
                xy_t0_for_t1,
            )
            range_t1_on_t1 = self._align_embedding(
                range_t1_on_t0,
                xy_t0_for_t1,
            )
            local_vertical_profile = torch.cat(
                [
                    height_cues,
                    self._signed_log_pair(std_t1_on_t1 - std_t0_on_t1),
                    self._signed_log_pair(range_t1_on_t1 - range_t0_on_t1),
                ],
                dim=-1,
            )

        xyz_prior_t0 = self.semantic_prior_encoder(
            self._build_match_prior(xyz_t1_for_t0)
        )
        xyz_prior_t1 = self.semantic_prior_encoder(
            self._build_match_prior(xyz_t0_for_t1)
        )
        xy_prior_t0 = self.semantic_prior_encoder(
            self._build_match_prior(xy_t1_for_t0)
        )
        xy_prior_t1 = self.semantic_prior_encoder(
            self._build_match_prior(xy_t0_for_t1)
        )
        prior_embedding = self.xy_xyz_prior_encoder(
            torch.cat(
                [
                    self._align_embedding(xyz_prior_t0, xyz_t0_for_t1),
                    xyz_prior_t1,
                    self._align_embedding(xy_prior_t0, xy_t0_for_t1),
                    xy_prior_t1,
                    height_cues,
                ],
                dim=-1,
            )
        )
        semantic_gate = self.xy_xyz_change_gate(
            torch.cat(
                [
                    change_repr,
                    semantic_relation,
                    xy_relation,
                    prior_embedding,
                    height_cues,
                ],
                dim=-1,
            )
        )
        semantic_message = self.xy_xyz_change_message(
            torch.cat([semantic_relation, xy_relation, height_cues], dim=-1)
        )
        fused_change_repr = self.xy_xyz_change_norm(
            change_repr + semantic_gate * semantic_message
        )

        local_vertical_profile_gate = None
        if local_vertical_profile is not None:
            local_vertical_message = self.local_vertical_profile_encoder(
                local_vertical_profile
            )
            local_vertical_profile_gate = self.local_vertical_profile_gate(
                torch.cat(
                    [
                        fused_change_repr,
                        local_vertical_message,
                        prior_embedding,
                    ],
                    dim=-1,
                )
            )
            fused_change_repr = fused_change_repr + torch.tanh(
                self.local_vertical_profile_scale
            ) * local_vertical_profile_gate * local_vertical_message

        t1_ordered_transition = None
        t1_ordered_transition_gate = None
        if self.use_t1_ordered_semantic_transition:
            semantic_signed_difference = semantic_t1 - semantic_t0_on_t1
            t1_ordered_transition = self.t1_ordered_transition_encoder(
                torch.cat(
                    [
                        semantic_t1,
                        semantic_t0_on_t1,
                        semantic_signed_difference,
                        semantic_signed_difference.abs(),
                    ],
                    dim=-1,
                )
            )
            t1_ordered_transition_gate = self.t1_ordered_transition_gate(
                torch.cat(
                    [
                        fused_change_repr,
                        t1_ordered_transition,
                        prior_embedding,
                    ],
                    dim=-1,
                )
            )
            ordered_transition_delta = self.t1_ordered_transition_adapter(
                t1_ordered_transition
            )
            fused_change_repr = fused_change_repr + torch.tanh(
                self.t1_ordered_transition_scale
            ) * t1_ordered_transition_gate * ordered_transition_delta

        residual_norm_t0 = (
            F.normalize(updated_t0, dim=-1)
            - F.normalize(aligned_t1_for_t0, dim=-1)
        ).norm(dim=-1, keepdim=True)
        residual_norm_t1 = (
            F.normalize(updated_t1, dim=-1)
            - F.normalize(aligned_t0_for_t1, dim=-1)
        ).norm(dim=-1, keepdim=True)
        relation_residual = 0.5 * (
            self._align_embedding(residual_norm_t0, xyz_t0_for_t1)
            + residual_norm_t1
        )

        # Every tensor below is defined on T1 anchors. Equal input indices are
        # never treated as temporal correspondences.
        point_input = torch.cat(
            [
                updated_t0_on_t1,
                updated_t1,
                updated_diff,
                residual_t0_on_t1,
                residual_t1,
                fused_change_repr,
                self._align_embedding(x_dense0, xyz_t0_for_t1),
                x_dense1,
            ],
            dim=-1,
        )
        extra_outputs = {
            "relation_residual": relation_residual,
            "semantic_logits_t0": semantic_logits_t0,
            "semantic_logits_t1": semantic_logits_t1,
            "semantic_change_gate": semantic_gate,
        }
        if t1_ordered_transition is not None:
            extra_outputs.update(
                {
                    "t1_ordered_semantic_transition": t1_ordered_transition,
                    "t1_ordered_transition_gate": t1_ordered_transition_gate,
                    "t1_ordered_transition_scale": torch.tanh(
                        self.t1_ordered_transition_scale
                    ),
                }
            )
        if local_vertical_profile is not None:
            extra_outputs.update(
                {
                    "local_vertical_profile": local_vertical_profile,
                    "local_vertical_profile_gate": local_vertical_profile_gate,
                    "local_vertical_profile_scale": torch.tanh(
                        self.local_vertical_profile_scale
                    ),
                }
            )
        return self._format_point_output(
            point_input,
            extra_outputs=extra_outputs,
        )


class PTV3UpdateRefinePoint(PTV3DirectionFusionPoint):
    """
    PTv3 update-refine ablation.

    Pipeline:
      PTv3 point encoding for T0/T1 separately
      -> one bidirectional point-level local soft matching
      -> gated cross-temporal feature update
      -> lightweight point refinement
      -> change representation head

    This intentionally does not run a second matching stage. The goal is to
    test whether updating the temporal features before prediction improves the
    strong PTv3 no-direction baseline.
    """

    def __init__(
        self,
        num_classes=3,
        match_k=16,
        match_tau_feat=0.1,
        match_tau_geo=1.0,
        signed_diff=True,
        ptv3_order=("z", "z-trans", "hilbert", "hilbert-trans"),
        ptv3_patch_size=512,
        me_cpt_root=osp.join(osp.dirname(osp.abspath(__file__)), "vendor", "ME-CPT"),
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            match_k=match_k,
            match_tau_feat=match_tau_feat,
            match_tau_geo=match_tau_geo,
            signed_diff=signed_diff,
            direction_aware=False,
            ptv3_order=ptv3_order,
            ptv3_patch_size=ptv3_patch_size,
            me_cpt_root=me_cpt_root,
        )
        channels = 64
        pair_dim = channels * 5
        self.cross_update = nn.Sequential(
            nn.Linear(pair_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.cross_update_gate = nn.Sequential(
            nn.Linear(pair_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        refine_dim = channels * 3
        self.point_refine = nn.Sequential(
            nn.Linear(refine_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        head_in_dim = channels * 7 + 12
        self.point_head = nn.Sequential(
            nn.Linear(head_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def _update_features(self, query_feat, aligned_feat):
        pair = self._pair_features(query_feat, aligned_feat)
        message = self.cross_update(pair)
        gate = self.cross_update_gate(pair)
        return query_feat + gate * message

    def _refine_features(self, updated_feat, aligned_feat):
        if self.signed_diff:
            diff = updated_feat - aligned_feat
        else:
            diff = torch.abs(updated_feat - aligned_feat)
        refine_input = torch.cat([updated_feat, aligned_feat, diff], dim=-1)
        return updated_feat + self.point_refine(refine_input)

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_dense_inputs(data)
        feat_t0 = self._encode_points(coord0, x_dense0, time_value=0.0)
        feat_t1 = self._encode_points(coord1, x_dense1, time_value=1.0)

        aligned_t1_for_t0 = self._point_soft_match(coord0, coord1, feat_t0, feat_t1)
        aligned_t0_for_t1 = self._point_soft_match(coord1, coord0, feat_t1, feat_t0)

        updated_t0 = self._update_features(feat_t0, aligned_t1_for_t0)
        updated_t1 = self._update_features(feat_t1, aligned_t0_for_t1)
        refined_t0 = self._refine_features(updated_t0, aligned_t1_for_t0)
        refined_t1 = self._refine_features(updated_t1, aligned_t0_for_t1)

        if self.signed_diff:
            refined_diff = refined_t1 - refined_t0
            update_residual_t0 = updated_t0 - aligned_t1_for_t0
            update_residual_t1 = updated_t1 - aligned_t0_for_t1
        else:
            refined_diff = torch.abs(refined_t1 - refined_t0)
            update_residual_t0 = torch.abs(updated_t0 - aligned_t1_for_t0)
            update_residual_t1 = torch.abs(updated_t1 - aligned_t0_for_t1)

        change_repr = 0.5 * (
            self.change_fusion_t0(self._pair_features(refined_t0, aligned_t1_for_t0))
            + self.change_fusion_t1(self._pair_features(refined_t1, aligned_t0_for_t1))
        )
        point_input = torch.cat(
            [
                refined_t0,
                refined_t1,
                refined_diff,
                update_residual_t0,
                update_residual_t1,
                change_repr,
                torch.abs(refined_t1 - refined_t0),
                x_dense0,
                x_dense1,
            ],
            dim=-1,
        )
        return self.point_head(point_input)


class PTV3CleanChangePoint(PTV3DirectionFusionPoint):
    """
    PTv3 clean change-representation ablation.

    Pipeline:
      PTv3 point encoding for T0/T1 separately
      -> bidirectional point-level local soft matching
      -> shared no-direction change representation encoder
      -> compact point-level 3-class prediction

    Compared with PTV3DirectionFusionPoint(direction_aware=False), this avoids
    dumping all handcrafted residual features into the final classifier. The
    classifier sees the two temporal features, their direct difference, one
    learned change representation, and raw point attributes.
    """

    def __init__(
        self,
        num_classes=3,
        match_k=16,
        match_tau_feat=0.1,
        match_tau_geo=1.0,
        signed_diff=True,
        ptv3_order=("z", "z-trans", "hilbert", "hilbert-trans"),
        ptv3_patch_size=512,
        me_cpt_root=osp.join(osp.dirname(osp.abspath(__file__)), "vendor", "ME-CPT"),
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            match_k=match_k,
            match_tau_feat=match_tau_feat,
            match_tau_geo=match_tau_geo,
            signed_diff=signed_diff,
            direction_aware=False,
            ptv3_order=ptv3_order,
            ptv3_patch_size=ptv3_patch_size,
            me_cpt_root=me_cpt_root,
        )
        channels = 64
        head_in_dim = channels * 4 + 12
        self.point_head = nn.Sequential(
            nn.Linear(head_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_dense_inputs(data)
        feat_t0 = self._encode_points(coord0, x_dense0, time_value=0.0)
        feat_t1 = self._encode_points(coord1, x_dense1, time_value=1.0)

        aligned_t1_for_t0 = self._point_soft_match(coord0, coord1, feat_t0, feat_t1)
        aligned_t0_for_t1 = self._point_soft_match(coord1, coord0, feat_t1, feat_t0)

        if self.signed_diff:
            point_feat_diff = feat_t1 - feat_t0
        else:
            point_feat_diff = torch.abs(feat_t1 - feat_t0)

        change_t0 = self.change_fusion_t0(
            self._pair_features(feat_t0, aligned_t1_for_t0)
        )
        change_t1 = self.change_fusion_t1(
            self._pair_features(feat_t1, aligned_t0_for_t1)
        )
        change_repr = 0.5 * (change_t0 + change_t1)
        point_input = torch.cat(
            [
                feat_t0,
                feat_t1,
                point_feat_diff,
                change_repr,
                x_dense0,
                x_dense1,
            ],
            dim=-1,
        )
        return self.point_head(point_input)


class PTV3UpdatePTV3RefinePoint(PTV3DirectionFusionPoint):
    """
    PTv3 correspondence-update + PTv3 refinement ablation.

    Pipeline:
      PTv3 point encoding for T0/T1 separately
      -> one bidirectional point-level local soft matching
      -> gated cross-temporal feature update
      -> second lightweight PTv3 refinement for T0/T1
      -> shared no-direction change fusion
      -> point-level 3-class prediction

    The second PTv3 operates after correspondence has modified the features, so
    the refinement stage can propagate change-aware information spatially.
    """

    def __init__(
        self,
        num_classes=3,
        match_k=16,
        match_tau_feat=0.1,
        match_tau_geo=1.0,
        signed_diff=True,
        ptv3_order=("z", "z-trans", "hilbert", "hilbert-trans"),
        ptv3_patch_size=512,
        me_cpt_root=osp.join(osp.dirname(osp.abspath(__file__)), "vendor", "ME-CPT"),
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            match_k=match_k,
            match_tau_feat=match_tau_feat,
            match_tau_geo=match_tau_geo,
            signed_diff=signed_diff,
            direction_aware=False,
            ptv3_order=ptv3_order,
            ptv3_patch_size=ptv3_patch_size,
            me_cpt_root=me_cpt_root,
        )
        try:
            from pointcept.models.point_transformer_v3.ptv3_cd_multi import (
                PointTransformerV3,
            )
            _patch_pointcept_hilbert_for_old_torch()
        except Exception as exc:
            raise ImportError(
                "Could not import ME-CPT PointTransformerV3 for PTv3 refinement."
            ) from exc

        channels = 64
        pair_dim = channels * 5
        self.cross_update = nn.Sequential(
            nn.Linear(pair_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.cross_update_gate = nn.Sequential(
            nn.Linear(pair_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        enc_patch_size = tuple([int(ptv3_patch_size)] * 5)
        dec_patch_size = tuple([int(ptv3_patch_size)] * 4)
        self.refiner = PointTransformerV3(
            in_channels=65,
            order=tuple(ptv3_order),
            stride=(2, 2, 2, 2),
            enc_depths=(1, 1, 1, 2, 1),
            enc_channels=(64, 64, 64, 128, 256),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=enc_patch_size,
            dec_depths=(1, 1, 1, 1),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=dec_patch_size,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.2,
            shuffle_orders=False,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=False,
            upcast_attention=True,
            upcast_softmax=True,
            cls_mode=False,
            pdnorm_bn=False,
            pdnorm_ln=False,
        )
        head_in_dim = channels * 6 + 12
        self.point_head = nn.Sequential(
            nn.Linear(head_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def _update_features(self, query_feat, aligned_feat):
        pair = self._pair_features(query_feat, aligned_feat)
        message = self.cross_update(pair)
        gate = self.cross_update_gate(pair)
        return query_feat + gate * message

    def _refine_points(self, coord, feat, time_value):
        batch_size, num_points, _ = coord.shape
        flat_coord = coord.reshape(batch_size * num_points, 3).contiguous()
        time_feat = torch.full(
            (batch_size, num_points, 1),
            float(time_value),
            dtype=feat.dtype,
            device=feat.device,
        )
        refiner_feat = torch.cat([feat, time_feat], dim=-1)
        flat_feat = refiner_feat.reshape(
            batch_size * num_points, refiner_feat.size(-1)
        ).contiguous()
        batch = torch.arange(batch_size, device=coord.device).repeat_interleave(num_points)
        offset = torch.arange(
            1,
            batch_size + 1,
            device=coord.device,
            dtype=torch.long,
        ) * num_points
        point = self.refiner(
            {
                "coord": flat_coord,
                "feat": flat_feat,
                "batch": batch,
                "offset": offset,
                "grid_size": 0.03,
            }
        )
        return point.feat.reshape(batch_size, num_points, -1).contiguous()

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_dense_inputs(data)
        feat_t0 = self._encode_points(coord0, x_dense0, time_value=0.0)
        feat_t1 = self._encode_points(coord1, x_dense1, time_value=1.0)

        aligned_t1_for_t0 = self._point_soft_match(coord0, coord1, feat_t0, feat_t1)
        aligned_t0_for_t1 = self._point_soft_match(coord1, coord0, feat_t1, feat_t0)

        updated_t0 = self._update_features(feat_t0, aligned_t1_for_t0)
        updated_t1 = self._update_features(feat_t1, aligned_t0_for_t1)
        refined_t0 = self._refine_points(coord0, updated_t0, time_value=0.0)
        refined_t1 = self._refine_points(coord1, updated_t1, time_value=1.0)

        if self.signed_diff:
            refined_diff = refined_t1 - refined_t0
            update_residual_t0 = updated_t0 - aligned_t1_for_t0
            update_residual_t1 = updated_t1 - aligned_t0_for_t1
        else:
            refined_diff = torch.abs(refined_t1 - refined_t0)
            update_residual_t0 = torch.abs(updated_t0 - aligned_t1_for_t0)
            update_residual_t1 = torch.abs(updated_t1 - aligned_t0_for_t1)

        change_t0 = self.change_fusion_t0(self._pair_features(refined_t0, aligned_t1_for_t0))
        change_t1 = self.change_fusion_t1(self._pair_features(refined_t1, aligned_t0_for_t1))
        change_repr = 0.5 * (change_t0 + change_t1)
        point_input = torch.cat(
            [
                refined_t0,
                refined_t1,
                refined_diff,
                update_residual_t0,
                update_residual_t1,
                change_repr,
                x_dense0,
                x_dense1,
            ],
            dim=-1,
        )
        return self.point_head(point_input)


class TwoStageInteractionMambaPoint(Net_mamba_point):
    """
    Two-stage cross-temporal interaction ablation.

    Pipeline:
      Stage1 PointMamba token encoding for T0/T1 separately
      -> bidirectional local soft matching
      -> gated token update from matched temporal messages
      -> Stage2 token-level PointMamba refinement
      -> direction-aware token fusion
      -> token-to-point projection
      -> point-level 3-class prediction
    """

    def __init__(
        self,
        num_classes=3,
        token_match_k=16,
        token_match_tau_feat=0.1,
        token_match_tau_geo=1.0,
        token_match_detach_scores=True,
        num_group=128,
        group_size=16,
        signed_diff=True,
        stage2_depth=2,
        use_temporal_embedding=False,
        use_existence_branch=False,
        use_point_decoder=False,
    ) -> None:
        if stage2_depth < 1:
            raise ValueError(f"stage2_depth must be >= 1, got {stage2_depth}.")
        super().__init__(
            NUM_CLASS=num_classes,
            signed_diff=signed_diff,
            token_match_k=token_match_k,
            token_match_tau_feat=token_match_tau_feat,
            token_match_tau_geo=token_match_tau_geo,
            token_match_detach_scores=token_match_detach_scores,
            num_group=num_group,
            group_size=group_size,
            use_residual_token_mamba=False,
            use_change_aware_token_fusion=True,
            use_token_auxiliary_loss=False,
            use_explicit_change_decomposition=False,
            use_existence_aware_fusion=False,
            use_existence_auxiliary_loss=False,
            use_direction_auxiliary_loss=False,
            use_binary_change_auxiliary_loss=False,
            use_changed_direction_auxiliary_loss=False,
            use_temporal_embedding=use_temporal_embedding,
            use_temporal_4class_auxiliary_loss=False,
            use_null_correspondence=False,
        )
        update_in_dim = 384 * 5
        self.token_update_mlp = nn.Sequential(
            nn.Linear(update_in_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, 384),
        )
        self.token_update_gate = nn.Sequential(
            nn.Linear(update_in_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, 384),
            nn.Sigmoid(),
        )
        self.stage2_mamba = MixerModel(
            d_model=384,
            n_layer=stage2_depth,
            rms_norm=False,
            drop_out=0,
            drop_path=0.0,
        )
        self.use_existence_branch = use_existence_branch
        self.use_point_decoder = use_point_decoder
        two_stage_head_in_dim = 384 * 7 + 12
        decoder_dim = 384 if use_point_decoder else 0
        decoder_output_dim = two_stage_head_in_dim + decoder_dim
        self.point_decoder = (
            nn.Sequential(
                nn.Linear(two_stage_head_in_dim, 384),
                nn.LayerNorm(384),
                nn.GELU(),
                nn.Linear(384, 384),
            )
            if use_point_decoder
            else None
        )
        self.existence_point_head_t0 = (
            nn.Sequential(
                nn.Linear(decoder_output_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 2),
            )
            if use_existence_branch
            else None
        )
        self.existence_point_head_t1 = (
            nn.Sequential(
                nn.Linear(decoder_output_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 2),
            )
            if use_existence_branch
            else None
        )
        if use_existence_branch or use_point_decoder:
            self.point_head = nn.Sequential(
                nn.Linear(decoder_output_dim + (4 if use_existence_branch else 0), 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

    def _gated_token_update(self, query_tokens, aligned_key_tokens):
        pair_features = self._build_token_pair_features(query_tokens, aligned_key_tokens)
        message = self.token_update_mlp(pair_features)
        gate = self.token_update_gate(pair_features)
        return query_tokens + gate * message

    def _stage2_refine(self, token_centers, token_features):
        pos = self.mamba.pos_embed(token_centers[:, :, :3])
        return self.stage2_mamba(token_features, pos)

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_mamba_inputs(data)

        t0_info = self.mamba(x_dense0, return_token_info=True)
        t1_info = self.mamba(x_dense1, return_token_info=True)
        stage1_t0 = t0_info["token_features"]
        stage1_t1 = t1_info["token_features"]
        t0_centers = t0_info["token_centers"]
        t1_centers = t1_info["token_centers"]
        if self.time_embed is not None:
            stage1_t0 = stage1_t0 + self.time_embed[0].view(1, 1, -1)
            stage1_t1 = stage1_t1 + self.time_embed[1].view(1, 1, -1)

        aligned_stage1_t1_for_t0 = self._token_soft_match(
            t0_centers,
            t1_centers,
            stage1_t0,
            stage1_t1,
        )
        aligned_stage1_t0_for_t1 = self._token_soft_match(
            t1_centers,
            t0_centers,
            stage1_t1,
            stage1_t0,
        )

        updated_t0 = self._gated_token_update(stage1_t0, aligned_stage1_t1_for_t0)
        updated_t1 = self._gated_token_update(stage1_t1, aligned_stage1_t0_for_t1)
        stage2_t0 = self._stage2_refine(t0_centers, updated_t0)
        stage2_t1 = self._stage2_refine(t1_centers, updated_t1)

        point_feat_t0 = self._project_tokens_to_points(coord0, t0_centers, stage2_t0)
        point_feat_t1 = self._project_tokens_to_points(coord1, t1_centers, stage2_t1)
        if self.signed_diff:
            point_feat_diff = point_feat_t1 - point_feat_t0
        else:
            point_feat_diff = torch.abs(point_feat_t1 - point_feat_t0)

        aligned_stage2_t1_for_t0 = self._token_soft_match(
            t0_centers,
            t1_centers,
            stage2_t0,
            stage2_t1,
        )
        aligned_stage2_t0_for_t1 = self._token_soft_match(
            t1_centers,
            t0_centers,
            stage2_t1,
            stage2_t0,
        )

        aligned_t1_for_t0 = self._project_tokens_to_points(
            coord0, t0_centers, aligned_stage2_t1_for_t0
        )
        aligned_t0_for_t1 = self._project_tokens_to_points(
            coord1, t1_centers, aligned_stage2_t0_for_t1
        )
        if self.signed_diff:
            recon_diff_t0 = point_feat_t0 - aligned_t1_for_t0
            recon_diff_t1 = point_feat_t1 - aligned_t0_for_t1
        else:
            recon_diff_t0 = torch.abs(point_feat_t0 - aligned_t1_for_t0)
            recon_diff_t1 = torch.abs(point_feat_t1 - aligned_t0_for_t1)

        change_token_t0 = self._change_aware_token_fusion(
            stage2_t0,
            aligned_stage2_t1_for_t0,
            match_info=None,
            direction="t0",
        )
        change_token_t1 = self._change_aware_token_fusion(
            stage2_t1,
            aligned_stage2_t0_for_t1,
            match_info=None,
            direction="t1",
        )

        point_input = torch.cat(
            [
                point_feat_t0,
                point_feat_t1,
                point_feat_diff,
                recon_diff_t0,
                recon_diff_t1,
                self._project_tokens_to_points(coord0, t0_centers, change_token_t0),
                self._project_tokens_to_points(coord1, t1_centers, change_token_t1),
                x_dense0,
                x_dense1,
            ],
            dim=-1,
        )
        if self.point_decoder is not None:
            point_input = torch.cat([point_input, self.point_decoder(point_input)], dim=-1)
        if not self.use_existence_branch:
            return self.point_head(point_input)

        existence_logits_t0 = self.existence_point_head_t0(point_input)
        existence_logits_t1 = self.existence_point_head_t1(point_input)
        existence_prob_t0 = torch.softmax(existence_logits_t0, dim=-1)
        existence_prob_t1 = torch.softmax(existence_logits_t1, dim=-1)
        point_input = torch.cat(
            [point_input, existence_prob_t0, existence_prob_t1],
            dim=-1,
        )
        return {
            "point_logits": self.point_head(point_input),
            "existence_logits_t0": existence_logits_t0,
            "existence_logits_t1": existence_logits_t1,
        }


class JointTemporalMambaPoint(nn.Module):
    """
    Diagnostic early-interaction baseline.

    Pipeline:
      concatenate T0/T1 points with a scalar time feature
      -> one shared PointMamba token encoding over the joint cloud
      -> token-to-point projection back to T0/T1 point coordinates
      -> point-level 3-class prediction
    """

    def __init__(
        self,
        num_classes=3,
        num_group=128,
        group_size=16,
        signed_diff=True,
    ) -> None:
        super().__init__()
        self.mamba = PointMambaScan(
            num_group=num_group,
            group_size=group_size,
            input_channels=7,
        )
        self.signed_diff = signed_diff
        head_in_dim = 384 * 3 + 12
        self.point_head = nn.Sequential(
            nn.Linear(head_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _project_tokens_to_points(point_coord, token_centers, token_features):
        dist = torch.cdist(point_coord, token_centers[:, :, :3])
        weight = torch.softmax(-dist, dim=-1)
        return torch.matmul(weight, token_features)

    @staticmethod
    def _build_mamba_inputs(data):
        coord0 = data["coord0"]
        feat0 = data["feat0"]
        coord1 = data["coord1"]
        feat1 = data["feat1"]

        if coord0.dim() == 2:
            coord0 = coord0.unsqueeze(0)
            feat0 = feat0.unsqueeze(0)
            coord1 = coord1.unsqueeze(0)
            feat1 = feat1.unsqueeze(0)

        x_dense0 = torch.cat([coord0, feat0], dim=-1).contiguous()
        x_dense1 = torch.cat([coord1, feat1], dim=-1).contiguous()
        return coord0, coord1, x_dense0, x_dense1

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_mamba_inputs(data)
        time0 = torch.zeros(
            (*coord0.shape[:2], 1),
            dtype=x_dense0.dtype,
            device=x_dense0.device,
        )
        time1 = torch.ones(
            (*coord1.shape[:2], 1),
            dtype=x_dense1.dtype,
            device=x_dense1.device,
        )
        joint0 = torch.cat([x_dense0, time0], dim=-1)
        joint1 = torch.cat([x_dense1, time1], dim=-1)
        joint = torch.cat([joint0, joint1], dim=1).contiguous()

        joint_info = self.mamba(joint, return_token_info=True)
        joint_tokens = joint_info["token_features"]
        joint_centers = joint_info["token_centers"]

        point_feat_t0 = self._project_tokens_to_points(coord0, joint_centers, joint_tokens)
        point_feat_t1 = self._project_tokens_to_points(coord1, joint_centers, joint_tokens)
        if self.signed_diff:
            point_feat_diff = point_feat_t1 - point_feat_t0
        else:
            point_feat_diff = torch.abs(point_feat_t1 - point_feat_t0)

        point_input = torch.cat(
            [point_feat_t0, point_feat_t1, point_feat_diff, x_dense0, x_dense1],
            dim=-1,
        )
        return self.point_head(point_input)


class JointTemporalMatchingMambaPoint(Net_mamba_point):
    """
    Early-interaction plus matching diagnostic baseline.

    Pipeline:
      concatenate T0/T1 points with a scalar time feature
      -> one shared PointMamba token encoding over the joint cloud
      -> split joint-aware tokens by token time center
      -> bidirectional local soft matching
      -> direction-aware change fusion
      -> token-to-point projection
      -> point-level 3-class prediction
    """

    def __init__(
        self,
        num_classes=3,
        token_match_k=16,
        token_match_tau_feat=0.1,
        token_match_tau_geo=1.0,
        token_match_detach_scores=True,
        num_group=128,
        group_size=16,
        signed_diff=True,
    ) -> None:
        super().__init__(
            NUM_CLASS=num_classes,
            signed_diff=signed_diff,
            token_match_k=token_match_k,
            token_match_tau_feat=token_match_tau_feat,
            token_match_tau_geo=token_match_tau_geo,
            token_match_detach_scores=token_match_detach_scores,
            num_group=num_group,
            group_size=group_size,
            use_residual_token_mamba=False,
            use_change_aware_token_fusion=True,
            use_token_auxiliary_loss=False,
            use_explicit_change_decomposition=False,
            use_existence_aware_fusion=False,
            use_existence_auxiliary_loss=False,
            use_direction_auxiliary_loss=False,
            use_binary_change_auxiliary_loss=False,
            use_changed_direction_auxiliary_loss=False,
            use_temporal_embedding=False,
            use_temporal_4class_auxiliary_loss=False,
            use_null_correspondence=False,
        )
        self.mamba = PointMambaScan(
            num_group=num_group,
            group_size=group_size,
            input_channels=7,
        )

    @staticmethod
    def _gather_token_subset(values, indices):
        feature_dim = values.size(-1)
        expanded = indices.unsqueeze(-1).expand(-1, -1, feature_dim)
        return values.gather(1, expanded)

    def _split_joint_tokens_by_time(self, joint_info):
        tokens = joint_info["token_features"]
        centers = joint_info["token_centers"]
        token_times = joint_info.get("token_times")
        if token_times is None:
            raise RuntimeError("JointTemporalMatchingMambaPoint requires token_times.")

        scores = token_times.squeeze(-1)
        k = min(self.num_group, tokens.size(1))
        idx_t0 = torch.topk(-scores, k=k, dim=1).indices
        idx_t1 = torch.topk(scores, k=k, dim=1).indices

        t0_tokens = self._gather_token_subset(tokens, idx_t0)
        t1_tokens = self._gather_token_subset(tokens, idx_t1)
        t0_centers = self._gather_token_subset(centers, idx_t0)
        t1_centers = self._gather_token_subset(centers, idx_t1)
        return t0_tokens, t1_tokens, t0_centers, t1_centers

    def forward(self, data):
        coord0, coord1, x_dense0, x_dense1 = self._build_mamba_inputs(data)
        time0 = torch.zeros(
            (*coord0.shape[:2], 1),
            dtype=x_dense0.dtype,
            device=x_dense0.device,
        )
        time1 = torch.ones(
            (*coord1.shape[:2], 1),
            dtype=x_dense1.dtype,
            device=x_dense1.device,
        )
        joint = torch.cat(
            [
                torch.cat([x_dense0, time0], dim=-1),
                torch.cat([x_dense1, time1], dim=-1),
            ],
            dim=1,
        ).contiguous()

        joint_info = self.mamba(joint, return_token_info=True)
        t0_tokens, t1_tokens, t0_centers, t1_centers = self._split_joint_tokens_by_time(
            joint_info
        )

        point_feat_t0 = self._project_tokens_to_points(coord0, t0_centers, t0_tokens)
        point_feat_t1 = self._project_tokens_to_points(coord1, t1_centers, t1_tokens)
        if self.signed_diff:
            point_feat_diff = point_feat_t1 - point_feat_t0
        else:
            point_feat_diff = torch.abs(point_feat_t1 - point_feat_t0)

        aligned_t1_tokens_for_t0 = self._token_soft_match(
            t0_centers,
            t1_centers,
            t0_tokens,
            t1_tokens,
        )
        aligned_t0_tokens_for_t1 = self._token_soft_match(
            t1_centers,
            t0_centers,
            t1_tokens,
            t0_tokens,
        )

        interacted_t0_tokens = self._cross_temporal_interact(
            t0_tokens, aligned_t1_tokens_for_t0
        )
        interacted_t1_tokens = self._cross_temporal_interact(
            t1_tokens, aligned_t0_tokens_for_t1
        )

        aligned_t1_for_t0 = self._project_tokens_to_points(
            coord0, t0_centers, aligned_t1_tokens_for_t0
        )
        aligned_t0_for_t1 = self._project_tokens_to_points(
            coord1, t1_centers, aligned_t0_tokens_for_t1
        )
        if self.signed_diff:
            recon_diff_t0 = point_feat_t0 - aligned_t1_for_t0
            recon_diff_t1 = point_feat_t1 - aligned_t0_for_t1
        else:
            recon_diff_t0 = torch.abs(point_feat_t0 - aligned_t1_for_t0)
            recon_diff_t1 = torch.abs(point_feat_t1 - aligned_t0_for_t1)

        change_token_t0 = self._change_aware_token_fusion(
            interacted_t0_tokens,
            aligned_t1_tokens_for_t0,
            match_info=None,
            direction="t0",
        )
        change_token_t1 = self._change_aware_token_fusion(
            interacted_t1_tokens,
            aligned_t0_tokens_for_t1,
            match_info=None,
            direction="t1",
        )

        point_input = torch.cat(
            [
                point_feat_t0,
                point_feat_t1,
                point_feat_diff,
                recon_diff_t0,
                recon_diff_t1,
                self._project_tokens_to_points(coord0, t0_centers, change_token_t0),
                self._project_tokens_to_points(coord1, t1_centers, change_token_t1),
                x_dense0,
                x_dense1,
            ],
            dim=-1,
        )
        return self.point_head(point_input)


  

class MaskMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.mamba_config.mask_ratio
        self.trans_dim = config.mamba_config.trans_dim
        self.depth = config.mamba_config.depth
        self.drop_path_rate = config.mamba_config.drop_path_rate
        print_log(f'[args] {config.mamba_config}', logger='Mamba')
        # embedding
        self.encoder_dims = config.mamba_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.mamba_config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.config.rms_norm,
                                 drop_path=dpr)

        self.OrderScale_gamma_1, self.OrderScale_beta_1 = init_OrderScale(self.trans_dim)
        self.OrderScale_gamma_2, self.OrderScale_beta_2 = init_OrderScale(self.trans_dim)

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

    def _mask_center_rand(self, center, noaug=False):
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
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, neighborhood, center, order, order_index, noaug=False):  # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C

        if order_index == 0:
            group_input_tokens = apply_OrderScale(group_input_tokens, self.OrderScale_gamma_1, self.OrderScale_beta_1)
        elif order_index == 1:
            group_input_tokens = apply_OrderScale(group_input_tokens, self.OrderScale_gamma_2, self.OrderScale_beta_2)
        else:
            assert False

        batch_size, seq_len, C = group_input_tokens.size()
        #
        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        x_vis = self.blocks(x_vis, pos)

        return x_vis, bool_masked_pos, group_input_tokens


class MambaDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, norm_layer=nn.LayerNorm, drop_path=0, config=None):
        super().__init__()

        self.blocks = MixerModel(d_model=embed_dim,
                                 n_layer=depth,
                                 rms_norm=config.rms_norm,
                                 drop_path=drop_path)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, mask_pos):
        B, _, C = x.shape
        x = self.blocks(x, pos)

        x = self.head(x[mask_pos, :].view(B, -1, C))  # only return the mask tokens predict pixel
        return x

    # def forward(self, x, pos, N):
    #     B, _, C = x.shape
    #     x = self.blocks(x, pos)
    #
    #     x = self.head(x[:, -N:, :])  # only return the mask tokens predict pixel
    #     return x



class Point_MAE_Mamba_serializationV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointMamba] ', logger='PointMamba')
        self.config = config
        self.trans_dim = config.mamba_config.trans_dim
        self.MAE_encoder = MaskMamba(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.mamba_config.decoder_depth
        self.drop_path_rate = config.mamba_config.drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]

        self.MAE_decoder = MambaDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path=dpr,
            config=config,
        )

        print_log(f'[PointMamba] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='PointMamba')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, pts, vis=False, **kwargs):
        neighborhood, center = self.group_divider(pts)
        B, G, S, _ = neighborhood.shape

        order_list = ['hilbert', 'hilbert-trans']
        # order_list = ['hilbert']
        order_index = np.random.choice([i for i in range(len(order_list))], 1, replace=False)

        center, order, index_order, _, _ = serialization_func(center, None, None, order_list[order_index[0]])
        neighborhood = neighborhood.flatten(0, 1)[order].reshape(B, G, S, -1).contiguous()

        x_vis, mask, group_input_tokens = self.MAE_encoder(neighborhood, center, order=order,
                                                           order_index=order_index[0])
        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        x_full = torch.zeros_like(group_input_tokens, device=group_input_tokens.device)
        x_full[~mask] = x_vis.reshape(-1, C)
        x_full[mask] = self.mask_token.reshape(-1, C)
        pos_full = torch.zeros_like(group_input_tokens, device=group_input_tokens.device)
        pos_full[~mask] = pos_emd_vis.reshape(-1, C)
        pos_full[mask] = pos_emd_mask.reshape(-1, C)

        x_rec = self.MAE_decoder(x_full, pos_full, mask)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        if vis:  # visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss1


def init_OrderScale(dim):
    gamma = nn.Parameter(torch.ones(dim))
    beta = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1, std=.02)
    nn.init.normal_(beta, std=.02)
    return gamma, beta


def apply_OrderScale(x, gamma, beta):
    assert gamma.shape == beta.shape
    if x.shape[-1] == gamma.shape[0]:
        return x * gamma + beta
    elif x.shape[1] == gamma.shape[0]:
        return x * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


def serialization_func(p, x, x_res, order, layers_outputs=[]):
    p, order, inverse_order, x, x_res = serialization(p, x, x_res=x_res, order=order,
                                                      layers_outputs=layers_outputs,
                                                      grid_size=0.02)
    return p, order, inverse_order, x, x_res
