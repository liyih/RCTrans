# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import warnings
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_attention,
                                         build_feedforward_network)
from mmcv.cnn.bricks.drop import build_dropout
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import build_norm_layer, xavier_init
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.utils import deprecated_api_warning, ConfigDict
import copy
from torch.nn import ModuleList
from .attention import FlashMHA
import torch.utils.checkpoint as cp
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.runner import auto_fp16

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class RCTransTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):

        super(RCTransTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.bev_size = 128
        self.test_breaking = 2
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, key, value, key_pos, query_pos, temp_memory, temp_pos, key_padding_mask, attn_masks, \
                reg_branch, cls_branches, reg_branches, reference_points, img_metas, query_embed, temporal_alignment_pos):

        outputs_classes = []
        outputs_coords = []
        intermediate = []
        assert reference_points is not None
        
        bev_key = key[:self.bev_size * self.bev_size, :, :]
        rv_key = key[self.bev_size * self.bev_size:, :, :]
        bev_key_pos = key_pos[:self.bev_size * self.bev_size, :, :]
        rv_key_pos = key_pos[self.bev_size * self.bev_size:, :, :]

        bev_temp_pos = temp_pos[0].transpose(1,0).contiguous()
        rv_temp_pos = temp_pos[1].transpose(1,0).contiguous()
        temp_memory = temp_memory.transpose(1,0).contiguous()

        bev_query_pos = query_pos[0].transpose(1,0).contiguous()
        rv_query_pos = query_pos[1].transpose(1,0).contiguous()

        for index in range(int(len(self.layers)/2)):

            query = self.layers[2*index](query, bev_key, bev_key, bev_query_pos, bev_key_pos, temp_memory, bev_temp_pos, attn_masks) # [Nq, B, C]
            query = self.layers[2*index + 1](query, rv_key, rv_key, rv_query_pos, rv_key_pos, temp_memory, rv_temp_pos, attn_masks) # [Nq, B, C]
            
            if self.post_norm is not None:
                temp_out = self.post_norm(query)
                
                temp_out = torch.nan_to_num(temp_out).transpose(1, 0)

                intermediate.append(temp_out)

                # predict
                reference = inverse_sigmoid(reference_points.clone())
                assert reference.shape[-1] == 3
                outputs_class =cls_branches[index](temp_out)
                tmp = reg_branches[index](temp_out)

                tmp[..., 0:3] += reference[..., 0:3]
                tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

                outputs_coord = tmp

                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            if index == self.test_breaking:
                if not self.training:
                    return outputs_classes, outputs_coords, torch.stack(intermediate)
            # update query pos
            if index < (int(len(self.layers)/2)-1):
                reference_points = tmp[..., 0:3].clone()
                bev_query_embeds, rv_query_embeds = query_embed(reference_points, img_metas)
                bev_query_pos, rv_query_pos = temporal_alignment_pos(bev_query_embeds, rv_query_embeds, reference_points)
                bev_query_pos = bev_query_pos.transpose(1,0).contiguous()
                rv_query_pos = rv_query_pos.transpose(1,0).contiguous()

        return outputs_classes, outputs_coords, torch.stack(intermediate)



@TRANSFORMER.register_module()
class RCTransTemporalTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(RCTransTemporalTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        decoder['num_layers'] = decoder['num_layers']*2
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True


    def forward(self, memory, tgt, query_pos, pos_embed, attn_masks, temp_memory=None, temp_pos=None, cls_branches=None, \
                reg_branches=None, reference_points=None, img_metas=None, query_embed=None, temporal_alignment_pos=None, \
                mask=None, reg_branch=None, ):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        memory = memory.transpose(0, 1).contiguous()
        pos_embed = pos_embed.transpose(0, 1).contiguous()
        
        n, bs, c = memory.shape

        if tgt is None:
            tgt = torch.zeros_like(query_pos[0])
        else:
            tgt = tgt.transpose(0, 1).contiguous()

        # out_dec: [num_layers, num_query, bs, dim]
        outputs_classes, outputs_coords, outs_dec = self.decoder(
            query=tgt,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_pos,
            temp_memory=temp_memory,
            temp_pos=temp_pos,
            key_padding_mask=mask,
            attn_masks=[attn_masks, None],
            reg_branch=reg_branch,
            cls_branches=cls_branches,
            reg_branches=reg_branches,
            reference_points=reference_points,
            img_metas=img_metas,
            query_embed=query_embed,
            temporal_alignment_pos=temporal_alignment_pos,
            )
        return  outputs_classes, outputs_coords, outs_dec
