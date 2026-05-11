# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from unicore import utils
from torch import nn
from . import LayerNorm, SelfMultiheadAttention

class TransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)

        self.self_attn = SelfMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln


    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool=False,
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        # new added
        x = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x, None, None, None, None, None
        else:
            return x, attn_weights, attn_probs, None, None, None



from unimol.st_moe import MoE, SparseMoEBlock

class TransformerMoEEncoderLayer(nn.Module):
    """  
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """  
            
    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln = False,
        ## MoE ##
        num_experts:int = 8,
        gating_top_n:int = 2,
        threshold_train:float = 0.2,
        threshold_eval:float = 0.2,
        capacity_factor_train:float = 1.25,
        capacity_factor_eval:float = 2.0,
        balance_loss_coef:float = 1e-2,
        router_z_loss_coef:float = 1e-3,
        add_ff_before:bool = True,
        add_ff_after:bool = True,
        expert_dropout:float = 0.0,
    ) -> None:
        super().__init__()
            
        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
            
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)
            
        self.self_attn = SelfMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )                                                                                                                                     
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        moe = MoE(
            dim = self.embed_dim,
            num_experts=num_experts,               # increase the experts (# parameters) of your model without increasing computation
            gating_top_n=gating_top_n,               # default to top 2 gating, but can also be more (3 was tested in the paper with a lower threshold)
            threshold_train=threshold_train,          # at what threshold to accept a token to be routed to second expert and beyond - 0.2 was optimal for 2 expert routing, and apparently should be lower for 3
            threshold_eval=threshold_eval,
            capacity_factor_train=capacity_factor_train,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            capacity_factor_eval=capacity_factor_eval,      # capacity_factor_* should be set to a value >=1
            balance_loss_coef=balance_loss_coef,       # multiplier on the auxiliary expert balancing auxiliary loss
            router_z_loss_coef=router_z_loss_coef,      # loss weight for router z-loss
            expert_dim_hidden=ffn_embed_dim,
            expert_dropout=expert_dropout,  # from Table.4 of https://arxiv.org/pdf/2101.03961
        )
        self.moe_block = SparseMoEBlock(moe, add_ff_before=add_ff_before, add_ff_after=add_ff_after)
        
        #self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        #self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln                                                                  
        
    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool=False,
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        # new added
        x = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
            residual = x  
            x = self.final_layer_norm(x)  
        
        x, total_aux_loss, balance_loss, router_z_loss = self.moe_block(x)

        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x, None, None, total_aux_loss, balance_loss, router_z_loss
        else:
            return x, attn_weights, attn_probs, total_aux_loss, balance_loss, router_z_loss
        
        
                