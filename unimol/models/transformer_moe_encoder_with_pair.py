from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import TransformerEncoderLayer, LayerNorm, TransformerMoEEncoderLayer



class TransformerMoEEncoderWithPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
        ## MoE ##
        moe_arch:str = 'alternating',
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
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        if not post_ln:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None

        if not no_final_head_layer_norm:
            self.final_head_layer_norm = LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None
            
        if moe_arch == 'alternating':
            self.layers = self.build_alternating_stack(
                encoder_layers,
                embed_dim,
                ffn_embed_dim,
                attention_heads,
                dropout,
                attention_dropout,
                activation_dropout,
                activation_fn,
                post_ln,
                ## MoE ##
                num_experts,
                gating_top_n,
                threshold_train,
                threshold_eval,
                capacity_factor_train,
                capacity_factor_eval,
                balance_loss_coef,
                router_z_loss_coef,
                add_ff_before,
                add_ff_after,
                expert_dropout,
            )
        elif moe_arch == 'sparse':
            self.layers = self.build_sparse_stack(
                encoder_layers,
                embed_dim,
                ffn_embed_dim,
                attention_heads,
                dropout,
                attention_dropout,
                activation_dropout,
                activation_fn,
                post_ln,
                ## MoE ##
                num_experts,
                gating_top_n,
                threshold_train,
                threshold_eval,
                capacity_factor_train,
                capacity_factor_eval,
                balance_loss_coef,
                router_z_loss_coef,
                add_ff_before,
                add_ff_after,
                expert_dropout,
            )
        else:
            self.layers = self.build_dense_stack(
                encoder_layers,
                embed_dim,
                ffn_embed_dim,
                attention_heads,
                dropout,
                attention_dropout,
                activation_dropout,
                activation_fn,
                post_ln,
            )
        

    def build_alternating_stack(self, 
            encoder_layers,
            embed_dim,
            ffn_embed_dim,
            attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            activation_fn,
            post_ln,
            ## MoE ##
            num_experts,
            gating_top_n,
            threshold_train,
            threshold_eval,
            capacity_factor_train,
            capacity_factor_eval,
            balance_loss_coef,
            router_z_loss_coef,
            add_ff_before,
            add_ff_after,
            expert_dropout,
            ):
        layers = nn.ModuleList([])
        for i in range(encoder_layers):
            if i == 0 or i % 2 == 0:
                layers.append(
                        TransformerMoEEncoderLayer(
                            embed_dim=embed_dim,
                            ffn_embed_dim=ffn_embed_dim,
                            attention_heads=attention_heads,
                            dropout=dropout,
                            attention_dropout=attention_dropout,
                            activation_dropout=activation_dropout,
                            activation_fn=activation_fn,
                            post_ln=post_ln,
                            ## MoE ##
                            num_experts=num_experts,
                            gating_top_n=gating_top_n,
                            threshold_train=threshold_train,
                            threshold_eval=threshold_eval,
                            capacity_factor_train=capacity_factor_train,
                            capacity_factor_eval=capacity_factor_eval,
                            balance_loss_coef=balance_loss_coef,
                            router_z_loss_coef=router_z_loss_coef,
                            add_ff_before=add_ff_before,
                            add_ff_after=add_ff_after,
                            expert_dropout=expert_dropout
                        )
                    )
            else:
                layers.append(
                        TransformerEncoderLayer(
                            embed_dim=embed_dim,
                            ffn_embed_dim=ffn_embed_dim,
                            attention_heads=attention_heads,
                            dropout=dropout,
                            attention_dropout=attention_dropout,
                            activation_dropout=activation_dropout,
                            activation_fn=activation_fn,
                            post_ln=post_ln,
                        )
                    )
        return layers
    
    def build_sparse_stack(self, 
            encoder_layers,
            embed_dim,
            ffn_embed_dim,
            attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            activation_fn,
            post_ln,
            ## MoE ##
            num_experts,
            gating_top_n,
            threshold_train,
            threshold_eval,
            capacity_factor_train,
            capacity_factor_eval,
            balance_loss_coef,
            router_z_loss_coef,
            add_ff_before,
            add_ff_after,
            expert_dropout,
            ):
        layers = nn.ModuleList([])
        for _ in range(encoder_layers):
            layers.append(
                    TransformerMoEEncoderLayer(
                        embed_dim=embed_dim,
                        ffn_embed_dim=ffn_embed_dim,
                        attention_heads=attention_heads,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                        post_ln=post_ln,
                        ## MoE ##
                        num_experts=num_experts,
                        gating_top_n=gating_top_n,
                        threshold_train=threshold_train,
                        threshold_eval=threshold_eval,
                        capacity_factor_train=capacity_factor_train,
                        capacity_factor_eval=capacity_factor_eval,
                        balance_loss_coef=balance_loss_coef,
                        router_z_loss_coef=router_z_loss_coef,
                        add_ff_before=add_ff_before,
                        add_ff_after=add_ff_after,
                        expert_dropout=expert_dropout,
                    )
                )
        return layers
    
    def build_dense_stack(self, 
            encoder_layers,
            embed_dim,
            ffn_embed_dim,
            attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            activation_fn,
            post_ln,):
        layers = nn.ModuleList([])
        for _ in range(encoder_layers):
            layers.append(
                    TransformerEncoderLayer(
                        embed_dim=embed_dim,
                        ffn_embed_dim=ffn_embed_dim,
                        attention_heads=attention_heads,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                        post_ln=post_ln,
                    )
                )
        return layers
        
        
    def forward(
        self,
        emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        bsz = emb.size(0)
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        assert attn_mask is not None
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)
        
        accum_aux_loss, accum_balance_loss, accum_router_z_loss = [], [], []
        for i in range(len(self.layers)):
            x, attn_mask, _, aux_loss, balance_loss, router_z_loss = self.layers[i](
                x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
            )
            if aux_loss:
                accum_aux_loss.append(aux_loss)
                accum_balance_loss.append(balance_loss)
                accum_router_z_loss.append(router_z_loss)

        def norm_loss(x, eps=1e-10, tolerance=1.0):
            x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (
                torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()

        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones_like(x_norm, device=x_norm.device)
        x_norm = masked_mean(token_mask, x_norm)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(
            pair_mask, delta_pair_repr_norm, dim=(-1, -2)
        )

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        #return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm
        if accum_aux_loss:
            accum_aux_loss = torch.stack(accum_aux_loss).mean()
            accum_balance_loss = torch.stack(accum_balance_loss).mean()
            accum_router_z_loss = torch.stack(accum_router_z_loss).mean()
            return x, accum_aux_loss, accum_balance_loss, accum_router_z_loss
        else:
            return x, None, None, None