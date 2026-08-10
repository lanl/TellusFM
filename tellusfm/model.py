import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from einops import repeat
from fairscale.nn import checkpoint_wrapper
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


class FixedModelCheckpoint(ModelCheckpoint):

    def _save_topk_checkpoint(self, trainer: Trainer, monitor_candidates: Dict[str, Tensor]) -> None:
        if self.save_top_k == 0:
            return

        # validate metric
        if self.monitor is not None:
            if self.monitor in monitor_candidates:
                self._save_monitor_checkpoint(trainer, monitor_candidates)
        else:
            self._save_none_monitor_checkpoint(trainer, monitor_candidates)


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def mlp(num_channels: int):
    return Sequential(
                        nn.LayerNorm(num_channels),
                        nn.Linear(num_channels, num_channels),
                        nn.GELU(),
                        nn.Linear(num_channels, num_channels),
                    )


def cross_attention_layer( num_q_channels: int, 
                           num_kv_channels: int, 
                           num_heads: int, 
                           dropout: float, 
                           activation_checkpoint: bool = False):
    
    layer = Sequential(
        Residual(CrossAttention(num_q_channels, num_kv_channels, num_heads, dropout), dropout),
        Residual(mlp(num_q_channels), dropout),
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)


def self_attention_layer(num_channels: int, 
                         num_heads: int, 
                         dropout: float, 
                         activation_checkpoint: bool = False):
    
    layer = Sequential(
        Residual(SelfAttention(num_channels, num_heads, dropout), dropout), 
        Residual(mlp(num_channels), dropout)
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)


def self_attention_block(num_layers: int, 
                         num_channels: int, 
                         num_heads: int, 
                         dropout: float, 
                         activation_checkpoint: bool = False
                        ):
    
    layers = [self_attention_layer(
                            num_channels, 
                            num_heads, 
                            dropout, 
                            activation_checkpoint) for _ in range(num_layers)]
    
    return Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]


class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_q_channels: int, 
                 num_kv_channels: int, 
                 num_heads: int, 
                 dropout: float):
        
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        return self.attention(x_q, x_kv, x_kv, key_padding_mask=pad_mask, attn_mask=attn_mask)[0]


class CrossAttention(nn.Module):
    
    def __init__(self, 
                 num_q_channels: int, 
                 num_kv_channels: int, 
                 num_heads: int, 
                 dropout: float):
        
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention(
                                            num_q_channels=num_q_channels, 
                                            num_kv_channels=num_kv_channels, 
                                            num_heads=num_heads, 
                                            dropout=dropout
                                            )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)


class SelfAttention(nn.Module):
    def __init__(self, num_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
                                            num_q_channels=num_channels, 
                                            num_kv_channels=num_channels, 
                                            num_heads=num_heads, 
                                            dropout=dropout
                                            )

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


class Encoder(nn.Module):
     
    def __init__(
        self,
        input_ch,
        preproc_ch,
        num_latents: int,
        num_latent_channels: int,
        num_layers: int = 3,
        num_cross_attention_heads: int = 4,
        num_self_attention_heads: int = 4,
        num_self_attention_layers_per_block: int = 6,
        dropout: float = 0.0,
        activation_checkpoint: bool = False,
    ):
       
        super().__init__()

        self.num_layers = num_layers
        
        # add a linear layer to expand input dims
        if preproc_ch:
            self.preproc = nn.Linear(input_ch, preproc_ch)
        else:
            self.preproc = None
            preproc_ch   = input_ch
            
        def create_layer():
            return Sequential(
                cross_attention_layer(
                    num_q_channels=num_latent_channels,
                    num_kv_channels=preproc_ch, 
                    num_heads=num_cross_attention_heads,
                    dropout=dropout,
                    activation_checkpoint=activation_checkpoint,
                ),
                self_attention_block(
                    num_layers=num_self_attention_layers_per_block,
                    num_channels=num_latent_channels,
                    num_heads=num_self_attention_heads,
                    dropout=dropout,
                    activation_checkpoint=activation_checkpoint,
                ),
            )

        self.layer_1 = create_layer()

        if num_layers > 1:
            # will be used recurrently depending on num_layers
            self.layer_n = create_layer()

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, pad_mask=None):
        
        b, *_ = x.shape
        
        if self.preproc:
            x = self.preproc(x)

        # repeat initial latent vector along batch dimension
        x_latent = repeat(self.latent, "... -> b ...", b=b)
         
        x_latent = self.layer_1(x_latent, x, pad_mask)
        for i in range(self.num_layers - 1):
            x_latent = self.layer_n(x_latent, x, pad_mask)

        return x_latent


class Decoder(nn.Module):
    def __init__(
        self,
        ff_channels: int,
        preproc_ch,
        num_latent_channels: int,
        latent_size,
        num_output_channels,
        num_latents: int,
        llm_embedding_dim: int = 4096,
        num_cross_attention_heads: int = 4,
        dropout: float = 0.0,
        activation_checkpoint: bool = False,
    ):
        
        super().__init__()

        self.num_latents = num_latents

        q_chan = ff_channels + num_latent_channels
        if preproc_ch:
            q_in = preproc_ch
        else:
            q_in = q_chan


        self.postproc_field = nn.Linear(q_in, num_output_channels)
        self.postproc_single = nn.Linear(q_in, num_output_channels)
        
        if preproc_ch:
            self.preproc = nn.Linear(q_chan, preproc_ch)
        else:
            self.preproc = None
            
        self.preproc_single = nn.Linear( num_latent_channels, preproc_ch)
        
        self.cross_attention = cross_attention_layer(
                                    num_q_channels=q_in,
                                    num_kv_channels=num_latent_channels*2,
                                    num_heads=num_cross_attention_heads,
                                    dropout=dropout,
                                    activation_checkpoint=activation_checkpoint,
                                )

        self.output_field  = nn.Parameter(torch.empty(latent_size,num_latent_channels))
        self.output_single = nn.Parameter(torch.empty(latent_size,num_latent_channels))
        
        self.linear_llm = nn.Linear(llm_embedding_dim,num_latent_channels)
        self.cross_attention_llm = cross_attention_layer(
                                    num_q_channels=num_latent_channels,
                                    num_kv_channels=num_latent_channels,
                                    num_heads=num_cross_attention_heads,
                                    dropout=dropout,
                                    activation_checkpoint=activation_checkpoint,
                                )

        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.output_field.normal_(0.0, 0.02).clamp_(-2.0, 2.0)
            self.output_single.normal_(0.0, 0.02).clamp_(-2.0, 2.0)


    def forward(self, x, coords, llm):
        # LLM
        llm = self.linear_llm(llm)
        llm_repeat_count = max(1, (self.num_latents + llm.shape[0] - 1) // llm.shape[0])
        llm = llm.repeat(llm_repeat_count, 1)[:self.num_latents]
        llm = self.cross_attention_llm(llm,llm).unsqueeze(0) 
        
        x = torch.cat((x, llm), dim=2)
            
        b, *_ = x.shape
        output_field = repeat(self.output_field, "... -> b ...", b=b)
        output_field = torch.repeat_interleave(output_field, coords.shape[1], axis=1)
        output_field = torch.cat([coords,output_field], axis=-1) 
        if self.preproc:
            output_field = self.preproc(output_field)
        output_field = self.cross_attention(output_field, x)

        output_single = repeat(self.output_single, "... -> b ...", b=b)
        if self.preproc:
            output_single = self.preproc_single(output_single)
        output_single = self.cross_attention(output_single, x)

        return self.postproc_field(output_field), self.postproc_single(output_single)