import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import MBartConfig
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.models.mbart.modeling_mbart import MBartAttention, MBartPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class GaussianNet(nn.Module):
    def __init__(self, variational_cfg, variator_cfg):
        super().__init__()

        self.in_out_dim = variational_cfg['input_embed_dim']
        self.latent_dim = variational_cfg['latent_dim']

        self.prior_net = nn.Linear(self.in_out_dim, self.latent_dim * 2)
        self.posterior_net = nn.Linear(self.in_out_dim, self.latent_dim * 2)
        self.recover_layer = nn.Linear(self.latent_dim, self.in_out_dim)
        self.norm = variational_cfg.get("norm", "prefix")
        self.ln = nn.LayerNorm(self.in_out_dim)

        self.prior_variator = Attention(variator_cfg, variational_cfg)
        if variational_cfg.get("attention_shared", True):
            self.posterior_variator = self.prior_variator
        else:
            self.posterior_variator = Attention(variator_cfg, variational_cfg)

    def forward(
            self,
            prior_sign_encoder_out,
            posterior_sign_encoder_out,
            posterior_text_encoder_out,
            prior_sign_attnention_mask,
            posterior_sign_attnention_mask,
            posterior_text_attnention_mask,
    ):
        # prior
        prior_sign_rep = self.prior_variator.forward(
            inputs_embeds=prior_sign_encoder_out,  # q,k,v
            attention_mask=prior_sign_attnention_mask,
            encoder_hidden_states=prior_sign_encoder_out,  # key/value
            encoder_attention_mask=prior_sign_attnention_mask,  # key/value mask
            # return_dict=True,
        )[0]
        prior_residual = prior_sign_encoder_out
        prior_mean, prior_logvar = self.prior_net(prior_sign_rep).chunk(2, dim=-1)
        prior_z = GaussianNet.reparameterize(
            prior_mean, prior_logvar, is_logv=True,
            temperature=1.0 if self.training else 0.0
        )
        prior_recover = self.combine(prior_residual, prior_z)

        # posterior
        posterior_rep = self.posterior_variator.forward(
            inputs_embeds=posterior_sign_encoder_out,  # query
            attention_mask=posterior_sign_attnention_mask,  # query mask
            encoder_hidden_states=posterior_text_encoder_out,  # key/value
            encoder_attention_mask=posterior_text_attnention_mask,
            # return_dict=True,
        )[0]
        posterior_residual = posterior_sign_encoder_out
        delta_posterior_mean, delta_posterior_logvar = self.posterior_net(posterior_rep).chunk(2, dim=-1)
        posterior_mean = prior_mean + delta_posterior_mean
        posterior_logvar = prior_logvar + delta_posterior_logvar
        posterior_z = GaussianNet.reparameterize(
            posterior_mean, posterior_logvar, is_logv=True,
            temperature=1.0 if self.training else 0.0
        )
        posterior_recover = self.combine(posterior_residual, posterior_z)

        return {
            "posterior": {
                "mean": posterior_mean, "logvar": posterior_logvar, "z": posterior_z,
                "encoder_out": posterior_recover,
            } if posterior_sign_encoder_out is not None else None,
            "prior": {
                "mean": prior_mean, "logvar": prior_logvar, "z": prior_z,
                "encoder_out": prior_recover,
            } if prior_sign_encoder_out is not None else None,
        }

    def combine(self, inputs, z):
        z_recover = self.ln(self.recover_layer(z)) if self.norm == "prefix" else self.recover_layer(z)
        outputs = self.ln(inputs + z_recover) if self.norm == "postfix" else inputs + z_recover
        return outputs

    @staticmethod
    def reparameterize(mean, var, is_logv=False, sample_size=1, temperature=1.0):
        if sample_size > 1:
            mean = mean.contiguous().unsqueeze(1).expand(-1, sample_size, -1).reshape(-1, mean.size(-1))
            var = var.contiguous().unsqueeze(1).expand(-1, sample_size, -1).reshape(-1, var.size(-1))

        if not is_logv:
            sigma = torch.sqrt(var + 1e-10)
        else:
            sigma = torch.exp(0.5 * var)

        epsilon = torch.randn_like(sigma)
        z = mean + epsilon * sigma * temperature
        return z


class Attention(MBartPreTrainedModel):
    def __init__(self, config: MBartConfig, variator_cfg: dict):
        super().__init__(config)
        self.embed_dim = config.d_model
        self.dropout = variator_cfg['dropout'] if variator_cfg.get("dropout", None) is not None else config.dropout
        self.attn = MBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # work as query
        attention_mask: Optional[torch.Tensor] = None,  # query mask, not used since only cross-attn exist.
        encoder_hidden_states: Optional[torch.Tensor] = None,   # work as key/value
        encoder_attention_mask: Optional[torch.Tensor] = None,  # key/value mask
        input_ids=None,  # not used
        layer_head_mask: Optional[torch.Tensor] = None,  # not used
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,  # not used
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # not used
        output_attentions: Optional[bool] = False,  # not used
        use_cache: Optional[bool] = True,  # not used
        past_key_values=None,  # not used
        return_dict=None,
    ):
        """
        Args:
            inputs_embeds (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        """
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        hidden_states = inputs_embeds

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (cross_attn_weights,)

        if not return_dict:
            return outputs
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=cross_attn_weights
        )


def gaussian_kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar):
    kl_loss = -0.5 * torch.sum(
        1 + (posterior_logvar - prior_logvar)
        - torch.div(
            torch.pow(prior_mean - posterior_mean, 2) + posterior_logvar.exp(),
            prior_logvar.exp(),
        )
    )
    return kl_loss
