import math
import copy
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration, MBartConfig
from transformers.modeling_outputs import BaseModelOutput, ModelOutput

from utils.loss import XentLoss
from utils.misc import freeze_params, get_logger
from .Tokenizer import GlossTokenizer_G2T, TextTokenizer
from modelling.gaussian_net import GaussianNet, gaussian_kl_loss


class TranslationNetwork(torch.nn.Module):
    def __init__(self, input_type, cfg, task) -> None:
        super().__init__()
        self.frozen_modules = []
        self.logger = get_logger()
        self.task = task
        self.input_type = input_type
        assert self.input_type in ['gloss', 'feature', 'text']
        self.text_tokenizer = TextTokenizer(tokenizer_cfg=cfg['TextTokenizer'])

        if 'pretrained_model_name_or_path' in cfg:
            self.logger.info('Initialize translation network from {}'.format(cfg['pretrained_model_name_or_path']))
            self.model = MBartForConditionalGeneration.from_pretrained(
                cfg['pretrained_model_name_or_path'],
                **cfg.get('overwrite_cfg', {})
            )
        elif 'model_config' in cfg:
            self.logger.info('Train translation network from scratch using config={}'.format(cfg['model_config']))
            config = MBartConfig.from_pretrained(cfg['model_config'])
            for k, v in cfg.get('overwrite_cfg', {}).items():
                setattr(config, k, v)
                self.logger.info('Overwrite {}={}'.format(k, v))
            if cfg['TextTokenizer'].get('level', 'sentencepiece') == 'word':
                setattr(config, 'vocab_size', len(self.text_tokenizer.id2token))
                self.logger.info('Vocab_size {}'.format(config.vocab_size))
            self.model = MBartForConditionalGeneration(config=config)

            if 'pretrained_pe' in cfg:
                pe = torch.load(cfg['pretrained_pe']['pe_file'], map_location='cpu')
                self.logger.info('Load pretrained positional embedding from ', cfg['pretrained_pe']['pe_file'])
                with torch.no_grad():
                    self.model.model.encoder.embed_positions.weight = torch.nn.parameter.Parameter(
                        pe['model.encoder.embed_positions.weight']
                    )
                    self.model.model.decoder.embed_positions.weight = torch.nn.parameter.Parameter(
                        pe['model.decoder.embed_positions.weight']
                    )
                if cfg['pretrained_pe']['freeze']:
                    self.logger.info('Set positional embedding frozen')
                    freeze_params(self.model.model.encoder.embed_positions)
                    freeze_params(self.model.model.decoder.embed_positions)
                else:
                    self.logger.info('Set positional embedding trainable')
        else:
            raise ValueError

        self.translation_loss_fun = XentLoss(
            pad_index=self.text_tokenizer.pad_index,
            smoothing=cfg['label_smoothing']
        )
        self.input_dim = self.model.config.d_model
        self.input_embed_scale = cfg.get('input_embed_scale', math.sqrt(self.model.config.d_model))

        if self.task in ['S2T', 'G2T'] and 'pretrained_model_name_or_path' in cfg:
            # in both S2T or G2T, we need gloss_tokenizer and gloss_embedding
            self.gloss_tokenizer = GlossTokenizer_G2T(tokenizer_cfg=cfg['GlossTokenizer'])
            self.gloss_embedding = self.build_gloss_embedding(**cfg['GlossEmbedding'])
            # debug
            self.gls_eos = cfg.get('gls_eos', 'gls')  # gls or txt
        elif self.task in ['S2T_glsfree']:
            self.gls_eos = None
            self.gloss_tokenizer, self.gloss_embedding = None, None
        elif 'pretrained_model_name_or_path' not in cfg:
            self.gls_eos = 'txt'
            self.gloss_tokenizer, self.gloss_embedding = None, None
        else:
            raise ValueError

        if cfg.get('from_scratch', False):
            self.model.init_weights()
            self.logger.info('Build Translation Network with scratch config!')
        if cfg.get('freeze_txt_embed', False):
            freeze_params(self.model.model.shared)
            self.logger.info('Set txt embedding frozen')

        if 'load_ckpt' in cfg:
            self.load_from_pretrained_ckpt(cfg['load_ckpt'])

    def load_from_pretrained_ckpt(self, pretrained_ckpt):
        logger = get_logger()
        logger.info(
            'Loading and Reinitializing Translation network from pretrained ckpt {}'.format(pretrained_ckpt)
        )
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
        load_dict = {}
        for k, v in checkpoint.items():
            if 'translation_network' in k:
                load_dict[k.replace('translation_network.', '')] = v
        self.load_state_dict(load_dict)

    def build_gloss_embedding(self, gloss2embed_file, from_scratch=False, freeze=False):
        gloss_embedding = torch.nn.Embedding(
            num_embeddings=len(self.gloss_tokenizer.id2gloss),
            embedding_dim=self.model.config.d_model,
            padding_idx=self.gloss_tokenizer.gloss2id['<pad>']
        )
        self.logger.info('gloss2embed_file ' + gloss2embed_file)
        if from_scratch:
            self.logger.info('Train Gloss Embedding from scratch')
            assert freeze is False
        else:
            gls2embed = torch.load(gloss2embed_file)
            self.gls2embed = gls2embed
            self.logger.info('Initialize gloss embedding from {}'.format(gloss2embed_file))
            with torch.no_grad():
                for id_, gls in self.gloss_tokenizer.id2gloss.items():
                    if gls in gls2embed:
                        assert gls in gls2embed, gls
                        gloss_embedding.weight[id_, :] = gls2embed[gls]
                    else:
                        self.logger.info('{} not in gls2embed train from scratch'.format(gls))

        if freeze:
            freeze_params(gloss_embedding)
            self.logger.info('Set gloss embedding frozen')
        return gloss_embedding

    def prepare_gloss_inputs(self, input_ids):
        input_emb = self.gloss_embedding(input_ids) * self.input_embed_scale
        return input_emb

    def prepare_feature_inputs(self, input_feature, input_lengths, gloss_embedding=None, gloss_lengths=None):
        if self.task == 'S2T_glsfree':
            suffix_len = 0
            suffix_embedding = None
        else:
            if self.gls_eos == 'gls':
                assert self.gloss_embedding is not None
                # add </s> embedding tag to the tail of input_feature.
                suffix_embedding = [self.gloss_embedding.weight[self.gloss_tokenizer.convert_tokens_to_ids('</s>'), :]]
            else:  # self.gls_eos == 'txt':
                # add <src_lang> embedding tag to the tail of input_feature.
                suffix_embedding = [self.model.model.shared.weight[self.text_tokenizer.eos_index, :]]
            if self.task in ['S2T', 'G2T']:
                if self.gls_eos == 'gls':
                    assert self.gloss_embedding is not None
                    src_lang_code_embedding = self.gloss_embedding.weight[ \
                                              self.gloss_tokenizer.convert_tokens_to_ids(self.gloss_tokenizer.src_lang),
                                              :]  # to-debug
                else:  # self.gls_eos == 'txt':
                    src_lang_id = self.text_tokenizer.lang_index
                    src_lang_code_embedding = self.model.model.shared.weight[src_lang_id, :]
                suffix_embedding.append(src_lang_code_embedding)
            suffix_len = len(suffix_embedding)
            suffix_embedding = torch.stack(suffix_embedding, dim=0)

        max_length = torch.max(input_lengths) + suffix_len
        inputs_embeds = []
        attention_mask = torch.zeros(
            [input_feature.shape[0], max_length],
            dtype=torch.long,
            device=input_feature.device
        )
        # concat the suffix_embedding and original input_feature, and prepare the padding mask.
        for ii, feature in enumerate(input_feature):
            valid_len = input_lengths[ii]
            if 'gloss+feature' in self.input_type:
                valid_feature = torch.cat(
                    [gloss_embedding[ii, :gloss_lengths[ii], :], feature[:valid_len - gloss_lengths[ii], :]],
                    dim=0
                )
            else:
                valid_feature = feature[:valid_len, :]  # t,D
            if suffix_embedding is not None:
                feature_w_suffix = torch.cat([valid_feature, suffix_embedding], dim=0)  # t+2, D
            else:
                feature_w_suffix = valid_feature
            if feature_w_suffix.shape[0] < max_length:
                pad_len = max_length - feature_w_suffix.shape[0]
                padding = torch.zeros(
                    [pad_len, feature_w_suffix.shape[1]],
                    dtype=feature_w_suffix.dtype,
                    device=feature_w_suffix.device
                )
                padded_feature_w_suffix = torch.cat([feature_w_suffix, padding], dim=0)  # t+2+pl,D
                inputs_embeds.append(padded_feature_w_suffix)
            else:
                inputs_embeds.append(feature_w_suffix)
            attention_mask[ii, :valid_len + suffix_len] = 1
        transformer_inputs = {
            'inputs_embeds': torch.stack(inputs_embeds, dim=0) * self.input_embed_scale,  # B,T,D
            'attention_mask': attention_mask  # attention_mask
        }
        return transformer_inputs

    def forward(self, **kwargs):
        if self.input_type == 'gloss':
            kwargs.pop('text_length', None)
            input_ids = kwargs.pop('input_ids')
            kwargs['inputs_embeds'] = self.prepare_gloss_inputs(input_ids)
        elif self.input_type == 'feature':
            input_feature = kwargs.pop('input_feature')
            input_lengths = kwargs.pop('input_lengths')
            # quick fix
            kwargs.pop('input_ids', None)
            kwargs.pop('text_length', None)
            kwargs.pop('gloss_ids', None)
            kwargs.pop('gloss_lengths', None)
            new_kwargs = self.prepare_feature_inputs(input_feature, input_lengths)
            kwargs = {**kwargs, **new_kwargs}
        else:
            raise ValueError
        output_dict = self.model(**kwargs, output_hidden_states=None if self.training else True, return_dict=True)
        # print(output_dict.keys()) loss, logits, past_key_values, encoder_last_hidden_state
        log_prob = torch.nn.functional.log_softmax(output_dict['logits'], dim=-1)  # B, T, L
        batch_loss_sum = self.translation_loss_fun(log_probs=log_prob, targets=kwargs['labels'])
        output_dict['translation_loss'] = batch_loss_sum / log_prob.shape[0]

        output_dict['transformer_inputs'] = kwargs  # for later use (decoding)
        return output_dict

    def generate(
            self,
            input_ids=None, attention_mask=None,  # decoder_input_ids,
            inputs_embeds=None, input_lengths=None,
            num_beams=4, max_length=100, length_penalty=1, **kwargs
    ):
        assert attention_mask is not None
        batch_size = attention_mask.shape[0]
        decoder_input_ids = torch.ones(
            [batch_size, 1], dtype=torch.long,
            device=attention_mask.device
        ) * self.text_tokenizer.sos_index
        assert inputs_embeds is not None and attention_mask is not None
        output_dict = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,  # same with forward
            decoder_input_ids=decoder_input_ids,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_length=max_length,
            return_dict_in_generate=True
        )
        output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'])
        return output_dict


class VariationalTranslationNetwork(TranslationNetwork):
    def __init__(self, input_type, translation_cfg, variation_cfg, task) -> None:
        super().__init__(input_type, translation_cfg, task)
        self.gls_eos = "txt"
        self.encoder = self.model.get_encoder()

        self.gaussian_net = GaussianNet(variation_cfg, self.model.config)

        self.prior_weight = variation_cfg.get('prior_weight', 1.0)
        self.gkl_factor = variation_cfg.get('gkl_factor', 1.0)
        self.gkl_step_scheduler = StepWarmUpScheduler(
            start_ratio=variation_cfg.get("gkl_start_ratio", 0.0),
            end_ratio=self.gkl_factor,
            warmup_start_step=variation_cfg.get("gkl_warmup_start", 0),
            warmup_step=variation_cfg.get("gkl_warmup_step", 4000),
        )
        self.kl_factor = variation_cfg.get('kl_factor', 1.0)
        self.kl_step_scheduler = StepWarmUpScheduler(
            start_ratio=variation_cfg.get("kl_start_ratio", 0.0),
            end_ratio=self.kl_factor,
            warmup_start_step=variation_cfg.get("kl_warmup_start", 0),
            warmup_step=variation_cfg.get("kl_warmup_step", 4000),
        )

        if hasattr(self, "gloss_embedding"):
            delattr(self, "gloss_embedding")
        if hasattr(self, "gloss_tokenizer"):
            delattr(self, "gloss_tokenizer")

    def set_num_updates(self, num_updates):
        self.kl_factor = self.kl_step_scheduler.forward(num_updates)
        self.gkl_factor = self.gkl_step_scheduler.forward(num_updates)

    def prepare_gaussian_net_feature_inputs(self, sign_embeds, sign_mask, text=None):
        text_embeds = self.model.model.shared(text) * self.input_embed_scale
        text_mask = text.ne(self.text_tokenizer.pad_index)
        transformer_inputs = {
            'inputs_embeds': torch.cat([sign_embeds, text_embeds], dim=1),  # [B, T_sign + T_text, D]
            'attention_mask': torch.cat([sign_mask, text_mask], dim=1),  # attention_mask
        }
        return transformer_inputs

    def _compute_kl_loss(self, prior_out, posterior_out):
        kl_1 = F.kl_div(prior_out.log_softmax(-1), posterior_out.softmax(-1), reduction="sum")
        kl_2 = F.kl_div(posterior_out.log_softmax(-1), prior_out.softmax(-1), reduction="sum")
        kl_loss = (kl_1 + kl_2) / 2
        return kl_loss

    def _compute_gaussian_kl_loss(self, posterior_mean, posterior_logvar, prior_mean, prior_logvar):

        gkl_loss = gaussian_kl_loss(
            posterior_mean=posterior_mean, posterior_logvar=posterior_logvar,
            prior_mean=prior_mean, prior_logvar=prior_logvar,
        )
        return gkl_loss

    def forward(self, **kwargs):
        # quick fix
        kwargs.pop('gloss_ids', None)
        kwargs.pop('gloss_lengths', None)

        input_feature, input_lengths = kwargs.pop('input_feature'), kwargs.pop('input_lengths')
        kwargs.pop('text_length', None)
        kwargs.pop('input_ids', None)

        encoder_kwargs = self.prepare_feature_inputs(input_feature, input_lengths)
        kwargs = {**kwargs, **encoder_kwargs}

        posterior_encoder_kwargs = self.prepare_gaussian_net_feature_inputs(
            kwargs["inputs_embeds"], kwargs["attention_mask"], text=kwargs['labels']
        )
        prior_encoder_output_dict = self.encoder(**encoder_kwargs, return_dict=True,)
        posterior_encodet_output_dict = self.encoder(**posterior_encoder_kwargs, return_dict=True)
        sign_max_length = kwargs["inputs_embeds"].size(1)
        prior_sign_encoder_out = prior_encoder_output_dict["last_hidden_state"]
        posterior_sign_encoder_out, posterior_text_encoder_out = (
            posterior_encodet_output_dict["last_hidden_state"][:, :sign_max_length, :],
            posterior_encodet_output_dict["last_hidden_state"][:, sign_max_length:, :],
        )
        # gaussian net output.
        gaussian_out = self.gaussian_net(
            prior_sign_encoder_out=prior_sign_encoder_out,
            posterior_sign_encoder_out=posterior_sign_encoder_out,
            posterior_text_encoder_out=posterior_text_encoder_out,
            prior_sign_attnention_mask=kwargs["attention_mask"],
            posterior_sign_attnention_mask=posterior_encoder_kwargs['attention_mask'][:, :sign_max_length],
            posterior_text_attnention_mask=posterior_encoder_kwargs["attention_mask"][:, sign_max_length:],
        )

        prior_encoder_outputs = BaseModelOutput(last_hidden_state=gaussian_out["prior"]["encoder_out"])
        posterior_encoder_outputs = BaseModelOutput(last_hidden_state=gaussian_out["posterior"]["encoder_out"])

        prior_output_dict = self.model(**kwargs, return_dict=True, encoder_outputs=prior_encoder_outputs)
        posterior_output_dict = self.model(**kwargs, return_dict=True, encoder_outputs=posterior_encoder_outputs)

        prior_batch_loss_sum = self.translation_loss_fun(
            log_probs=prior_output_dict['logits'].log_softmax(-1), targets=kwargs['labels']
        )
        posterior_batch_loss_sum = self.translation_loss_fun(
            log_probs=posterior_output_dict['logits'].log_softmax(-1), targets=kwargs['labels']
        )

        # Tips: kl loss and gkl loss
        kl_loss = self._compute_kl_loss(
            prior_out=prior_output_dict["logits"], posterior_out=posterior_output_dict["logits"],
        )

        gkl_loss = self._compute_gaussian_kl_loss(
            posterior_mean=gaussian_out["posterior"]["mean"], posterior_logvar=gaussian_out["posterior"]["logvar"],
            prior_mean=gaussian_out["prior"]["mean"], prior_logvar=gaussian_out["prior"]["logvar"],
        )
        sample_size = kwargs['labels'].size(0)
        prior_output_dict['posterior_translation_loss'] = posterior_batch_loss_sum / sample_size
        prior_output_dict['prior_translation_loss'] = prior_batch_loss_sum / sample_size * self.prior_weight
        prior_output_dict["kl_loss"] = kl_loss / sample_size * self.kl_factor
        prior_output_dict["gkl_loss"] = gkl_loss / sample_size * self.gkl_factor

        prior_output_dict['translation_loss'] = (
                prior_output_dict['posterior_translation_loss']
                + prior_output_dict['prior_translation_loss']
                + prior_output_dict['gkl_loss']
                + prior_output_dict['kl_loss']
        )
        prior_output_dict['gkl_factor'], prior_output_dict['kl_factor'] = self.gkl_factor, self.kl_factor

        kwargs["encoder_outputs"] = prior_encoder_outputs
        prior_output_dict['transformer_inputs'] = kwargs
        prior_output_dict['posterior_encoder_outputs'] = posterior_encoder_outputs

        return prior_output_dict

    def generate(
            self,
            input_ids=None, attention_mask=None,  # decoder_input_ids,
            encoder_outputs=None,
            inputs_embeds=None, input_lengths=None,
            num_beams=4, max_length=100, length_penalty=1, **kwargs
    ):
        assert attention_mask is not None
        assert encoder_outputs is not None  # to make sure the decoder input embeds is not None.
        assert inputs_embeds is not None
        batch_size = attention_mask.shape[0]
        decoder_input_ids = torch.ones(
            [batch_size, 1], dtype=torch.long,
            device=attention_mask.device
        ) * self.text_tokenizer.sos_index
        output_dict = self.model.generate(
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,  # same with forward
            decoder_input_ids=decoder_input_ids,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_length=max_length,
            return_dict_in_generate=True,
            # output_scores=True,
        )
        output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'])
        return output_dict


class StepWarmUpScheduler(object):
    def __init__(self, start_ratio, end_ratio, warmup_start_step, warmup_step):
        super().__init__()
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.warmup_start_step = warmup_start_step
        self.warmup_step = warmup_step + int(warmup_step == 0)
        self.step_ratio = (end_ratio - start_ratio) / self.warmup_step

    def forward(self, step_num):
        if step_num < self.warmup_start_step:
            return self.start_ratio
        elif step_num >= self.warmup_step:
            return self.end_ratio
        else:
            ratio = self.start_ratio + self.step_ratio * (step_num - self.warmup_start_step)
            return ratio

