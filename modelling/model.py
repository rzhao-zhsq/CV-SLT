import torch
from utils.misc import get_logger
from modelling.recognition import RecognitionNetwork
from modelling.translation import TranslationNetwork, VariationalTranslationNetwork
from modelling.translation_ensemble import TranslationNetwork_Ensemble
from modelling.vl_mapper import VLMapper


class VariationalSignLanguageModel(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.logger = get_logger()
        self.task = cfg['task']
        self.device = cfg['device']
        model_cfg = cfg['model']
        self.frozen_modules = []

        if self.task == 'S2G':
            self.text_tokenizer = None
            self.recognition_network = RecognitionNetwork(
                cfg=model_cfg['RecognitionNetwork'],
                input_type='video',
                transform_cfg=cfg['data']['transform_cfg'],
                input_streams=cfg['data'].get('input_streams', 'rgb')
            )
            self.gloss_tokenizer = self.recognition_network.gloss_tokenizer

            if self.recognition_network.visual_backbone is not None:
                self.frozen_modules.extend(self.recognition_network.visual_backbone.get_frozen_layers())
            if self.recognition_network.visual_backbone_keypoint is not None:
                self.frozen_modules.extend(self.recognition_network.visual_backbone_keypoint.get_frozen_layers())

        elif self.task == 'G2T':
            self.translation_network = TranslationNetwork(
                input_type='gloss',
                cfg=model_cfg['TranslationNetwork'],
                task=self.task
            )
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.gloss_tokenizer = self.translation_network.gloss_tokenizer  # G2T

        elif self.task == 'S2T':
            # --------------------------- for recognition --------------------------- #
            self.recognition_weight = model_cfg.get('recognition_weight', 0.0)
            self.recognition_network = RecognitionNetwork(
                cfg=model_cfg['RecognitionNetwork'],
                input_type='feature',
                input_streams=cfg['data'].get('input_streams', 'rgb'),
                transform_cfg=cfg['data'].get('transform_cfg', {})
            )
            self.gloss_tokenizer = self.recognition_network.gloss_tokenizer
            if model_cfg['RecognitionNetwork'].get('freeze', False):
                self.logger.info('freeze recognition_network')
                self.frozen_modules.append(self.recognition_network)
                for param in self.recognition_network.parameters():
                    param.requires_grad = False
                self.recognition_network.eval()
            # --------------------------- for translation --------------------------- #
            self.translation_weight = model_cfg.get('translation_weight', 1.0)
            input_type = model_cfg['TranslationNetwork'].pop('input_type', 'feature')
            self.translation_network = VariationalTranslationNetwork(
                input_type=input_type,
                translation_cfg=model_cfg['TranslationNetwork'],
                variation_cfg=model_cfg['VariationalNetwork'],
                task=self.task
            )
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.frozen_modules.extend(self.translation_network.frozen_modules)
            # --------------------------- for Visual language Mapper --------------------------- #
            if 'in_features' in model_cfg['VLMapper']:
                projection_in_features = model_cfg['VLMapper'].pop('in_features')
            else:
                projection_in_features = model_cfg['RecognitionNetwork']['visual_head']['hidden_size']
            self.vl_mapper = VLMapper(
                cfg=model_cfg['VLMapper'],
                projection_in_features=projection_in_features,
                embedding_in_feature=len(self.gloss_tokenizer),
                out_features=self.translation_network.input_dim,
                gloss_id2str=self.gloss_tokenizer.id2gloss,
                gls2embed=getattr(self.translation_network, 'gls2embed', None),
            )
        elif self.task == 'S2T_Ensemble':
            self.recognition_weight = 0
            self.translation_network = TranslationNetwork_Ensemble(
                cfg=model_cfg['TranslationNetwork_Ensemble']
            )
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.gloss_tokenizer = None

    def set_num_updates(self, num_updates):
        self.update_num = num_updates
        if hasattr(self, "translation_network") is not None:
            self.translation_network.set_num_updates(num_updates)

    def forward(self, is_train, translation_inputs={}, recognition_inputs={}, global_step=None, **kwargs):
        if is_train and hasattr(self, "set_num_updates"):
            self.set_num_updates(global_step)
        if self.task == 'S2G':
            model_outputs = self.recognition_network.forward(is_train=is_train, **recognition_inputs)
            model_outputs['total_loss'] = model_outputs['recognition_loss']
        elif self.task == 'G2T':
            model_outputs = self.translation_network.forward(**translation_inputs)
            model_outputs['total_loss'] = model_outputs['translation_loss']
        elif self.task == 'S2T':
            recognition_outputs = self.recognition_network.forward(is_train=is_train, **recognition_inputs)
            mapped_feature = self.vl_mapper.forward(visual_outputs=recognition_outputs)
            translation_inputs = {
                **translation_inputs,
                'input_feature': mapped_feature,
                'input_lengths': recognition_outputs['input_lengths']
            }
            translation_outputs = self.translation_network.forward(**translation_inputs)
            model_outputs = {**translation_outputs, **recognition_outputs}
            model_outputs['transformer_inputs'] = model_outputs['transformer_inputs']  # for latter use of decoding
            model_outputs['total_loss'] = (
                model_outputs['recognition_loss'] * self.recognition_weight
                + model_outputs['translation_loss'] * self.translation_weight
            )
        elif self.task == 'S2T_Ensemble':
            assert 'inputs_embeds_list' in translation_inputs and 'attention_mask_list' in translation_inputs
            assert len(translation_inputs['inputs_embeds_list']) == len(self.translation_network.model.model_list)
            model_outputs = self.translation_network.forward(**translation_inputs)
        return model_outputs

    def generate_txt(self, transformer_inputs=None, generate_cfg={}, **kwargs):
        model_outputs = self.translation_network.generate(**transformer_inputs, **generate_cfg)
        return model_outputs

    def predict_gloss_from_logits(self, gloss_logits, beam_size, input_lengths):
        return self.recognition_network.decode(
            gloss_logits=gloss_logits,
            beam_size=beam_size,
            input_lengths=input_lengths)

    def set_train(self):
        self.train()
        for m in self.frozen_modules:
            m.eval()

    def set_eval(self):
        self.eval()


def build_model(cfg):
    model = VariationalSignLanguageModel(cfg)
    return model.to(cfg['device'])
