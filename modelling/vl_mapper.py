import torch
import os
from utils.misc import freeze_params, get_logger


class VLMapper(torch.nn.Module):
    def __init__(
            self,
            cfg,
            projection_in_features,
            embedding_in_feature,
            out_features,
            gloss_id2str=None,
            gls2embed=None,
            freeze=False
    ) -> None:
        super().__init__()
        logger = get_logger()
        self.type = cfg.get('type', 'projection')
        if self.type == 'projection':
            self.mapping = torch.nn.Sequential(
                torch.nn.Linear(in_features=projection_in_features, out_features=out_features),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=out_features, out_features=out_features)
            )
        elif self.type == 'embedding':
            self.mapping = torch.nn.Linear(
                in_features=embedding_in_feature,
                out_features=out_features,
                bias=False
            )
            assert embedding_in_feature == len(gloss_id2str), (embedding_in_feature, gloss_id2str)
            assert gls2embed is not None
            logger.info("VL-Mapper type is embedding, so initialize VL-Mapper with gls2embed.")
            with torch.no_grad():
                for i, s in gloss_id2str.items():
                    if s in gls2embed:
                        self.mapping.weight[:, i] = gls2embed[s]
                    else:
                        logger.info('[Initialize VL-Mapper] {} not in gls2embed, set fc to zero'.format(s))
                        self.mapping.weight[:, i] = 0.
            if cfg['freeze']:
                logger.info('Freeze parameters in VLMapper ')
                freeze_params(self.mapping)

    def forward(self, visual_outputs):
        if self.type == 'projection':
            output = self.mapping(visual_outputs['gloss_feature'])
        elif self.type == 'embedding':
            output = self.mapping(visual_outputs['gloss_probabilities'])
        else:
            raise ValueError
        return output
