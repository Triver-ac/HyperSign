import torch
import os
from utils.misc import freeze_params, get_logger
from modelling.FusionNet.HyperNet import FusionNet
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
        self.FusionNet2_choice = cfg['FusionNet2'].get("choice",None)
        self.FusionNet2_global_local_reverse = cfg['FusionNet2']['global_local_reverse']
        self.FusionNet2_dropout = cfg['FusionNet2']['dropout']
        self.FusionNet2_AlignNet_choice = cfg['FusionNet2']['AlignNet_choice']
        
        print("type",self.type)
        if 'projection' in self.type:
            self.mapping = torch.nn.Sequential(
                torch.nn.Linear(in_features=projection_in_features, out_features=out_features),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=out_features, out_features=out_features)
            )
        elif 'embedding' in self.type:
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
        elif 'both' in self.type:
            self.ln = torch.nn.LayerNorm(out_features)
            self.projection_mapping = torch.nn.Sequential(
                torch.nn.Linear(in_features=projection_in_features, out_features=out_features),
                torch.nn.GELU(),
                torch.nn.Linear(in_features=out_features, out_features=out_features)
            )

            self.embed_mapping = torch.nn.Linear(
                in_features=embedding_in_feature,
                out_features=out_features,
                bias=False
            )
            assert embedding_in_feature == len(gloss_id2str), (embedding_in_feature, gloss_id2str)
            with torch.no_grad():
                for i, s in gloss_id2str.items():
                    if s in gls2embed:
                        self.embed_mapping.weight[:, i] = gls2embed[s]
                    else:
                        logger.info('{} not in gls2embed, set fc to zero'.format(s))
                        self.embed_mapping.weight[:, i] = 0
            if cfg['freeze']:
                logger.info('Freeze parameters in VLMapper ')
                freeze_params(self.mapping)

        # Check if 'separate' is in self.type and initialize the second mapping
        if 'separate' in self.type:
            if 'projection' in self.type:
                self.mapping2 = torch.nn.Sequential(
                    torch.nn.Linear(in_features=projection_in_features, out_features=out_features),
                    torch.nn.ReLU(),
                    torch.nn.Linear(in_features=out_features, out_features=out_features)
                )
            elif 'embedding' in self.type:
                self.mapping2 = torch.nn.Linear(
                    in_features=embedding_in_feature,
                    out_features=out_features,
                    bias=False
                )
                assert embedding_in_feature == len(gloss_id2str), (embedding_in_feature, gloss_id2str)
                assert gls2embed is not None
                logger.info("VL-Mapper type is embedding, so initialize VL-Mapper2 with gls2embed.")
                with torch.no_grad():
                    for i, s in gloss_id2str.items():
                        if s in gls2embed:
                            self.mapping2.weight[:, i] = gls2embed[s]
                        else:
                            logger.info('[Initialize VL-Mapper2] {} not in gls2embed, set fc to zero'.format(s))
                            self.mapping2.weight[:, i] = 0.
                if cfg['freeze']:
                    logger.info('Freeze parameters in VLMapper2 ')
                    freeze_params(self.mapping2)
            elif 'both' in self.type:
                self.projection_mapping2 = torch.nn.Sequential(
                    torch.nn.Linear(in_features=projection_in_features, out_features=out_features),
                    torch.nn.GELU(),
                    torch.nn.Linear(in_features=out_features, out_features=out_features)
                )
                self.embed_mapping2 = torch.nn.Linear(
                    in_features=embedding_in_feature,
                    out_features=out_features,
                    bias=False
                )
                assert embedding_in_feature == len(gloss_id2str), (embedding_in_feature, gloss_id2str)
                assert gls2embed is not None
                logger.info("VL-Mapper type is both, so initialize embed_mapping2 with gls2embed.")
                with torch.no_grad():
                    for i, s in gloss_id2str.items():
                        if s in gls2embed:
                            self.embed_mapping2.weight[:, i] = gls2embed[s]
                        else:
                            logger.info('{} not in gls2embed, set fc to zero'.format(s))
                            self.embed_mapping2.weight[:, i] = 0.
                if cfg['freeze']:
                    logger.info('Freeze parameters in embed_mapping2')
                    freeze_params(self.embed_mapping2)

        if self.FusionNet2_choice !=-1:
            self.FusionNet = FusionNet(choice=self.FusionNet2_choice, global_dim=out_features, local_dim=out_features, dropout = self.FusionNet2_dropout, AlignNet_choice=self.FusionNet2_AlignNet_choice).cuda()
      
    def forward(self, visual_outputs, lengths=None):
        if 'nonfuse' in self.type:
            if 'projection' in self.type:
                if 'rgb_gloss_feature' in visual_outputs:
                    visual_input = visual_outputs['rgb_gloss_feature']
                else:
                    visual_input = visual_outputs['gloss_feature']
                output = self.mapping(visual_input)
            elif 'embedding' in self.type:
                output = self.mapping(visual_outputs['gloss_probabilities']) #KeyError: 'gloss_probabilities'
            elif 'both' in self.type:
                output = self.ln(
                    self.projection_mapping(visual_outputs['gloss_feature'])
                    + self.embed_mapping(visual_outputs['gloss_probabilities'])
                )
            else:
                raise ValueError
        elif 'fuse' in self.type : #fuse
            if 'share' in self.type:
                if 'projection' in self.type:
                    rgb_output = self.mapping(visual_outputs['rgb_gloss_feature'])
                    keypoint_output = self.mapping(visual_outputs['keypoint_gloss_feature'])
                elif 'embedding' in self.type:
                    rgb_output = self.mapping(visual_outputs['rgb_gloss_probabilities']) #KeyError: 'gloss_probabilities'
                    keypoint_output = self.mapping(visual_outputs['keypoint_gloss_probabilities']) 
                elif 'both' in self.type:
                    rgb_output = self.ln(
                        self.projection_mapping(visual_outputs['rgb_gloss_feature'])
                        + self.embed_mapping(visual_outputs['rgb_gloss_probabilities'])
                    )
                    keypoint_output = self.ln(
                        self.projection_mapping(visual_outputs['keypoint_gloss_feature'])
                        + self.embed_mapping(visual_outputs['keypoint_gloss_probabilities'])
                    )
                else:
                    raise ValueError
            else: #separate
                if 'projection' in self.type:
                    rgb_output = self.mapping(visual_outputs['rgb_gloss_feature'])
                    keypoint_output = self.mapping2(visual_outputs['keypoint_gloss_feature'])
                elif 'embedding' in self.type:
                    rgb_output = self.mapping(visual_outputs['rgb_gloss_probabilities']) #KeyError: 'gloss_probabilities'
                    keypoint_output = self.mapping2(visual_outputs['keypoint_gloss_probabilities']) 
                elif 'both' in self.type:
                    rgb_output = self.ln(
                        self.projection_mapping(visual_outputs['rgb_gloss_feature'])
                        + self.embed_mapping(visual_outputs['rgb_gloss_probabilities'])
                    )
                    keypoint_output = self.ln(
                        self.projection_mapping2(visual_outputs['keypoint_gloss_feature'])
                        + self.embed_mapping2(visual_outputs['keypoint_gloss_probabilities'])
                    )
                else:
                    raise ValueError
            if self.FusionNet2_choice == -1:
                output = rgb_output + keypoint_output
            else:
                if not self.FusionNet2_global_local_reverse:
                    output = self.FusionNet(global_input=rgb_output, local_input=keypoint_output)
                else:
                    output = self.FusionNet(global_input=keypoint_output, local_input=rgb_output)
                
        result = {k: v for k, v in {
                "output": output if 'output' in locals() else None,
                "rgb_feature": rgb_output if 'rgb_output' in locals() else None,
                "keypoint_feature": keypoint_output if 'keypoint_output' in locals() else None
            }.items() if v is not None}
        return result