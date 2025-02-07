import math
import copy
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration, MBartConfig
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from scipy.stats import truncnorm

from utils.loss import XentLoss
from utils.misc import freeze_params, get_logger
from .Tokenizer import GlossTokenizer_G2T, TextTokenizer
from modelling.SST.SST_network import SST_network


class TranslationNetwork(torch.nn.Module):
    def __init__(self, input_type, cfg, task) -> None:
        super().__init__()
        self.frozen_modules = []
        self.logger = get_logger()
        self.task = task
        self.input_type = input_type
        assert self.input_type in ['gloss', 'feature', 'text']
        self.text_tokenizer = TextTokenizer(tokenizer_cfg=cfg['TextTokenizer'])
        self.sentence_avg = cfg.get("avg_level", "sentence") == "sentence"

        if 'pretrained_model_name_or_path' in cfg:
            self.logger.info('Initialize translation network from {}'.format(cfg['pretrained_model_name_or_path']))
            # self.model = MBartForConditionalGeneration.from_pretrained(
            #     cfg['pretrained_model_name_or_path'],
            #     **cfg.get('overwrite_cfg', {})
            # )
            if cfg.get("knn_mode", "vanilla") == "vanilla":
                self.model = MBartForConditionalGeneration.from_pretrained(
                    cfg['pretrained_model_name_or_path'],
                    **cfg.get('overwrite_cfg', {})
                )
            else:
                self.model = KNNMBartForConditionalGeneration.from_pretrained(
                    cfg['pretrained_model_name_or_path'],
                    **cfg.get('overwrite_cfg', {})
                )
                self.model.build_datastore(
                    cfg,
                    cfg['knn_datastore_path'],
                    vocab_size=self.model.model.shared.num_embeddings,
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
        elif self.task in ['S2T_glsfree', 'T2T']:
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
                    # self.text_tokenizer.pruneids[self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)]
                    # src_lang_id = self.text_tokenizer.pruneids[30]
                    # assert src_lang_id == 31
                    # src_lang_code_embedding = self.model.model.shared.weight[src_lang_id, :]
                    # raw_src_lang_id = self.text_tokenizer.tokenizer.convert_tokens_to_ids(
                    #     self.text_tokenizer.tokenizer.tgt_lang
                    # )
                    # src_lang_id = self.text_tokenizer.pruneids[raw_src_lang_id]
                    src_lang_id = self.text_tokenizer.lang_index
                    src_lang_code_embedding = self.model.model.shared.weight[src_lang_id, :]
                suffix_embedding.append(src_lang_code_embedding)
            suffix_len = len(suffix_embedding) #2
            suffix_embedding = torch.stack(suffix_embedding, dim=0) #(2,1024)

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
        # if self.input_type == 'text':
        #     kwargs.pop('text_length', None)
        #     kwargs['attention_mask'] = kwargs['input_ids'].ne(self.text_tokenizer.pad_index)
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


class HyperTranslationNetwork(TranslationNetwork):
    def __init__(self, input_type, translation_cfg, distillation_cfg, task, SST_cfg=None) -> None:
        super().__init__(input_type, translation_cfg, task)
        self.gls_eos = "txt"  # gls or txt
        self.encoder = self.model.get_encoder()
        # self.decoder = self.model.get_encoder()
        self.hidden_kl = distillation_cfg.get("hidden_kl", False)
        self.cat_eos = distillation_cfg.get("cat_eos", False)
        self.unidirection_kl = distillation_cfg.get("unidirection_kl", False)
        self.gkl_func_type = distillation_cfg.get("gkl_func_type", "gkl")
        self.gkl_weight = distillation_cfg.get('gkl_weight', 1.0)
        self.interpolate_type = distillation_cfg.get('interpolate_type', "interpolate")

        self.SST_mean_alpha = SST_cfg.get('SST_mean_alpha',0)
        self.SST_initial_std_dev = SST_cfg.get('SST_initial_std_dev', 0.5)
        self.SST_final_std_dev = SST_cfg.get('SST_final_std_dev',0.01)
        self.SST_current_step = 0
        self.SST_final_step = SST_cfg.get('SST_final_step',0)
        
        self.gkl_factor = distillation_cfg.get('gkl_factor', 1.0)
        self.gkl_step_scheduler = StepWarmUpScheduler(
            start_ratio=distillation_cfg.get("gkl_start_ratio", 0.0),
            end_ratio=self.gkl_factor,
            warmup_start_step=distillation_cfg.get("gkl_warmup_start", 0),
            warmup_step=distillation_cfg.get("gkl_warmup_step", 4000),
        )
        
        self.kl_factor = distillation_cfg.get('kl_factor', 1.0)
        self.kl_step_scheduler = StepWarmUpScheduler(
            start_ratio=distillation_cfg.get("kl_start_ratio", 0.0),
            end_ratio=self.kl_factor,
            warmup_start_step=distillation_cfg.get("kl_warmup_start", 0),
            warmup_step=distillation_cfg.get("kl_warmup_step", 4000),
        )
        self.sample_factor = distillation_cfg.get('sample_start_factor', 0.0)
        self.sample_step_scheduler = StepWarmUpScheduler(
            start_ratio=self.sample_factor,
            end_ratio=distillation_cfg.get('sample_end_factor', 0.0),
            warmup_start_step=0,
            warmup_step=distillation_cfg.get("sample_warmup_step", 0),
        )
        self.mixup_factor = distillation_cfg.get('mixup_end_factor', 0.0)
        self.mixup_step_scheduler = StepWarmUpScheduler(
            start_ratio=distillation_cfg.get('mixup_start_factor', 0.0),
            end_ratio=self.mixup_factor,
            warmup_start_step=0,
            warmup_step=distillation_cfg.get("mixup_warmup_step", 0),
        )
        self.SST = False
        # --------------------------- for SST --------------------------- #
        if SST_cfg:   
            self.SST = True
            self.SST_weight = SST_cfg.get('SST_weight',1)
            # 初始化SST网络
            self.SST_network = SST_network(
                cfg=SST_cfg,
                video_feature_dim=SST_cfg.get('video_feature_dim', 128),
                text_feature_dim=SST_cfg.get('text_feature_dim', 128),
                type="1",
            )


        if hasattr(self, "gloss_embedding"):
            delattr(self, "gloss_embedding")
        if hasattr(self, "gloss_tokenizer"):
            delattr(self, "gloss_tokenizer")

    def _init_mlm_head(self):
        with torch.no_grad():
            self.mlm_head.weight.data = self.model.lm_head.weight.clone().detach()

    def set_num_updates(self, num_updates):
        self.kl_factor = self.kl_step_scheduler.forward(num_updates)
        self.gkl_factor = self.gkl_step_scheduler.forward(num_updates)
        self.sample_factor = self.sample_step_scheduler.forward(num_updates)
        self.mixup_factor = self.mixup_step_scheduler.forward(num_updates)


    def prepare_gaussian_net_feature_inputs(self, sign_embeds, sign_mask, text=None, text_length=None):
        # concat the text_embedding and text_padding_mask.
        if self.cat_eos:
            assert text_length is not None
            suffix_tokens = torch.tensor(
                [self.text_tokenizer.lang_index],
                dtype=text.dtype,
                device=text.device,
            )
            max_length = torch.max(text_length) + len(suffix_tokens)
            text_with_suffix = torch.full(
                [text.size(0), max_length],
                self.text_tokenizer.pad_index,
                dtype=text.dtype,
                device=text.device
            )
            for i, line in enumerate(text):
                valid_line = text[i, :text_length[i]-1]
                line_with_suffix = torch.cat([valid_line, suffix_tokens])
                text_with_suffix[i, :len(line_with_suffix)] = line_with_suffix
            text_embeds = self.model.model.shared(text_with_suffix) * self.input_embed_scale
            text_mask = text_with_suffix.ne(self.text_tokenizer.pad_index)
        else:
            text_embeds = self.model.model.shared(text) * self.input_embed_scale
            text_mask = text.ne(self.text_tokenizer.pad_index)
        transformer_inputs = {
            'inputs_embeds': torch.cat([sign_embeds, text_embeds], dim=1),  # [B, T_sign + T_text, D]
            'attention_mask': torch.cat([sign_mask, text_mask], dim=1),  # attention_mask
        }
        return transformer_inputs
    
    def prepare_gold_instance_feature_inputs(self, rgb_feature, keypoint_feature, mask):
        # rgb_attention_mask = torch.zeros(
        #     [rgb_feature.shape[0], max_length],
        #     dtype=torch.long,
        #     device=rgb_feature.device
        # )
        # keypoint_attention_mask = torch.zeros(
        #     [keypoint_feature.shape[0], max_length],
        #     dtype=torch.long,
        #     device=keypoint_feature.device
        # )

        transformer_inputs = {
            'inputs_embeds': torch.cat([rgb_feature, keypoint_feature], dim=1),  # [B, 2T, D]
            'attention_mask': torch.cat([mask, mask], dim=1),  # attention_mask
        }

        return transformer_inputs

    def _compute_kl_loss(self, gkl_out, kl_out, unidirection=False):
        kl_1 = F.kl_div(gkl_out.log_softmax(-1), kl_out.softmax(-1), reduction="sum")
        if unidirection:
            kl_loss = kl_1
        else:
            kl_2 = F.kl_div(kl_out.log_softmax(-1), gkl_out.softmax(-1), reduction="sum")
            kl_loss = (kl_1 + kl_2) / 2
        return kl_loss

    def _compute_gaussian_kl_loss(self, kl_mean, kl_logvar, gkl_mean, gkl_logvar, func_type="gkl"):
        if func_type not in ["gkl", "cgkl"]:
            raise TypeError
        if func_type == "gkl":
            gkl_loss = gaussian_kl_loss(
                kl_mean=kl_mean, kl_logvar=kl_logvar,
                gkl_mean=gkl_mean, gkl_logvar=gkl_logvar,
            )
        else:
            gkl_loss = conditional_gaussian_kl_loss(
                kl_mean=kl_mean, kl_logvar=kl_logvar,
                gkl_mean=gkl_mean, gkl_logvar=gkl_logvar,
            )
        return gkl_loss

    # def _compute_mlm_loss(self, logits, labels):
    #     mlm_loss = F.kl_div(logits.log_softmax(-1), labels.softmax(-1), reduction="sum")
    #     return mlm_loss

    def forward_y(self, kwargs: dict):
        # encoder_output_dict = self.encoder(**encoder_kwargs, return_dict=True)
        output_dict = self.model(**kwargs, return_dict=True)
        gkl_y_hat = output_dict['logits'].detach().argmax(dim=-1)
        # kl_y_hat = self.y_generator.generate(
        #     sample_factor=self.sample_factor,
        #     ground_truth=kwargs['labels'],
        #     logits=output_dict['logits'],
        # )
        kl_y_hat = kwargs['labels']
        # TODO(rzhao): add noise to y_hat, in case that y_hat equal to y_truth in the later stage of training.
        #  1.select top-5 logits

        if not self.training:
            encoder_outputs = BaseModelOutput(last_hidden_state=output_dict['encoder_last_hidden_state'])
            generate_out = self.generate(
                **{**kwargs, **{"encoder_outputs": encoder_outputs}},
                num_beams=5, max_length=100, length_penalty=1,
            )
            gkl_y_hat = generate_out['sequences']
        return output_dict['logits'], gkl_y_hat, kl_y_hat

    def mask_y(self, y: torch.Tensor):
        masked_y = y.clone().detach()
        if not self.training:
            ind = torch.ones(masked_y.size(), device=masked_y.device).bool()
            return masked_y, ind
        else:
            ind = torch.bernoulli(torch.full(y.size(), self.sample_factor, device=y.device)).bool()
            masked_id = self.text_tokenizer.pruneids[self.text_tokenizer.tokenizer.mask_token_id]
            masked_y[ind] = masked_id
        return masked_y, ind
    
    def interpolate_features(self, input_feature, rgb_feature, keypoint_feature, step_add=1, alpha=0.1, type="interpolate"):
        if self.SST_current_step < self.SST_final_step and self.training:
            if type is None or type == 'interpolate':
                # 计算当前的标准差
                std_dev = self.SST_initial_std_dev - (self.SST_current_step / self.SST_final_step) * (self.SST_initial_std_dev - self.SST_final_std_dev)
                # 生成截断正态分布的alpha
                lower, upper = 0, 1
                alpha = truncnorm(
                    (lower - self.SST_mean_alpha) / std_dev, (upper - self.SST_mean_alpha) / std_dev, loc=self.SST_mean_alpha, scale=std_dev
                ).rvs()
                if step_add == 0:
                    combined_feature = alpha * rgb_feature + (1 - alpha) * input_feature
                else:
                    combined_feature = alpha * keypoint_feature + (1 - alpha) * input_feature
            else:
                # 计算当前的标准差
                std_dev = self.SST_initial_std_dev - (self.SST_current_step / self.SST_final_step) * (self.SST_initial_std_dev - self.SST_final_std_dev)
                # 生成截断正态分布的alpha
                lower, upper = 0, 1
                if step_add == 0:
                    single_feature = rgb_feature
                else:
                    single_feature = keypoint_feature
                if type == 'feature_mixup': #(1)
                    # 生成beta分布随机数
                    lam = truncnorm(
                        (lower - self.SST_mean_alpha) / std_dev, (upper - self.SST_mean_alpha) / std_dev, loc=self.SST_mean_alpha, scale=std_dev
                    ).rvs()
                    
                    # 在特征维度C上进行加权平均
                    combined_feature = lam * single_feature + (1 - lam) * input_feature
                elif type == 'choice0':
                    combined_feature = input_feature
                elif type == 'choice25':
                    combined_feature = 0.25 * single_feature + 0.75 * input_feature
                elif type == 'choice50':
                    combined_feature = 0.5 * single_feature + 0.5 * input_feature
                elif type == 'random_mixup':
                    lam = torch.rand(1).to(input_feature.device)
                    # Mix global and local features
                    combined_feature = lam * single_feature + (1 - lam) * input_feature
                elif type == 'temporal_mixup': #(T)
                    lam = truncnorm(
                        (lower - self.SST_mean_alpha) / std_dev, (upper - self.SST_mean_alpha) / std_dev, loc=self.SST_mean_alpha, scale=std_dev
                    ).rvs(size = (input_feature.size(1), 1))
                    lam = torch.from_numpy(lam).to(device=input_feature.device).float()
                    combined_feature = lam * single_feature + (1 - lam) * input_feature
                elif type == 'full_dimension_mixup': #lam (T,C)
                    lam = truncnorm(
                        (lower - self.SST_mean_alpha) / std_dev, (upper - self.SST_mean_alpha) / std_dev, loc=self.SST_mean_alpha, scale=std_dev
                    ).rvs(size = (input_feature.size(1), input_feature.size(2)))
                    lam = torch.from_numpy(lam).to(device=input_feature.device).float()
                    combined_feature = lam * single_feature + (1 - lam) * input_feature
                elif type == 'random_timepoint_feature_mixup':
                    t = torch.randint(0, input_feature.size(1), (1,)).item()
                    lam = truncnorm(
                        (lower - self.SST_mean_alpha) / std_dev, (upper - self.SST_mean_alpha) / std_dev, loc=self.SST_mean_alpha, scale=std_dev
                    ).rvs()
                    combined_feature = input_feature.clone()
                    combined_feature[:, t, :] = lam * single_feature[:, t, :] + (1 - lam) * input_feature[:, t, :]
                elif type == 'random_timepoint_temporal_mixup':
                    t = torch.randint(0, input_feature.size(1), (1,)).item()
                    lam = truncnorm(
                        (lower - self.SST_mean_alpha) / std_dev, (upper - self.SST_mean_alpha) / std_dev, loc=self.SST_mean_alpha, scale=std_dev
                    ).rvs(size = (input_feature.size(1), 1))
                    lam = torch.from_numpy(lam).to(device=input_feature.device).float()
                    combined_feature = input_feature.clone()
                    combined_feature[:, t, :] = lam[t] * single_feature[:, t, :] + (1 - lam[t]) * input_feature[:, t, :]
                elif type == 'random_timepoint_full_dimension_mixup':
                    t = torch.randint(0, input_feature.size(1), (1,)).item()
                    lam = truncnorm(
                        (lower - self.SST_mean_alpha) / std_dev, (upper - self.SST_mean_alpha) / std_dev, loc=self.SST_mean_alpha, scale=std_dev
                    ).rvs(size = (input_feature.size(1), input_feature.size(2)))
                    lam = torch.from_numpy(lam).to(device=input_feature.device).float()
                    combined_feature = input_feature.clone()
                    combined_feature[:, t, :] = lam[t] * single_feature[:, t, :] + (1 - lam[t]) * input_feature[:, t, :]
                elif type == 'hierarchical_mixup':
                    levels = 2
                    lam = truncnorm(
                        (lower - self.SST_mean_alpha) / std_dev, (upper - self.SST_mean_alpha) / std_dev, loc=self.SST_mean_alpha, scale=std_dev
                    ).rvs((levels,))
                    lam = torch.from_numpy(lam).to(device=input_feature.device).float()
                    combined_feature = input_feature.clone()
                    step = input_feature.size(1) // levels
                    for i in range(levels):
                        start = i * step
                        end = start + step
                        combined_feature[:, start:end, :] = lam[i] * single_feature[:, start:end, :] + (1 - lam[i]) * input_feature[:, start:end, :]
            self.SST_current_step = self.SST_current_step + step_add
            return combined_feature
        else:
            # final_step之后，直接返回input_feature
            return input_feature
        
    # def interpolate_features(self, input_feature, rgb_feature, keypoint_feature, step_add=1):
    #     if self.SST_current_step < self.SST_final_step and self.training:
    #         # 计算当前的标准差
    #         std_dev = self.SST_initial_std_dev - (self.SST_current_step / self.SST_final_step) * (self.SST_initial_std_dev - self.SST_final_std_dev)
    #         # 生成截断正态分布的alpha
    #         lower, upper = 0, 1
    #         alpha = truncnorm(  
    #             (lower - self.SST_mean_alpha) / std_dev, (upper - self.SST_mean_alpha) / std_dev, loc=self.SST_mean_alpha, scale=std_dev
    #         ).rvs()
    #         if step_add == 0:
    #             combined_feature = alpha * rgb_feature + (1 - alpha) * input_feature
    #         else:
    #             combined_feature = alpha * keypoint_feature + (1 - alpha) * input_feature
    #         self.SST_current_step = self.SST_current_step + step_add
    #         return combined_feature
    #     else:
    #         # final_step之后，直接返回input_feature
    #         return input_feature

    def forward(self, **kwargs):
        # quick fix
        input_feature, input_lengths = kwargs.pop('input_feature'), kwargs.pop('input_lengths')
        
        popped_keys = ['gloss_ids', 'gloss_lengths', 'input_ids']
        for key in popped_keys:
            kwargs.pop(key, None)
        text_length = kwargs.pop('text_length', None)
        kwargs2 = {key: kwargs.pop(key) for key in ['rgb_feature', 'keypoint_feature']}
        input_feature = self.interpolate_features(input_feature=input_feature, rgb_feature=kwargs2['rgb_feature'], keypoint_feature=kwargs2['keypoint_feature'], step_add=0, type=self.interpolate_type)
        input_feature2 = self.interpolate_features(input_feature=input_feature, rgb_feature=kwargs2['rgb_feature'], keypoint_feature=kwargs2['keypoint_feature'], step_add=1, type=self.interpolate_type)
        if not self.cat_eos:
            text_length = None

        encoder_kwargs = self.prepare_feature_inputs(input_feature, input_lengths)
        encoder_kwargs2 = self.prepare_feature_inputs(input_feature2, input_lengths)
        kwargs = {**kwargs, **encoder_kwargs}
        kwargs = {**kwargs, **encoder_kwargs2}

        gkl_encoder_output_dict = self.encoder(
            **encoder_kwargs, return_dict=True, output_hidden_states=not self.training
        )
        gkl_encoder_output_dict2 = self.encoder(
            **encoder_kwargs2, return_dict=True, output_hidden_states=not self.training 
        )      
        gkl_sign_encoder_out = gkl_encoder_output_dict["last_hidden_state"]
        gkl_sign_encoder_out2 = gkl_encoder_output_dict2["last_hidden_state"]
        gkl_encoder_outputs = BaseModelOutput(last_hidden_state=gkl_sign_encoder_out)
        gkl_encoder_outputs2 = BaseModelOutput(last_hidden_state=gkl_sign_encoder_out2)
        
        gkl_output_dict = self.model(
            **kwargs, return_dict=True,
            output_hidden_states=True,
            encoder_outputs=gkl_encoder_outputs,
        )
        # kl decoder output
        gkl_output_dict2 = self.model(
            **kwargs, return_dict=True,
            output_hidden_states=True,
            encoder_outputs=gkl_encoder_outputs2
        )

        gkl_batch_loss_sum = self.translation_loss_fun(
            log_probs=gkl_output_dict['logits'].log_softmax(-1), targets=kwargs['labels']
        )
        gkl_batch_loss_sum2 = self.translation_loss_fun(
            log_probs=gkl_output_dict2['logits'].log_softmax(-1), targets=kwargs['labels']
        )
        
        #使用R-drop进行kl散度的拉进
        if self.hidden_kl:
            kl_loss = self._compute_kl_loss(
                gkl_out=gkl_sign_encoder_out,
                kl_out=gkl_sign_encoder_out2,
                unidirection=self.unidirection_kl
            )
        else:
            kl_loss = self._compute_kl_loss(
                gkl_out=gkl_output_dict["logits"],
                kl_out=gkl_output_dict2["logits"],
                unidirection=self.unidirection_kl
            )

        gkl_loss = self._compute_kl_loss(   #gkl改成传统的，非CVAE的kl散度
            gkl_out=gkl_output_dict['encoder_last_hidden_state'],
            kl_out=gkl_output_dict2['encoder_last_hidden_state'],
            unidirection=self.unidirection_kl
        )
        # Tips: batch norm or tokens norm.
        if self.sentence_avg:
            sample_size = kwargs['labels'].size(0)
        else:
            sample_size = kwargs['labels'].ne(self.text_tokenizer.pad_index).sum()
        if self.SST:
            SST_outputs = self.SST_network(video_feature=gkl_output_dict['encoder_last_hidden_state'], text_feature=gkl_output_dict['decoder_hidden_states'][-1],video_feature2=gkl_output_dict2['encoder_last_hidden_state'], text_feature2=gkl_output_dict2['decoder_hidden_states'][-1])
            SST_outputs["SST_loss"] = SST_outputs["SST_loss"] / sample_size * self.SST_weight
        gkl_output_dict['gkl_translation_loss2'] = gkl_batch_loss_sum2 / sample_size
        gkl_output_dict['gkl_translation_loss'] = gkl_batch_loss_sum / sample_size * self.gkl_weight
        gkl_output_dict["kl_loss"] = kl_loss / sample_size * self.kl_factor
        gkl_output_dict["gkl_loss"] = gkl_loss / sample_size * self.gkl_factor

        gkl_output_dict['translation_loss'] = (
                gkl_output_dict['gkl_translation_loss2']
                + gkl_output_dict['gkl_translation_loss']
                + gkl_output_dict['gkl_loss']
                + gkl_output_dict['kl_loss']
        )
        (
            gkl_output_dict['gkl_factor'],
            gkl_output_dict['kl_factor'],
            gkl_output_dict['sample_factor'],
            gkl_output_dict['mixup_factor'],
        ) = self.gkl_factor, self.kl_factor,self.sample_factor, self.mixup_factor  # for tensorboard

        if self.SST:
            gkl_output_dict['translation_loss'] = gkl_output_dict['translation_loss'] + SST_outputs["SST_loss"]
            gkl_output_dict['SST_weight'] = self.SST_weight
        kwargs["encoder_outputs"] = gkl_encoder_outputs  # for later use (decoding), gkl only.
        gkl_output_dict['transformer_inputs'] = kwargs  # for later use (decoding), gkl only.
        # gkl_output_dict['first_decoded_sequences'] = self.text_tokenizer.batch_decode(kwargs['decoder_input_ids'])
        gkl_output_dict['kl_encoder_outputs'] = gkl_encoder_outputs2
        if not self.training:
            gkl_output_dict['gkl_encoder_hidden_states'] = gkl_encoder_output_dict['hidden_states']
            gkl_output_dict['kl_encoder_hidden_states'] = gkl_encoder_output_dict2['hidden_states']
            gkl_output_dict['kl_decoder_hidden_states'] = gkl_output_dict2.get("decoder_hidden_states", None)

        return gkl_output_dict

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
            output_hidden_states=hasattr(self.model, "knn_mode") and self.model.knn_mode == "inference",
            # output_scores=True,
        )
        output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'])
        return output_dict

class LN(nn.Module):
    def __init__(self, latent_dim, gamma=3.0):
        super().__init__()
        self.ln = nn.LayerNorm(latent_dim)
        self.ln.weight.requires_grad = False
        self.gamma = gamma
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.ln.weight.fill_(self.gamma)

    def forward(self, x):
        return self.ln(x)


class GateNet(nn.Module):
    def __init__(self, d_model, d_hidden, d_output, dropout=0.0):
        super().__init__()
        self.input_to_hidden = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.hidden_to_output = nn.Linear(d_hidden, d_output)
        self.output_activation = nn.Sigmoid()

    def forward(self, inputs):
        h = F.relu(self.input_to_hidden(inputs))
        h = self.dropout(h)
        h = self.hidden_to_output(h)
        return self.output_activation(h)


class StepWarmUpScheduler(object):
    def __init__(self, start_ratio, end_ratio, warmup_start_step, warmup_step):
        super().__init__()
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.warmup_start_step = warmup_start_step
        self.warmup_step = warmup_step + int(warmup_step == 0)
        self.step_ratio = (end_ratio - start_ratio) / self.warmup_step
        # self.anneal_end = warmup_start + warmup_step
        # self.print_ratio_every = args.print_ratio_every

    def forward(self, step_num):
        if step_num < self.warmup_start_step:
            return self.start_ratio
        elif step_num >= self.warmup_step:
            return self.end_ratio
        else:
            ratio = self.start_ratio + self.step_ratio * (step_num - self.warmup_start_step)
            # if (step_num + 1) % self.print_ratio_every == 0:
            #     print("=" * 15, "STEP: {} RATIO:{}".format(step_num + 1, ratio), "=" * 15)
            return ratio


class PseudoLabelGenerator(object):
    def __init__(self, end_ratio, k=4):
        super().__init__()
        self.end_ratio = end_ratio
        self.k = k

    def generate(self, sample_factor: float, ground_truth: torch.Tensor, logits: torch.Tensor):
        sample_num = int(sample_factor * ground_truth.size(1))
        if sample_num > 0:
            gkl_y_hat = logits.argmax(dim=-1)
            y_hat = ground_truth.detach()
            col_ind = torch.topk(logits.max(dim=-1).values, k=sample_num, largest=False, dim=-1).indices  # [B, L]
            row_ind = torch.arange(col_ind.size(0)).unsqueeze(1)
            assert row_ind.dim() == col_ind.dim()
            y_hat[row_ind, col_ind] = gkl_y_hat[row_ind, col_ind]
        else:
            y_hat = ground_truth
        return y_hat


def gaussian_kl_loss(kl_mean, kl_logvar, gkl_mean, gkl_logvar, norm_type="batchnorm"):
    kl_loss = -0.5 * torch.sum(
        1 + (kl_logvar - gkl_logvar)
        - torch.div(
            torch.pow(gkl_mean - kl_mean, 2) + kl_logvar.exp(),
            gkl_logvar.exp(),
        )
    )
    return kl_loss


def conditional_gaussian_kl_loss(kl_mean, kl_logvar, gkl_mean, gkl_logvar, norm_type="batchnorm"):
    kl_loss = -0.5 * torch.sum(
        1 + kl_logvar - gkl_logvar.exp() - gkl_mean.pow(2)
        - torch.div(
            torch.pow(gkl_mean - kl_mean, 2) + kl_logvar.exp(),
            gkl_logvar.exp(),
        )
    )
    return kl_loss
