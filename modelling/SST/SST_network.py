import torch
from torch import nn
import numpy as np
import math
import yaml
from easydict import EasyDict
from modelling.SST.sparsemax import Sparsemax
from modelling.SST.loss import LabelSmoothCELoss, ClipInfoCELoss
import torch.nn.functional as F

#---- attention models for SST
class Query_model(nn.Module):
    def __init__(self, ft_dim, sd_dim, temperature=1, att_func_type='sparsemax', pool_type='sum'):
        '''
        ft_dim: feature dim of image patch or text token
        sd_dim: dim of SST
        temperature: temperature for softmax or sparsemax
        att_func_type: attention normlization function type
        pool_type: pooling type for attention weights
        '''

        super().__init__()

        #activation 
        assert att_func_type in ['softmax', 'sigmoid', 'sparsemax']
        self.att_func_type = att_func_type

        assert pool_type in ['mean', 'max', 'sum']
        self.pool_type = pool_type

        if self.att_func_type == 'softmax':
            self.att_activation = nn.Softmax(dim=-1)
        elif self.att_func_type == 'sparsemax':
            self.att_activation = Sparsemax(dim=-1)
        else:
            self.att_activation = nn.Sigmoid()

        self.att_dim = sd_dim
        self.temperature = temperature
        
        #map patch/text tokens to codebook (query) spaces
        #---note that we donot use mapping for SST

        self.q_map = nn.Sequential(
            nn.LayerNorm(ft_dim),
            nn.Linear(ft_dim, sd_dim),
            nn.GELU(),
            nn.LayerNorm(sd_dim),
            nn.Linear(sd_dim, sd_dim)
        )



    def forward(self, ft, sd, mask=None, return_token_att=False):


        '''
        Args:
            ft: [batch, token_num, ft_dim]
            sd: [SST_num, sd_dim]
            mask: [batch, token_num]: mask for padded tokens.
            return_token_att: flag for returning attention weights before nomalization.
            used for visualizing SST.
        Returns:

        '''

        #map image/text token to query space
        q = self.q_map(ft) #bacth, token_num, dim

        k = sd #code_num, sd_dim
        k = k.unsqueeze(0) #[1, code_num, sd_dim]
        k = k.transpose(2, 1) #[1,sd_dim, sd_num]
        
        #-----calculate inner dot
        inner_dot = torch.matmul(q, k) #[bacth, token_num, code_num]

        if return_token_att: #cosine sim
            token_att = inner_dot

        inner_dot = inner_dot / math.sqrt(self.att_dim) #scale dot norm

        if mask is not None: # mask paded tokens
            
            assert mask.shape == q.shape[:2]
            mask = (mask == 0) * 1 #0 --> 1, inf --> 0

            inner_dot = inner_dot * mask.unsqueeze(-1) #sigmod(-inf) = 0, softmax(-inf) = 0

            if return_token_att: #if has pad, return maksed
                token_att = inner_dot


        # temptural norm
        inner_dot = inner_dot / self.temperature #[bacth, token_num, code_num]

        #pooling
        if self.pool_type == 'sum':
            inner_dot = inner_dot.sum(1) #mean poolings
        elif self.pool_type == 'mean':
            inner_dot = inner_dot.mean(1)
        else:
            inner_dot = inner_dot.max(1)[0]

        #----get attention weights
        att_weight = self.att_activation(inner_dot) #normaliztion

        #----calculate weighted sum of v
        #v = self.ln_v(ft) #map to v_space
        
        att_ft = att_weight @ sd  #[bacth, dictory_size] * [dictory_size, dim]  ---> [bacth, sd_num, dim]

        if self.att_func_type == 'sigmoid':
            att_ft = att_ft / att_weight.sum(dim=-1, keepdim=True)
        
        if return_token_att:
            return token_att, att_ft, sd
        return att_weight, att_ft, sd


class SST_network_ablation(nn.Module):
    def __init__(self, cfg, video_feature_dim, text_feature_dim, type="0", embeddings=None):
        super().__init__()
        self.sd_num=cfg.get('sd_num', 128)
        self.sd_dim=cfg.get('sd_dim', 128)
        self.sd_temperature=cfg.get('sd_temperature', 0.07)
        self.att_func_type=cfg.get('att_func_type', 'sparsemax')
        self.pool_type=cfg.get('pool_type', 'mean')
        self.AlignNet_Choice=cfg.get("AlignNet_Choice", -1)
        self.space_type = cfg.get("space_type", "RandInitTrain") #RandInitTrain/IndepFineTune/SharedFineTune
        self.type = type
        self.fc = nn.Linear(video_feature_dim, self.sd_dim)
        #learnable temperature for infoNCE loss
        self.logit_scale = nn.Parameter(torch.ones([1]))
        self.logit_scale_sd = nn.Parameter(torch.ones([1]))
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        nn.init.constant_(self.logit_scale_sd, np.log(1 / 0.07))
        # nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        self.criterion = ClipInfoCELoss()

        self.criterion = ClipInfoCELoss()

        
    def forward(self, video_feature, text_feature, video_pad_mask=None, text_pad_mask=None, video_feature2=None, text_feature2=None, pool_type="max"):
        if pool_type=="mean":
            # 对齐序列
            sd_video_ft  = torch.mean(video_feature, dim=1)
            sd_text_ft = torch.mean(text_feature, dim=1)
        else:
            sd_video_ft  = torch.max(video_feature, dim=1).values
            sd_text_ft = torch.max(text_feature, dim=1) .values
        # 特征转换
        sd_video_ft = self.fc(sd_video_ft)
        sd_text_ft = self.fc(sd_text_ft)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)
        logits_per_video_sd = sd_video_ft @ sd_text_ft.t() * logit_scale    #(B,B) = (B,C) * (C,B)
        logits_per_text_sd = sd_text_ft @ sd_video_ft.t() * logit_scale
        assert logits_per_video_sd.shape == logits_per_text_sd.shape
        SST_loss, _ = self.criterion(logits_per_video_sd, logits_per_text_sd, type="0")


        return {
            "logits_per_video_sd": logits_per_video_sd,
            "logits_per_text_sd": logits_per_text_sd,
            "SST_loss": SST_loss
        }




class SST_network(nn.Module):
    def __init__(self, cfg, video_feature_dim, text_feature_dim, type="0", embeddings=None):
        super().__init__()
        self.sd_num=cfg.get('sd_num', 128)
        self.sd_dim=cfg.get('sd_dim', 128)
        self.sd_temperature=cfg.get('sd_temperature', 0.07)
        self.att_func_type=cfg.get('att_func_type', 'sparsemax')
        self.pool_type=cfg.get('pool_type', 'mean')
        self.AlignNet_Choice=cfg.get("AlignNet_Choice", -1)
        self.space_type = cfg.get("space_type", "RandInitTrain") #RandInitTrain/IndepFineTune/SharedFineTune
        self.type = type
        if self.AlignNet_Choice != -1:
            self.AlignNet = AlignNet(dim=video_feature_dim, choice=self.AlignNet_Choice)
        #learnable SST
        
        
        if self.space_type in "IndepFineTune":
            initial_embedding = embeddings.detach().clone()
            self.space_dict = nn.Parameter(initial_embedding)
            self.sd_num, self.sd_dim = embeddings.shape
        elif self.space_type in "SharedFineTune":
            self.space_dict  = embeddings
            self.sd_num, self.sd_dim = embeddings.shape
        else:
            self.space_dict = nn.Parameter(torch.randn(self.sd_num, self.sd_dim))
        #query mapping
        self.video_query_model = Query_model(ft_dim=video_feature_dim, sd_dim=self.sd_dim, temperature=self.sd_temperature, att_func_type=self.att_func_type, pool_type=self.pool_type)
        self.text_query_model = Query_model(ft_dim=text_feature_dim, sd_dim=self.sd_dim, temperature=self.sd_temperature, att_func_type=self.att_func_type, pool_type=self.pool_type)
        if type == "1":
            self.video_query_model2 = Query_model(ft_dim=video_feature_dim, sd_dim=self.sd_dim, temperature=self.sd_temperature, att_func_type=self.att_func_type, pool_type=self.pool_type)
            self.text_query_model2 = Query_model(ft_dim=text_feature_dim, sd_dim=self.sd_dim, temperature=self.sd_temperature, att_func_type=self.att_func_type, pool_type=self.pool_type)
        #learnable temperature for infoNCE loss
        self.logit_scale = nn.Parameter(torch.ones([1]))
        self.logit_scale_sd = nn.Parameter(torch.ones([1]))
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        nn.init.constant_(self.logit_scale_sd, np.log(1 / 0.07))
        # nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        self.criterion = ClipInfoCELoss()

        
    def forward(self, video_feature, text_feature, video_pad_mask=None, text_pad_mask=None, video_feature2=None, text_feature2=None):
        #calculate SST-based features
        if self.AlignNet_Choice != -1:
            video_feature = self.AlignNet(video_feature)
            text_feature = self.AlignNet(text_feature)
        sd_video_att_weight, sd_video_ft, video_k = self.video_query_model(video_feature, self.space_dict,mask=video_pad_mask)
        sd_text_att_weight , sd_text_ft, text_k = self.text_query_model(text_feature, self.space_dict, mask=text_pad_mask)
        if self.type == "1":
            sd_video_att_weight2, sd_video_ft2, video_k2 = self.video_query_model(video_feature2, self.space_dict,mask=video_pad_mask)
            sd_text_att_weight2 , sd_text_ft2, text_k2 = self.text_query_model(text_feature2, self.space_dict, mask=text_pad_mask)
        # print("video_feature",video_feature.shape) #(B,T,C)
        # print("sd_video_ft",sd_video_ft.shape) #(B,C)
        #l2 normalization
        sd_video_ft = sd_video_ft / (sd_video_ft.norm(dim=-1, keepdim=True) + 1e-10)
        sd_text_ft = sd_text_ft / (sd_text_ft.norm(dim=-1, keepdim=True) + 1e-10)
        if self.type == "1":
            sd_video_ft2 = sd_video_ft2 / (sd_video_ft2.norm(dim=-1, keepdim=True) + 1e-10)
            sd_text_ft2 = sd_text_ft2 / (sd_text_ft2.norm(dim=-1, keepdim=True) + 1e-10)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)
        if self.type == "1":
            sd_video_ft = torch.cat((sd_video_ft, sd_video_ft2), dim=0)
            sd_text_ft = torch.cat((sd_text_ft, sd_text_ft2), dim=0)
        logits_per_video_sd = sd_video_ft @ sd_text_ft.t() * logit_scale    #(B,B) = (B,C) * (C,B)
        logits_per_text_sd = sd_text_ft @ sd_video_ft.t() * logit_scale
        assert logits_per_video_sd.shape == logits_per_text_sd.shape
        SST_loss, _ = self.criterion(logits_per_video_sd, logits_per_text_sd, type=self.type)


        return {
            "logits_per_video_sd": logits_per_video_sd,
            "logits_per_text_sd": logits_per_text_sd,
            "video_k": video_k,
            "text_k": text_k,
            "SST_loss": SST_loss
        }



# SST_cfg = dict()
# SST_cfg['sd_num']=50
# SST_cfg['sd_dim']=1024              #SST Dimension Example
# SST_cfg['sd_temperature']=0.07     # SST temperature example
# SST_cfg['att_func_type']='sparsemax' # Example of Attention Function Types
# SST_cfg['pool_type']='mean'        # Example of pooling type

# SST_network = SST_network(
#     cfg=SST_cfg,  
#     video_feature_dim=1024, 
#     text_feature_dim=1024, 
# )
# batch_size = 8 
# video_feature_dim = 1024 
# text_feature_dim = 1024   

# Generate random video and text feature matrices
# video_feature = torch.randn(batch_size, 51,video_feature_dim)
# text_feature = torch.randn(batch_size, 37,text_feature_dim)

# # # Simulate masks, assuming for simplicity that all features are valid (without padding)
# # video_pad_mask = torch.ones(batch_size, 5, dtype=torch.bool)
# # text_pad_mask = torch.ones(batch_size, 5, dtype=torch.bool)

# # Using SST_network to Calculate Feature Comparison Loss
# outputs = SST_network(video_feature, text_feature)
# print(outputs['logits_per_video_sd'].shape,outputs['logits_per_text_sd'].shape)