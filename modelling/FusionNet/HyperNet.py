import torch
import torch.nn as nn
from  modelling.FusionNet.Variants.DSG import DSG
from  modelling.FusionNet.Variants.DShG import DShG
from  modelling.FusionNet.Variants.PIG import PIG
from  modelling.FusionNet.Variants.CIG import CIG
class FusionNet(nn.Module):
    def __init__(self, choice, global_dim=None, local_dim=None, dropout=None, AlignNet_choice=-1):
        super(FusionNet, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.choice = choice
        self.dropout_rate=dropout
        self.AlignNet_choice = AlignNet_choice
        # Choose different Hyper-Net models based on your choice
        if choice == "DSG":
            self.block = DSG(dim=self.global_dim, dropout_rate=self.dropout_rate).cuda()
        if choice == "DShG":
            self.block = DShG(dim=self.global_dim).cuda()
        if choice == "PIG":
            self.block = PIG(dim=self.global_dim).cuda()
        if choice == "CIG":
            self.block = CIG(dim=self.global_dim).cuda()
        if self.choice is None:
            self.out = nn.Sequential(
                nn.Linear(global_dim + local_dim, local_dim), 
                nn.ReLU(),
                nn.Linear(local_dim, local_dim) 
            )
        
    def forward(self,global_input, local_input):
        output = self.block(global_input, local_input)  
        if self.choice is None:  
            output = self.out(output) 
        return output
