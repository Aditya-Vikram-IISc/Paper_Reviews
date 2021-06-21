import torch
import torch.nn as nn


#Refer Squeeze-and-Excitation Networks (arXiv: 1709.01507)

class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_factor:int = 16):
        super(SEBlock, self).__init__()

        #Squueze Module
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        #Excitation Module
        self.excitation = nn.Sequential(
                                nn.Linear(in_features= input_channels, out_features= input_channels//reduction_factor, bias= False),
                                nn.ReLU(),
                                nn.Linear(in_features= input_channels//reduction_factor, out_features= input_channels, bias= False),
                                nn.Sigmoid()
                                        )

    def forward(self, x:torch.Tensor):
        b,c,h,w = x.size()

        y = self.squeeze(x).view(b,c)
        y = self.excitation(y).view(b,c,1,1)

        #Perform broadcasting in the spatial dimensions
        return x * y.expand_as(x)