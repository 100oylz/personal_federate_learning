
import torch.nn as nn
import torch.nn.functional as F

from .ConvBlock import ConvBlock,ConvBlockFunction

class base_model(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = ConvBlock(in_channel, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 64)
        self.conv5=ConvBlock(64,64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x=self.conv5(x)
        return x

    def functional_forward(self,x,params):
        x = ConvBlockFunction(x, params[f'fd_model.conv1.conv2d.weight'], params[f'fd_model.conv1.conv2d.bias'],
                              params.get(f'fd_model.conv1.bn.weight'), params.get(f'fd_model.conv1.bn.bias'))
        x = ConvBlockFunction(x, params[f'fd_model.conv2.conv2d.weight'], params[f'fd_model.conv2.conv2d.bias'],
                              params.get(f'fd_model.conv2.bn.weight'), params.get(f'fd_model.conv2.bn.bias'))
        x = ConvBlockFunction(x, params[f'fd_model.conv3.conv2d.weight'], params[f'fd_model.conv3.conv2d.bias'],
                              params.get(f'fd_model.conv3.bn.weight'), params.get(f'fd_model.conv3.bn.bias'))
        x = ConvBlockFunction(x, params[f'fd_model.conv4.conv2d.weight'], params[f'fd_model.conv4.conv2d.bias'],
                              params.get(f'fd_model.conv4.bn.weight'), params.get(f'fd_model.conv4.bn.bias'))
        x=ConvBlockFunction(x,params[f'fd_model.conv5.conv2d.weight'],params[f'fd_model.conv5.conv2d.bias'],
                              params.get(f'fd_model.conv5.bn.weight'),params.get(f'fd_model.conv5.bn.bias'))
        return x

    def lock(self):
        for param in self.parameters():
            param.requires_grad = False
    def unlock(self):
        for param in self.parameters():
            param.requires_grad = True
    
    