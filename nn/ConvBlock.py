import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel,out_channel,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv2d=nn.Conv2d(in_channel,out_channel,3,1,1)
        self.relu=nn.ReLU()
        self.maxpool2d=nn.MaxPool2d(2,2)
        self.bn=nn.BatchNorm2d(num_features=out_channel)
        
    def forward(self,x):
        x=self.conv2d(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.maxpool2d(x)
        return x
        
        

def ConvBlockFunction(x,w,b,w_bn,b_bn):
    x=F.conv2d(x,w,b,1,1)
    x=F.batch_norm(x,w_bn,b_bn,running_mean=None,running_var=None,training=True)
    x=F.relu(x)
    x=F.maxpool2d(x,2,2)
    return x