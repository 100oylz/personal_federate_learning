import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ConvBlock import ConvBlock
from base_model import base_model





class client_model(nn.Module):
    def __init__(self, in_channel, num_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fd_model = base_model(in_channel)
        self.personal_model = base_model(in_channel)
        self.logit = nn.Linear(64*2, num_features)

    def forward(self, x):
        x1 = self.fd_model(x)
        x2 = self.personal_model(x)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)
        x = self.logit(x)
        return x


def client_train(model, train_loader, optimizer, criterion, device,is_train=True):
    model.train()
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        if(is_train):
            optimizer.step()
            optimizer.zero_grad()
        loss = loss.item()
        predict=torch.max(output,dim=1)[1]
        acc=torch.eq(target,predict).sum().item()/len(target)
    return loss,acc
        
def client_load_params(model:client_model,params):
    model.fd_model.conv1.conv2d.weight.data=params['fd_model.conv1.conv2d.weight']
    model.fd_model.conv1.conv2d.bias.data=params['fd_model.conv1.conv2d.bias']
    model.fd_model.conv1.bn.weight.data=params['fd_model.conv1.bn.weight']
    model.fd_model.conv1.bn.bias.data=params['fd_model.conv1.bn.bias']
    model.fd_model.conv1.bn.running_mean.data=params['fd_model.conv1.bn.running_mean']
    model.fd_model.conv1.bn.running_var.data=params['fd_model.conv1.bn.running_var']
    model.fd_model.conv2.conv2d.weight.data=params['fd_model.conv2.conv2d.weight']
    model.fd_model.conv2.conv2d.bias.data=params['fd_model.conv2.conv2d.bias']
    model.fd_model.conv2.bn.weight.data=params['fd_model.conv2.bn.weight']
    model.fd_model.conv2.bn.bias.data=params['fd_model.conv2.bn.bias']
    model.fd_model.conv2.bn.running_mean.data=params['fd_model.conv2.bn.running_mean']
    model.fd_model.conv2.bn.running_var.data=params['fd_model.conv2.bn.running_var']
    model.fd_model.conv3.conv2d.weight.data=params['fd_model.conv3.conv2d.weight']
    model.fd_model.conv3.conv2d.bias.data=params['fd_model.conv3.conv2d.bias']
    model.fd_model.conv3.bn.weight.data=params['fd_model.conv3.bn.weight']
    model.fd_model.conv3.bn.bias.data=params['fd_model.conv3.bn.bias']
    model.fd_model.conv3.bn.running_mean.data=params['fd_model.conv3.bn.running_mean']
    model.fd_model.conv3.bn.running_var.data=params['fd_model.conv3.bn.running_var']
    model.fd_model.conv4.conv2d.weight.data=params['fd_model.conv4.conv2d.weight']
    model.fd_model.conv4.conv2d.bias.data=params['fd_model.conv4.conv2d.bias']
    model.fd_model.conv4.bn.weight.data=params['fd_model.conv4.bn.weight']
    model.fd_model.conv4.bn.bias.data=params['fd_model.conv4.bn.bias']
    model.fd_model.conv4.bn.running_mean.data=params['fd_model.conv4.bn.running_mean']
    model.fd_model.conv4.bn.running_var.data=params['fd_model.conv4.bn.running_var']


        
def LocalUpdate(global_model_params,model:client_model,client_train_loader,optimizer,criterion,device,is_train=True):
    client_load_params(model,global_model_params)
    loss,acc=client_train(model,client_train_loader,optimizer,criterion,device,is_train)
    return loss,acc


if __name__=='__main__':
    client=client_model(1,10)
    state_dict=client.state_dict()
    print(state_dict.keys())
    
