import collections

import torch
import torch.nn as nn

from .ConvBlock import ConvBlock
from .base_model import base_model





class client_model(nn.Module):
    def __init__(self, in_channel, num_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fd_model = base_model(in_channel)
        self.personal_model = base_model(in_channel)
        self.logit = nn.Linear(64*2, num_features)
        self.best_loss=float('inf')

    def forward(self, x):

        x1 = self.fd_model(x)
        x2 = self.personal_model(x)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)
        x = self.logit(x)
        return x

    def global_forward(self,x):
        x = self.fd_model(x)
        x = torch.cat((x, x), dim=1)
        x = x.view(x.size(0), -1)
        x = self.logit(x)
        return x


def client_train(model:client_model, train_loader, optimizer, criterion, device,client_id,file_base_name,code_value,mean,std,is_train=True):
    model.train()
    model.personal_model.lock()
    model.fd_model.unlock()
    batch_loss=0
    batch_acc=0
    torch.cuda.empty_cache()
    model=model.to(device)
    criterion=criterion.to(device)
    for param in model.fd_model.parameters():
        param.requires_grad=True
    for param in model.logit.parameters():
        param.requires_grad=True
    grads_list={}
    grads_list['fd_model']=[]
    grads_list['logit']=[]
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model.global_forward(data)
        loss = criterion(output, target)
        loss.backward()
        for name,param in model.fd_model.named_parameters():
            grads_list['fd_model'].append(add_gaussian_noise(param.grad.cpu().clone(),mean,std))
        for name,param in model.logit.named_parameters():
            grads_list['logit'].append(add_gaussian_noise(param.grad.cpu().clone(),mean,std))
        if(is_train):
            optimizer.zero_grad()
            optimizer.step()
        loss = loss.item()
        predict=torch.max(output,dim=1)[1]
        acc=torch.eq(target,predict).sum().item()/len(target)
        batch_loss+=loss
        batch_acc+=acc
    client_loss=batch_loss/len(train_loader)
    client_acc=batch_acc/len(train_loader)
    if(client_loss<model.best_loss) and (is_train):
        model.best_loss=client_loss
        torch.save(model.state_dict(),file_base_name.format(code_value,f'client{client_id}'))
    model=model.to('cpu')
    criterion=criterion.to('cpu')
    return grads_list,client_loss,client_acc

def personal_fit(model:client_model, train_loader, optimizer, criterion, device,client_id,file_base_name,code_value,is_train=True):
    model.train()
    model.personal_model.unlock()
    model.fd_model.lock()
    batch_loss=0
    batch_acc=0
    torch.cuda.empty_cache()
    model.to(device)
    criterion.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward()
        if(is_train):
            optimizer.zero_grad()
            optimizer.step()
        loss = loss.item()
        predict=torch.max(output,dim=1)[1]
        acc=torch.eq(target,predict).sum().item()/len(target)
        batch_loss+=loss
        batch_acc+=acc
    client_loss=batch_loss/len(train_loader)
    client_acc=batch_acc/len(train_loader)
    if(client_loss<model.best_loss) and (is_train):
        model.best_loss=client_loss
        torch.save(model.state_dict(),file_base_name.format(code_value,f'client{client_id}'))
    model.to('cpu')
    criterion.to('cpu')
    return client_loss,client_acc
        
def client_load_params(model:client_model,params):
    params=collections.OrderedDict(params)
    model.fd_model.conv1.conv2d.weight.data=params['fd_model.conv1.conv2d.weight']
    model.fd_model.conv1.conv2d.bias.data=params['fd_model.conv1.conv2d.bias']
    model.fd_model.conv1.bn.weight.data=params['fd_model.conv1.bn.weight']
    model.fd_model.conv1.bn.bias.data=params['fd_model.conv1.bn.bias']
    model.fd_model.conv2.conv2d.weight.data=params['fd_model.conv2.conv2d.weight']
    model.fd_model.conv2.conv2d.bias.data=params['fd_model.conv2.conv2d.bias']
    model.fd_model.conv2.bn.weight.data=params['fd_model.conv2.bn.weight']
    model.fd_model.conv2.bn.bias.data=params['fd_model.conv2.bn.bias']
    model.fd_model.conv3.conv2d.weight.data=params['fd_model.conv3.conv2d.weight']
    model.fd_model.conv3.conv2d.bias.data=params['fd_model.conv3.conv2d.bias']
    model.fd_model.conv3.bn.weight.data=params['fd_model.conv3.bn.weight']
    model.fd_model.conv3.bn.bias.data=params['fd_model.conv3.bn.bias']
    model.fd_model.conv4.conv2d.weight.data=params['fd_model.conv4.conv2d.weight']
    model.fd_model.conv4.conv2d.bias.data=params['fd_model.conv4.conv2d.bias']
    model.fd_model.conv4.bn.weight.data=params['fd_model.conv4.bn.weight']
    model.fd_model.conv4.bn.bias.data=params['fd_model.conv4.bn.bias']
    model.fd_model.conv5.conv2d.weight.data=params['fd_model.conv5.conv2d.weight']
    model.fd_model.conv5.conv2d.bias.data=params['fd_model.conv5.conv2d.bias']
    model.fd_model.conv5.bn.weight.data=params['fd_model.conv5.bn.weight']
    model.fd_model.conv5.bn.bias.data=params['fd_model.conv5.bn.bias']
    model.logit.weight.data=params['logit.weight']
    model.logit.bias.data=params['logit.bias']



def add_gaussian_noise(grads, mean, std):

    noise = torch.normal(mean, std, size=grads.size())
    grads.add_(noise)
    return grads

def client_model_test(model:client_model,data_loader,criterion,device,client_id,file_base_name,code_value):
    file=file_base_name.format(code_value,f'client{client_id}')
    state_dict=torch.load(file)
    model.load_state_dict(state_dict)
    model=model.to(device)
    criterion=criterion.to(device)
    client_loss=0
    client_acc=0
    for data,label in data_loader:
        data=data.to(device)
        label=label.to(device)
        output=model(data)
        loss=criterion(output,label)
        predict=torch.max(output,dim=1)[1]
        acc=torch.eq(predict,label).sum().item()/len(label)
        client_loss+=loss.item()
        client_acc+=acc
    client_loss=client_loss/len(data_loader)
    client_acc=client_acc/len(data_loader)
    return client_loss,client_acc




if __name__=='__main__':
    client=client_model(1,10)
    state_dict=client.state_dict()
    print(state_dict.keys())
    
