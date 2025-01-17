from .ConvBlock import ConvBlock,ConvBlockFunction
import torch.nn as nn
import torch.nn.functional as F
import torch
from .base_model import base_model
import collections
import numpy as np


class server_model(nn.Module):
    def __init__(self, in_channel,num_features):
        super().__init__()
        self.fd_model=base_model(in_channel)
        self.logit=nn.Linear(64*2,num_features)
        self.best_loss=float('inf')
        
    def forward(self,x):
        x=self.fd_model(x)
        x=torch.concat((x,x),dim=1)
        x=x.view(x.size(0),-1)
        x=self.logit(x)
        return x

    def functional_forward(self,x,params):
        x=self.fd_model.functional_forward(x,params)
        x=x.view(x.size(0),-1)
        x=torch.concat((x,x),dim=1)
        x=F.linear(x,params['logit.weight'],params['logit.bias'])
        return x
    
def params_merge(paramslist:list,name):
    length=len(paramslist)
    data=torch.empty_like(paramslist[0][name].data)
    for i in range(length):
        data.add_(paramslist[i][name].data)
    data.div_(length)
    return data
    
def server_load_params(model:server_model,params_list:list):
    model.fd_model.conv1.conv2d.weight.data=params_merge(params_list,'fd_model.conv1.conv2d.weight')
    model.fd_model.conv1.conv2d.bias.data=params_merge(params_list,'fd_model.conv1.conv2d.bias')
    model.fd_model.conv1.bn.weight.data=params_merge(params_list,'fd_model.conv1.bn.weight')
    model.fd_model.conv1.bn.bias.data=params_merge(params_list,'fd_model.conv1.bn.bias')
    model.fd_model.conv2.conv2d.weight.data=params_merge(params_list,'fd_model.conv2.conv2d.weight')
    model.fd_model.conv2.conv2d.bias.data=params_merge(params_list,'fd_model.conv2.conv2d.bias')
    model.fd_model.conv2.bn.weight.data=params_merge(params_list,'fd_model.conv2.bn.weight')
    model.fd_model.conv2.bn.bias.data=params_merge(params_list,'fd_model.conv2.bn.bias')
    model.fd_model.conv3.conv2d.weight.data=params_merge(params_list,'fd_model.conv3.conv2d.weight')
    model.fd_model.conv3.conv2d.bias.data=params_merge(params_list,'fd_model.conv3.conv2d.bias')
    model.fd_model.conv3.bn.weight.data=params_merge(params_list,'fd_model.conv3.bn.weight')
    model.fd_model.conv3.bn.bias.data=params_merge(params_list,'fd_model.conv3.bn.bias')
    model.fd_model.conv4.conv2d.weight.data=params_merge(params_list,'fd_model.conv4.conv2d.weight')
    model.fd_model.conv4.conv2d.bias.data=params_merge(params_list,'fd_model.conv4.conv2d.bias')
    model.fd_model.conv4.bn.weight.data=params_merge(params_list,'fd_model.conv4.bn.weight')
    model.fd_model.conv4.bn.bias.data=params_merge(params_list,'fd_model.conv4.bn.bias')
    model.fd_model.conv5.conv2d.weight.data=params_merge(params_list,'fd_model.conv5.conv2d.weight')
    model.fd_model.conv5.conv2d.bias.data=params_merge(params_list,'fd_model.conv5.conv2d.bias')
    model.fd_model.conv5.bn.weight.data=params_merge(params_list,'fd_model.conv5.bn.weight')
    model.fd_model.conv5.bn.bias.data=params_merge(params_list,'fd_model.conv5.bn.bias')
    model.logit.weight.data=params_merge(params_list,'logit.weight')
    
def maml_train(model:server_model, data_loader, inner_step, inner_lr, optimizer,is_train=True):
    """
    Train the model using MAML method.
    Args:
        model: Any model
        data_loader: MAMLDataLoader
        inner_step: support data training step
        inner_lr: inner learning rate
        optimizer: optimizer
        is_train: whether train

    Returns: meta loss, meta accuracy

    """
    meta_loss = []
    meta_acc = []
    for support_image, support_label, query_image, query_label in data_loader:
        support_image=torch.stack(support_image, dim=0)
        support_label=torch.stack(support_label, dim=0)
        query_image=torch.stack(query_image, dim=0)
        query_label=torch.stack(query_label, dim=0)
        fast_weights = collections.OrderedDict(model.named_parameters())
        for _ in range(inner_step):
            # Update weight
            support_logit = model.functional_forward(support_image, fast_weights)
            support_label=support_label.view(-1)
            support_loss = nn.CrossEntropyLoss()(support_logit, support_label)
            grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict((name, param - inner_lr * grad)
                                                   for ((name, param), grad) in zip(fast_weights.items(), grads))

        # Use trained weight to get query loss
        query_logit = model.functional_forward(query_image, fast_weights)
        query_prediction = torch.max(query_logit, dim=1)[1]
        query_label=query_label.view(-1)
        query_loss = nn.CrossEntropyLoss()(query_logit, query_label)
        query_acc = torch.eq(query_label, query_prediction).sum() / len(query_label)

        meta_loss.append(query_loss)
        meta_acc.append(query_acc.data.numpy())

    # Zero the gradient
    optimizer.zero_grad()
    meta_loss = torch.stack(meta_loss).mean()
    meta_acc = np.mean(meta_acc)

    if is_train:
        meta_loss.backward()
        optimizer.step()

    return meta_loss, meta_acc

    
def GlobalUpdate(model:server_model,paramslist:list,support_images,support_labels,query_images,query_labels,inner_step,args,optimizer,is_train=True):
    server_load_params(model,paramslist)
    meta_loss=0
    meta_acc=0
    # meta_loss,meta_acc=maml_train(model,support_images,support_labels,query_images,query_labels,inner_step,args,optimizer,is_train)
    return meta_loss,meta_acc


if __name__=='__main__':
    server=server_model(1,10)
    state_dict=server.state_dict()
    print(state_dict.keys())