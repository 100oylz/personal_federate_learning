import collections
import random

import torch.utils.data

from .client import Client

from .baseServer import BaseServer
from source.nn.server_model import server_model
from source.utils import partation_main
from source.utils.dataloader import MAMLDataset
from .nn.server_model import server_load_params,maml_train
import collections
from .nn.client_model import client_load_params
class Server(BaseServer):
    def __init__(self, in_channel,args):
        super().__init__()
        self.maml_dataloader = None
        self.maml_dataset = None
        self.model=None
        self.num_clients=args.num_clients
        self.inner_step=args.inner_step
        self.inner_lr=args.inner_lr
        self.data_dict,self.num_features,self.maml_data=partation_main(args)
        self.clients_dict:dict[int,Client] = {i: Client(self.rsa_public_key,in_channel,self.num_features,args)
                        for i in range(self.num_clients)}
        self.set_model(in_channel,self.num_features)
        self.get_maml_dataset(args)
        self.get_maml_data_loader(1)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=args.outer_lr)
        self.args=args
        for key,value in self.clients_dict.items():
            value.get_data(self.data_dict[key])


    def choose_client(self,k):
        client_list=[i for i in range(self.num_clients)]
        client_choose_list=random.sample(client_list,k)
        return client_choose_list

    def set_model(self,in_channel,num_features):
        self.model=server_model(in_channel,num_features)

    def set_client_data(self,data_dict):
        self.data_dict=data_dict
        for i in range(self.num_clients):
            self.clients_dict[i].set_data_loader(self.data_dict[i])
    def get_maml_dataset(self,args):
        self.maml_dataset=MAMLDataset(self.maml_data,args.n_way,args.k_shot,args.k_query)

    def get_maml_data_loader(self,batch_size):
        self.maml_dataloader=torch.utils.data.DataLoader(self.maml_dataset,batch_size=batch_size)

    def update(self,lr,file_path_base_name,code_value,mean,std):
        choose_num=random.choice(range(1,self.num_clients+1))
        client_choose_list=self.choose_client(choose_num)
        client_params_list=[]
        client_loss_dict={}
        client_acc_dict={}
        for item in self.clients_dict.values():
            client_load_params(item.model,self.model.named_parameters())

        for i in client_choose_list:
            grads_list,client_loss,client_acc=self.clients_dict[i].model_train(i,file_path_base_name,code_value,mean,std)

            client_params_list.append(grads_list)
            client_loss_dict[i]=client_loss
            client_acc_dict[i]=client_acc

        server_load_params(self.model,client_params_list,lr,self.args.device)

        meta_loss,meta_acc=maml_train(self.model,self.maml_dataloader,self.inner_step,self.inner_lr,self.optimizer,self.args.device,is_train=True)
        if(meta_loss<self.model.best_loss):
            self.model.best_loss=meta_loss
            torch.save(self.model.state_dict(),file_path_base_name.format(code_value,'server_model'))
        return meta_loss,meta_acc,client_loss_dict,client_acc_dict

    def client_personal_fit(self,path_base_name,code_value):
        client_loss_dict={}
        client_acc_dict={}

        for i in range(self.num_clients):
            client_params,client_loss,client_acc=self.clients_dict[i].personal_fit(i,path_base_name,code_value)
            client_loss_dict[i]=client_loss
            client_acc_dict[i]=client_acc
        return client_loss_dict,client_acc_dict

    def init_personal_model(self):
        for i in range(self.num_clients):
            for value1,value2 in zip(self.clients_dict[i].model.fd_model.parameters(),self.clients_dict[i].model.personal_model.parameters()):
                value2.data=value1.data.clone()




    
        
