
from .baseServer import BaseServer
from source.nn.client_model import client_model,client_load_params,client_train,personal_fit,client_model_test
from sklearn.model_selection import train_test_split
import torch
from .utils.dataloader import BaseDataset
class Client(BaseServer):
    def __init__(self,server_public_keys,in_channel,num_features,args):
        super().__init__()
        self.test_loader = None
        self.train_loader = None
        self.model :client_model|None= None
        self.server_public_keys = None
        self.data=None
        self.train_data=None
        self.test_data=None
        self.num_features=num_features
        self.args=args
        self.set_server_public_keys(server_public_keys)
        self.set_model(in_channel,self.num_features)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.args.client_lr)
        self.criterion=torch.nn.CrossEntropyLoss()
        self.best_loss=float('inf')

    
    def set_server_public_keys(self,server_public_keys):
        self.server_public_keys=server_public_keys

    def get_data(self, data):
        self.data = data
        data,label=self.data
        data=torch.unsqueeze(data,1)
        data=data.numpy()
        label=label.numpy()
        train_data,test_data,train_label,test_label = train_test_split(data,label, test_size=self.args.test_size, random_state=self.args.seed)
        self.train_data=BaseDataset(torch.from_numpy(train_data),torch.from_numpy(train_label))
        self.test_data=BaseDataset(torch.from_numpy(test_data),torch.from_numpy(test_label))
        self.train_loader=torch.utils.data.DataLoader(self.train_data,batch_size=self.args.batch_size)
        self.test_loader=torch.utils.data.DataLoader(self.test_data,batch_size=self.args.batch_size)

    def set_model(self,in_channel,num_features):
        self.model=client_model(in_channel,num_features)

    def model_train(self,client_id,file_base_name,code_value,mean,std):

        grads_list,client_loss,client_acc=client_train(self.model,self.train_loader,self.optimizer,self.criterion,self.args.device,client_id,file_base_name,code_value,mean,std)

        return grads_list,client_loss,client_acc

    def personal_fit(self,client_id,file_base_name,code_value):
        client_loss,client_acc=personal_fit(self.model,self.train_loader,self.optimizer,self.criterion,self.args.device,client_id,file_base_name,code_value)

        return self.model.named_parameters(),client_loss,client_acc

    def test(self,client_id,file_base_name,code_value):
        test_loss,test_acc=client_model_test(self.model,self.test_loader,self.criterion,self.args.device,client_id,file_base_name,code_value)
        return test_loss,test_acc



