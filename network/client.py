import os
import shutil
import rsa
import hashlib
from baseServer import BaseServer
from ..nn.client_model import client_model,LocalUpdate
class Client(BaseServer):
    def __init__(self,server_public_keys,in_channel,num_features,data_loader):
        super().__init__()
        self.model = None
        self.server_public_keys = None
        self.set_server_public_keys(server_public_keys)
        self.set_model(in_channel,num_features)
        
    
    def set_server_public_keys(self,server_public_keys):
        self.server_public_keys=server_public_keys

    def get_data_loader(self,filepath:str):
        with open(filepath,'rb') as f:
            data=f.read()
        raise NotImplementedError
        
    def set_model(self,in_channel,num_features):
        self.model=client_model(in_channel,num_features)

    def model_train(self,global_model_params):
        raise NotImplementedError



