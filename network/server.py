from client import Client

from baseServer import BaseServer


class Server(BaseServer):
    def __init__(self, num_clients):
        super().__init__()
        self.num_clients=num_clients
        self.clients_dict = {i: Client(self.rsa_public_keys)
                        for i in range(self.num_clients)}
    def choose_client(self):
        pass

    def set_model(self):
        pass
    
    def model_global_train(self):
        pass
    
        
