from source.server import Server
from args import get_args
import torch
from source.utils import set_seed
from tqdm import tqdm
if __name__ == '__main__':
    args=get_args()
    set_seed(args.seed)
    server=Server(1,args)
    iter_bar=tqdm(range(args.num_epochs))
    for i in iter_bar:

        meta_loss,meta_acc,client_loss_dict,client_acc_dict=server.update()
        print(f'epoch:{i},meta_loss:{meta_loss},meta_acc:{meta_acc}')
        iter_bar.update(1)
        print(client_loss_dict)
        print(client_acc_dict)
    fit_bar=tqdm(range(args.fit_epochs))
    for i in fit_bar:
        meta_loss,meta_acc,client_loss_dict,client_acc_dict=server.client_personal_fit()
        print(f'epoch:{i},meta_loss:{meta_loss},meta_acc:{meta_acc}')
        fit_bar.update(1)
        print(client_loss_dict)
        print(client_acc_dict)






