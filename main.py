from source.server import Server
from args import get_args
import torch
from source.utils import set_seed
from tqdm import tqdm
from source.utils.logger import getlogger
from source.utils.partition import PartitionCode
from source.utils import save_global_train_to_csv,save_personal_fit_to_csv
if __name__ == '__main__':
    args=get_args()
    if (args.code == 1):
        code = PartitionCode.IID
    elif (args.code == 2):
        code = PartitionCode.NONIID_LABEL
    elif (args.code == 3):
        code = PartitionCode.NONIID_LABELDIR
    elif (args.code == 4):
        code = PartitionCode.UNBALANCE
    else:
        raise ValueError("Invalid code")
    logger=getlogger(f'{code.value}',f'log/{code.value}.log')
    save_code_value=code.value.replace('-','')
    save_code_value=save_code_value.replace('#','')
    set_seed(args.seed)
    server=Server(1,args)
    iter_bar=tqdm(range(args.num_epochs))
    global_train_dict={}
    global_train_dict['meta_loss']=[]
    global_train_dict['meta_acc']=[]
    global_train_dict['client_loss_dict']={}
    global_train_dict['client_acc_dict']={}
    for i in range(args.num_clients):
        global_train_dict['client_loss_dict'][i]=[]
        global_train_dict['client_acc_dict'][i]=[]
    save_pth_base_name='save/{}/{}_best.pth'
    for i in iter_bar:
        iter_bar.set_description(f'Global Train epoch:{i}')
        meta_loss,meta_acc,client_loss_dict,client_acc_dict=server.update(args.outer_lr,save_pth_base_name,save_code_value,args.noise_mean,args.noise_std)
        meta_loss=meta_loss.item()
        logger.info(f'epoch:{i},meta_loss:{meta_loss},meta_acc:{meta_acc}')
        logger.info(client_loss_dict)
        logger.info(client_acc_dict)
        global_train_dict['meta_loss'].append((i,meta_loss))
        global_train_dict['meta_acc'].append((i,meta_acc))
        for (key1,value1),(key2,value2) in zip(client_loss_dict.items(),client_acc_dict.items()):
            global_train_dict['client_loss_dict'][key1].append((i,value1))
            global_train_dict['client_acc_dict'][key2].append((i,value2))
    save_global_train_to_csv(global_train_dict,f'log/{save_code_value}_global_train.csv',args.num_clients,args.num_epochs)
    fit_bar=tqdm(range(args.fit_epochs))
    personal_fit_dict={}
    personal_fit_dict['client_loss_dict']={}
    personal_fit_dict['client_acc_dict']={}
    for i in range(args.num_clients):
        personal_fit_dict['client_loss_dict'][i]=[]
        personal_fit_dict['client_acc_dict'][i]=[]
    for i in fit_bar:
        fit_bar.set_description(f'Personal Fit epoch:{i}')
        client_loss_dict,client_acc_dict=server.client_personal_fit(save_pth_base_name,save_code_value)
        logger.info(client_loss_dict)
        logger.info(client_acc_dict)
        for (key1,value1),(key2,value2) in zip(client_loss_dict.items(),client_acc_dict.items()):
            personal_fit_dict['client_loss_dict'][key1].append((i,value1))
            personal_fit_dict['client_acc_dict'][key2].append((i,value2))
    save_personal_fit_to_csv(personal_fit_dict,f'log/{save_code_value}_personal_fit.csv',args.num_clients,args.fit_epochs)









