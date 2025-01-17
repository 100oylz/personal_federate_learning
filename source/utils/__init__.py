import random

import torchvision.transforms
from sklearn.model_selection import train_test_split
from .partition import *
from torchvision.datasets import MNIST
import torch
import numpy as np
import pandas as pd


def get_path(dataset_name: str, code: PartitionCode):
    """
    根据给定的数据集名称和分区代码生成相关的文件路径。

    参数:
        dataset_name (str): 数据集的名称。

        code (PartitionCode): 分区代码，用于确定文件路径的一部分。

    返回:
        partation_path (str): 数据分区文件的路径。

        label_csv (str): 数据分区报告的 CSV 文件路径。

        label_png (str): 数据分区报告的 PNG 文件路径。
    """

    str_insert = dataset_name + '_' + code.value
    partation_path = "data/" + str_insert + ".pkl"
    label_csv = "log/" + str_insert + ".csv"
    label_png = "log/" + str_insert + ".png"
    maml_path="data/" + str_insert + "_maml.pkl"
    return partation_path, label_csv, label_png,maml_path


def partation_main(args):
    """
    主函数，用于生成数据分区文件并生成数据分区报告。
    """
    dataset_name = args.dataset_name
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
    partation_path, label_csv, label_png ,maml_path= get_path(dataset_name, code)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = MNIST(f'data/{dataset_name}', train=True, download=True, transform=transform)
    testset = MNIST(f'data/{dataset_name}', train=False, download=True, transform=transform)
    num_classes = max(int(trainset.targets.max() + 1), int(testset.targets.max() + 1))
    # 将训练集和测试集合并
    data = torch.cat([trainset.data, testset.data], dim=0)
    targets = torch.cat([trainset.targets, testset.targets], dim=0)
    # 将数据转换为浮点数并进行归一化
    data = data.float()
    data.div_(255)
    data = torchvision.transforms.Resize((32, 32), antialias=False)(data)
    data = torchvision.transforms.Normalize((0.1307,), (0.3081,))(data)
    # 设置数据和标签
    trainset.data = data
    trainset.targets = targets
    # 从参数中获取其他参数
    num_clients = args.num_clients
    dir_alpha = args.dir_alpha
    major_classes_num = args.major_classes_num
    seed = args.seed
    # 划分训练集和maml集
    train_data, maml_data, train_label, maml_label = train_test_split(trainset.data, trainset.targets,
                                                                      test_size=args.mamlratio, random_state=seed,
                                                                      stratify=trainset.targets)
    trainset.data = train_data
    trainset.targets = train_label
    # 生成mamldict
    maml_dict = {}
    for i in range(num_classes):
        maml_dict[i] = maml_data[np.where(maml_label.numpy() == i)]

    # 生成数据分区
    raw_data = get_partation(trainset.targets, num_clients, code, dir_alpha, major_classes_num, seed)
    # 生成报告文件
    report_partition(targets=trainset.targets, partition_data=raw_data, csv_file=label_csv)
    # 进行绘图
    plot_partition(label_csv, num_classes, label_png, show=False)
    # 保存数据分区
    save_dict(raw_data.client_dict, partation_path)
    # 保存maml数据
    save_dict(maml_dict, maml_path)

    # 获取客户端数据
    client_data = set_partition(trainset, raw_data.client_dict)

    return client_data, num_classes, maml_dict

def save_pt(model,path):
    torch.save(model.state_dict(), path)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_global_train_to_csv(global_train_dict,path,num_clients,num_epochs):
    clients_column=[]
    for i in range(num_clients):
        clients_column.append('client_'+str(i)+'_loss')
        clients_column.append('client_'+str(i)+'_acc')
    columns=['epoch']+clients_column+['meta_loss','meta_acc']
    rows=[item for item in range(num_epochs)]
    df=pd.DataFrame(columns=columns,index=rows)
    for i in range(num_epochs):
        df.loc[i,'epoch']=i
    for key,value in global_train_dict.items():
        if(key=='meta_loss' or key=='meta_acc'):
            for item in value:
                epoch,data=item
                df.loc[epoch,key]=data
        elif(key=='client_loss_dict'):
            for client_id,item in value.items():
                for epoch,data in item:
                    df.loc[epoch,'client_'+str(client_id)+'_loss']=data
        elif(key=='client_acc_dict'):
            for client_id,item in value.items():
                for epoch,data in item:
                    df.loc[epoch,'client_'+str(client_id)+'_acc']=data

    df.to_csv(path)

def save_personal_fit_to_csv(personal_fit_dict,path,num_clients,num_epochs):
    clients_column=[]
    for i in range(num_clients):
        clients_column.append('client_'+str(i)+'_loss')
        clients_column.append('client_'+str(i)+'_acc')
    columns=['epoch']+clients_column
    rows=[item for item in range(num_epochs)]
    df=pd.DataFrame(columns=columns,index=rows)
    for i in range(num_epochs):
        df.loc[i,'epoch']=i
    for key,value in personal_fit_dict.items():
        if(key=='client_loss_dict'):
            for client_id,item in value.items():
                for epoch,data in item:
                    df.loc[epoch,'client_'+str(client_id)+'_loss']=data
        elif(key=='client_acc_dict'):
            for client_id,item in value.items():
                for epoch,data in item:
                    df.loc[epoch,'client_'+str(client_id)+'_acc']=data
    df.to_csv(path)

