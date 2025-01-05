
import torchvision.datasets
from fedlab.utils.dataset import MNISTPartitioner
from fedlab.utils.functional import partition_report

from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd
import pickle

class PartitionCode(Enum):
    IID = "iid"
    NONIID_LABEL = "noniid-#label"
    NONIID_LABELDIR = "noniid-labeldir"
    UNBALANCE = "unbalance"


def get_partation(targets, num_clients: int, code: PartitionCode, dir_alpha: float, major_classes_num: int, seed: int):
    partition = MNISTPartitioner(targets, num_clients, code.value, dir_alpha, major_classes_num, seed=seed)
    return partition


def report_partition(targets, partition_data, csv_file):
    partition_report(targets, partition_data.client_dict, file=csv_file)


def plot_partition(csv_file, num_classes, save_path: str, show: bool = False):
    col_names = [f"class{i}" for i in range(num_classes)]

    plt.rcParams['figure.facecolor'] = 'white'
    df = pd.read_csv(csv_file, header=1)
    df = df.set_index('client')
    for col in col_names:
        df[col] = df[col] * df['Amount'].astype(int)
    df[col_names].plot.barh(stacked=True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('sample num')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    if (show):
        plt.show()

def save_dict(data,save_path):
    with open(save_path,'wb') as f:
        f.write(pickle.dumps(data.client_dict))

def load_dict(load_path):
    with open(load_path,'rb') as f:
        client_dict=pickle.load(f)
    return client_dict

def set_partition(datasets:torchvision.datasets.mnist.MNIST,client_dict:dict):
    client_data={}
    datas=datasets.data
    targets=datasets.targets

    for key, value in client_dict.items():
        client_data[key]=(datas[value],targets[value])
    return client_data


