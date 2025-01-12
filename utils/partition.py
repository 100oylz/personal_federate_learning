import torchvision.datasets
from fedlab.utils.dataset import MNISTPartitioner
from fedlab.utils.functional import partition_report

from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd
import pickle


class PartitionCode(Enum):
    """
      定义了数据分区的不同类型。

      属性:
          IID: 独立同分布数据分区。
          NONIID_LABEL: 非独立同分布数据分区，基于标签的。
          NONIID_LABELDIR: 非独立同分布数据分区，基于标签分布的。
          UNBALANCE: 不平衡数据分区。
      """
    IID = "iid"
    NONIID_LABEL = "noniid-#label"
    NONIID_LABELDIR = "noniid-labeldir"
    UNBALANCE = "unbalance"


def get_partation(targets, num_clients: int, code: PartitionCode, dir_alpha: float, major_classes_num: int, seed: int):
    """
    根据给定的参数生成数据分区。

    参数:
        targets (list or array-like): 数据的标签列表。
        num_clients (int): 客户端的数量。
        code (PartitionCode): 数据分区的类型。
        dir_alpha (float): Dirichlet 分布的参数，用于控制数据的非独立同分布程度。
        major_classes_num (int): 主要类别的数量。
        seed (int): 随机数种子，用于保证分区的可重复性。

    返回:
        partition (MNISTPartitioner): 生成的数据分区对象。
    """
    partition = MNISTPartitioner(targets, num_clients, code.value, dir_alpha, major_classes_num, seed=seed)
    return partition



def report_partition(targets, partition_data, csv_file):
    """
    生成数据分区的报告，并将报告保存到指定的 CSV 文件中。

    参数:
        targets (list or array-like): 数据的标签列表。
        partition_data (MNISTPartitioner): 数据分区对象。
        csv_file (str): 要保存报告的 CSV 文件路径。

    返回:
        None
    """
    # 使用 partition_report 函数生成数据分区的报告
    partition_report(targets, partition_data.client_dict, file=csv_file)


def plot_partition(csv_file, num_classes, save_path: str, show: bool = False):
    """
    绘制数据分区的柱状图，并将图表保存到指定的文件路径。

    参数:
        csv_file (str): 包含数据分区信息的 CSV 文件路径。
        num_classes (int): 数据集中的类别数量。
        save_path (str): 图表保存的文件路径。
        show (bool): 是否显示图表。默认为 False。

    返回:
        None
    """
    # 生成列名列表，用于表示每个类别的数据
    col_names = [f"class{i}" for i in range(num_classes)]
    # 设置图表的背景颜色为白色
    plt.rcParams['figure.facecolor'] = 'white'
    # 读取 CSV 文件，跳过第一行（可能是标题行）
    df = pd.read_csv(csv_file, header=1)
    # 将 'client' 列设置为索引
    df = df.set_index('client')
    # 遍历每个类别列，将其值乘以 'Amount' 列的整数部分，以计算每个类别的样本数量
    for col in col_names:
        df[col] = df[col] * df['Amount'].astype(int)
    # 绘制水平堆叠柱状图
    df[col_names].plot.barh(stacked=True)
    # 设置图例位置为 'center left'，并将其锚定在图表的 (1, 0.5) 位置
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # 设置 x 轴标签为 'sample num'
    plt.xlabel('sample num')
    # 将图表保存到指定的文件路径，dpi 为 400，边界框紧凑
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    # 如果 show 参数为 True，则显示图表
    if (show):
        plt.show()

def save_dict(data, save_path):
    """
    将数据对象的 client_dict 属性保存到指定路径的 pickle 文件中。

    参数:
        data (object): 包含 client_dict 属性的数据对象。
        save_path (str): 要保存的 pickle 文件路径。

    返回:
        None
    """
    with open(save_path, 'wb') as f:
        f.write(pickle.dumps(data.client_dict))


def load_dict(load_path):
    """
    从指定路径的 pickle 文件中加载 client_dict。

    参数:
        load_path (str): 要加载的 pickle 文件路径。

    返回:
        client_dict (dict): 从文件中加载的 client_dict。
    """
    with open(load_path, 'rb') as f:
        client_dict = pickle.load(f)
    return client_dict


def set_partition(datasets: torchvision.datasets.mnist.MNIST, client_dict: dict):
    """
    根据给定的 client_dict 为每个客户端设置数据分区。

    参数:
        datasets (torchvision.datasets.mnist.MNIST): MNIST 数据集对象。
        client_dict (dict): 包含客户端索引及其对应数据索引的字典。

    返回:
        client_data (dict): 包含每个客户端数据分区的字典。
    """
    client_data = {}
    datas = datasets.data
    targets = datasets.targets

    for key, value in client_dict.items():
        client_data[key] = (datas[value], targets[value])
    return client_data
