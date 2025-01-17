from argparse import ArgumentParser

ROOTPATH='data/MNIST'
IIDPATH='data/MNIST_iid'
UNBALANCEPATH='data/MNIST_unbalance'
NONIID_LABELPATH='data/MNIST_noniid_label'
NONIID_LABELDIRPATH='data/MNIST_noniid_labeldir'



def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--dir_alpha', type=float, default=0.3)
    parser.add_argument('--major_classes_num', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--code', type=int, default=1,choices=[1,2,3,4])
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mamlratio', type=float, default=0.01)
    parser.add_argument('--n_way', type=int, default=10)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--k_query', type=int, default=1)
    parser.add_argument('--inner_step', type=int, default=3)
    parser.add_argument('--inner_lr', type=float, default=1e-4)
    parser.add_argument('--outer_lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--client_lr',type=float,default=0.001)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--fit_epoch', type=int, default=10)
    args = parser.parse_args()
    return args