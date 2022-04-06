import torch
import torch.nn as nn
import torchvision
import torch.multiprocessing as mp
import torch.distributed as dist
from argparse import ArgumentParser
import os
from train import train
import psutil
"""
python train_ddp.py --nodes=2 --local_ranks=0 --ip_adress=192.168.100.2 --ngpus=1 --epochs=100
"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--local_ranks', default=0, type=int,
                        help="Node's order number in [0, num_of_nodes-1]")
    parser.add_argument('--ip_adress', type=str, required=True,
                        help='ip address of the host node')
    parser.add_argument("--checkpoint", default=None,
                        help="path to checkpoint to restore")
    parser.add_argument('--ngpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--iface', default='auto', type=str,
                        help='internet interface')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()
    # Total number of gpus availabe to us.
    args.world_size = args.ngpus * args.nodes
    # add the ip address to the environment variable so it can be easily avialbale
    if args.iface == 'auto':
        iface = list(filter(lambda x: 'en' in x, psutil.net_if_addrs().keys()))[0]
    os.environ['GLOO_SOCKET_IFNAME'] = iface
    os.environ['MASTER_ADDR'] = args.ip_adress
    print("ip_adress is", args.ip_adress)
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    # nprocs: number of process which is equal to args.ngpu here
    mp.spawn(train, nprocs=args.ngpus, args=(args,), join=True)
