import argparse

from yaml import parse

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script")
    # default parameters
    parser.add_argument('-na','--name',type=str, default='ICLR18',help='method name')    
    parser.add_argument('-d','--dataset',type=str, default='/home/data/CLIC2020/',help='Training dataset')
    parser.add_argument("--learning_rate",type=float, default=1e-4,help="initial learning rate.")
    parser.add_argument('--N',default=128,type=int, help='Number of Channels')
    parser.add_argument('--M',default=192,type=int, help='Number of Channels in last layer of encoder and hyperdecoder')
    parser.add_argument('-e','--epochs',default=100, type=int, help='Number of epochs (default: %(default)s)')
    parser.add_argument( '-n', '--num-workers', type=int, default=8, help='Dataloaders threads (default: %(default)s)')
    parser.add_argument('--batch-size','-b',type=int, default=16, help='Batch size (default: %(default)s)')
    parser.add_argument('--test-batch-size',type=int, default=1, help='Test batch size (default: %(default)s)')
    parser.add_argument('--aux-learning-rate', default=1e-3, help='Auxiliary loss learning rate (default: %(default)s)')
    parser.add_argument('--patch-size',type=int, nargs=2, default=(256, 256),help='Size of the patches to be cropped (default: %(default)s)')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use cuda')
    parser.add_argument('--save',action='store_true', default=True, help='Save model to disk')
    parser.add_argument('-m','--metric',type=str, default='mse', help='Optimized metric, choose from mse or msssim or identity')
    parser.add_argument('-q','--quality',type=int, default=0, required=True, help='Quality levels (1: lowest, highest: 8)')
    parser.add_argument('--out_dir',type=str, default='logdir_energy',help='path of saved models')
    parser.add_argument('--en',action='store_true', help='flag of enhancement for model')
    parser.add_argument('--seed',type=float, default=100, help='Set random seed for reproducibility')
    parser.add_argument('--clip_max_norm', default=5, type=float,help='gradient clipping max norm')
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument('--pretrain', action='store_true', help='flag of pretrain')
    parser.add_argument('--nt', default='GDN', type=str, required=True, help='flag of Nonlinear Transform')
    parser.add_argument('--SIC', default=50, type=int, help='The inference time (default: %(default)s)')
# 
    # yapf: enable

    parser.add_argument('--model', type=str, default='M1', choices=['M1','M2','M3'])
    parser.add_argument('--resume', action='store_true', help="resume training")

    args = parser.parse_args(argv)
    return args
