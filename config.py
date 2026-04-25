import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_channels", type=int, default=3, help="input image channel")
parser.add_argument("--class_nums", type=int, default=1000, help="class nums")
parser.add_argument("--channels", type=list, default=[32, 64, 128, 256], help="channel nums in stages")
parser.add_argument("--block_nums", type=list, default=[2, 2, 6, 2], help="block nums in stages")
parser.add_argument("--grid_size", type=int, default=7, help="grid size of kan")
parser.add_argument("--drop_path", type=float, default=0.1, help="drop path rate")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--decay_rate', type=float, default=1e-5, help='weight decay [default: 1e-4]')
parser.add_argument('--epoch', default=100, type=int, help='Epoch to run')
parser.add_argument('--warmup', default=5, type=int, help='Epoch for warmup')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_workers', type=int, default=4, help='num workers of the Data Loader')
parser.add_argument('--local_rank', dest='local_rank', type=int, default=0)

args = parser.parse_args()
