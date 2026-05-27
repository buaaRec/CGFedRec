import argparse
from utils import *

parser = argparse.ArgumentParser()
# Common
parser.add_argument('--alias', type=str, default='FedCIA')
parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
parser.add_argument('--clients_sample_num', type=int, default=100)
parser.add_argument('--clients_sample_method', type=str, default='same', choices=['random', 'same'])
parser.add_argument('--num_round', type=int, default=100)
parser.add_argument('--local_epoch', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--dataset', type=str, default='100k')
parser.add_argument('--latent_dim', type=int, default=32)
parser.add_argument('--num_negative', type=int, default=4)
parser.add_argument('--l2_regularization', type=float, default=0.)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--neg_num', type=int, default=198)
parser.add_argument('--evaluate', type=str, default='part', choices=['part', 'all'])
parser.add_argument('--num_users', type=int)
parser.add_argument('--num_items', type=int)
parser.add_argument('--model_dir', type=str, default='checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model')
parser.add_argument('--dp', type=float, default=float)
parser.add_argument('--bit', type=int, default=-1)

parser.add_argument('--lr_eta', type=int, default=80)
parser.add_argument('--rho', type=float, default=0)

# CGFedRec CoFedRec SGFedRec
parser.add_argument('--item_cluster', type=int, default=5)
parser.add_argument('--reg', type=float, default=0.1)
parser.add_argument('--cl_t', type=float, default=0.1)
parser.add_argument('--base_t', type=float, default=0.5)


# GPFedRec
parser.add_argument('--layers', type=str, default='64, 32, 16, 8')
parser.add_argument('--neighborhood_size', type=int, default=0)
parser.add_argument('--neighborhood_threshold', type=float, default=1.)
parser.add_argument('--mp_layers', type=int, default=1)
parser.add_argument('--similarity_metric', type=str, default='cosine')

# FedCA
parser.add_argument('--agg_clients_ratio', type=float, default=0.1)
parser.add_argument('--k_principal', type=int, default=4)
parser.add_argument('--alpha', type=float, default=0.3)
parser.add_argument('--beta', type=float, default=0.3)
parser.add_argument('--interpolation', type=float, default=0.9)

# FedRAP
parser.add_argument('--lambda', type=float, default=0.1)
parser.add_argument('--mu', type=float, default=0.1)
parser.add_argument('--regular', type=str, default='l1')
parser.add_argument('--vary_param', type=str, default='tanh')
parser.add_argument('--lr_network', type=float, default=5e-1)
parser.add_argument('--lr_args', type=float, default=1e2)
parser.add_argument('--decay_rate', type=float, default=1)

# FedCIA
# parser.add_argument('--sim_lr', type=float, default=1e-1)
parser.add_argument('--sim_epoch', type=float, default=0)
args = parser.parse_args()

# Model.
config = vars(args)
config['sim_lr'] = config['lr']
if len(config['layers']) > 1:
    config['layers'] = [int(item) for item in config['layers'].split(',')]
else:
    config['layers'] = int(config['layers'])

if config['dataset'] == 'ml-1m':
    config['num_users'] = 6040
    config['num_items'] = 3706
elif config['dataset'] == '100k':
    config['num_users'] = 943
    config['num_items'] = 1682
elif config['dataset'] == 'lastfm-2k':
    config['num_users'] = 1600
    config['num_items'] = 12454
elif config['dataset'] == 'filmtrust':
    config['num_users'] = 1227
    config['num_items'] = 2059
elif config['dataset'] == 'supplier':
    config['num_users'] = 1081
    config['num_items'] = 21517
elif config['dataset'] == 'beauty':
    config['num_users'] = 253
    config['num_items'] = 356
elif config['dataset'] == 'card':
    config['num_users'] = 377
    config['num_items'] = 129
elif config['dataset'] == 'video':
    config['num_users'] = 94762
    config['num_items'] = 25612
elif config['dataset'] == 'ku':
    config['num_users'] = 204
    config['num_items'] = 560
elif config['dataset'] == 'hetrec':
    config['num_users'] = 2113
    config['num_items'] = 10109
else:
    pass
config['top_k'] = [5, 10, 20, 50, 100]

setup_seed(config['seed'])