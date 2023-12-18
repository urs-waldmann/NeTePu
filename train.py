from data.synthDataset import get_synthetic_dataset
from data.h36m import get_h36m_dataset
from models import training
import argparse
import os
import yaml
import torch

from models.NePu import get_encoder, get_decoder, get_renderer
from data.utils import view_dataset

parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-cfg_file', required=True, type=str)
parser.add_argument('-data', required=True, type=str)

parser.add_argument('-cuda_device', default=0, type=int)

parser.add_argument('-interp_dec', action='store_true')
parser.set_defaults(interp_dec=False)
parser.add_argument('-pointnet_enc', action='store_true')
parser.set_defaults(pointnet_enc=False)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

assert args.cfg_file is not None
CFG = yaml.safe_load(open(args.cfg_file, 'r'))

fname_data_cfg = './configs/data_configs.yaml'
with open(fname_data_cfg, 'r') as f:
    data_cfg = yaml.safe_load(f)

CFG['data'] = {}
CFG['data']['nkps'] = data_cfg['nkps'][args.data]
CFG['data']['ncams'] = data_cfg['ncams'][args.data]
CFG['data']['radius'] = data_cfg['radii'][args.data]
CFG['data']['num_datapoints'] = data_cfg['num_datapoints'][args.data]
CFG['data']['noise_aug_kpts'] = data_cfg['keypoint_noise_augmentation'][args.data]
CFG['data']['std_near'] = data_cfg['boundary_sigma_near'][args.data]
CFG['data']['std_far'] = data_cfg['boundary_sigma_far'][args.data]
CFG['data']['VCA_nneigh'] = data_cfg['VCA_nneigh'][args.data]
CFG['geometry'] = {}
CFG['geometry']['exp_name'] = args.exp_name
CFG['geometry']['data'] = args.data

exp_dir = './experiments/{}/'.format(args.exp_name)
fname = exp_dir + 'configs.yaml'
if not os.path.exists(exp_dir):
    print('Creating experiment dir: ' + exp_dir)
    os.makedirs(exp_dir)
    with open(fname, 'w') as yaml_file:
        yaml.safe_dump(CFG, yaml_file, default_flow_style=False)
else:
    with open(fname, 'r') as f:
        print('Loading config file from: ' + fname)
        CFG = yaml.safe_load(f)

# TODO print in nicer format!!
print('[CONFIG]', CFG)

torch.cuda.set_device(args.cuda_device)

if args.data == "h36m":
    train_dataset = get_h36m_dataset(data_type=args.data, mode='train', sup_distr='training', cfg=CFG)
    val_dataset = get_h36m_dataset(data_type=args.data, mode='val', sup_distr='training', cfg=CFG)
else:
    train_dataset = get_synthetic_dataset(data_type=args.data, mode='train', sup_distr='training', cfg=CFG)
    val_dataset = get_synthetic_dataset(data_type=args.data, mode='val', sup_distr='training', cfg=CFG)

# visualize n train/val data set samples from random camera views
# n_samples = 7
# print('Show {} samples from training set.'.format(n_samples))
# view_dataset(
#     data_loader=train_dataset.get_loader(shuffle=False),
#     n_samples=n_samples,
#     show_mask=True,
#     show_kpts=True  # implemented so far only for train & val split
# )
# print('Show {} samples from validation set.'.format(n_samples))
# view_dataset(
#     data_loader=val_dataset.get_loader(shuffle=False),
#     n_samples=n_samples,
#     show_mask=True,
#     show_kpts=True  # implemented so far only for train & val split
# )

device = torch.device("cuda")

encoder = get_encoder(CFG)
decoder = get_decoder(CFG)
renderer = get_renderer(CFG)

trainer = training.Trainer(
    encoder,
    decoder,
    renderer,
    CFG,
    device,
    train_dataset,
    val_dataset,
    args.exp_name,
    optimizer='Adam'
)
trainer.train_model(2000)
