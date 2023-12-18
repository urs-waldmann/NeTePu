from data.synthDataset import get_synthetic_texture_dataset
from data.h36m import get_h36m_texture_dataset
from models import training_texture
import argparse
import os
import yaml
import torch

from models.NePu import get_encoder, get_decoder, get_texture_decoder, get_texture_renderer
from models.VAE import get_texture_encoder
from data.utils import view_dataset

parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-cfg_file', required=True, type=str)

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
CFG_geometry = yaml.safe_load(open(os.path.join('./experiments', CFG['geometry']['exp_name'], 'configs.yaml'), 'r'))

fname_data_cfg = './configs/data_configs.yaml'
with open(fname_data_cfg, 'r') as f:
    data_cfg = yaml.safe_load(f)

CFG['data'] = {}
CFG['data']['nkps'] = data_cfg['nkps'][CFG_geometry['geometry']['data']]
CFG['data']['ncams'] = data_cfg['ncams'][CFG_geometry['geometry']['data']]
CFG['data']['radius'] = data_cfg['radii'][CFG_geometry['geometry']['data']]
CFG['data']['num_datapoints'] = data_cfg['num_datapoints'][CFG_geometry['geometry']['data']]
CFG['data']['noise_aug_kpts'] = data_cfg['keypoint_noise_augmentation'][CFG_geometry['geometry']['data']]
CFG['data']['std_near'] = data_cfg['boundary_sigma_near'][CFG_geometry['geometry']['data']]
CFG['data']['std_far'] = data_cfg['boundary_sigma_far'][CFG_geometry['geometry']['data']]
CFG['data']['VCA_nneigh'] = data_cfg['VCA_nneigh'][CFG_geometry['geometry']['data']]
CFG['texture'] = {}
CFG['texture']['exp_name'] = args.exp_name
CFG['texture']['data'] = CFG_geometry['geometry']['data']

exp_dir = './experiments/{}/'.format(CFG['texture']['exp_name'])
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

#TODO print in nicer format!!
print('[CONFIG]', CFG)

torch.cuda.set_device(args.cuda_device)

if CFG['texture']['data'] == 'h36m':
    train_dataset = get_h36m_texture_dataset(
        data_type=CFG['texture']['data'],
        mode='train',
        sup_distr='training',
        cfg=CFG
    )
    val_dataset = get_h36m_texture_dataset(
        data_type=CFG['texture']['data'],
        mode='val',
        sup_distr='training',
        cfg=CFG
    )
else:
    train_dataset = get_synthetic_texture_dataset(
        data_type=CFG['texture']['data'],
        mode='train',
        sup_distr='training',
        cfg=CFG,
        exp_name_pos_enc=None
    )
    val_dataset = get_synthetic_texture_dataset(
        data_type=CFG['texture']['data'],
        mode='val',
        sup_distr='training',
        cfg=CFG,
        exp_name_pos_enc=None
    )

# visualize n train/val data set samples from random camera views
# n_samples = 7
# print('Show {} samples from training set.'.format(n_samples))
# view_dataset(
#     data_loader=train_dataset.get_loader(shuffle=False),
#     n_samples=n_samples,
#     masked_image=True,
#     show_kpts=True  # implemented so far only for train & val split
# )
# print('Show {} samples from validation set.'.format(n_samples))
# view_dataset(
#     data_loader=val_dataset.get_loader(shuffle=False),
#     n_samples=n_samples,
#     masked_image=True,
#     show_kpts=True  # implemented so far only for train & val split
# )

device = torch.device("cuda")

encoder = get_encoder(CFG_geometry)
texture_encoder = get_texture_encoder(CFG)
decoder = get_decoder(CFG_geometry)
texture_decoder = get_texture_decoder(CFG)
texture_renderer = get_texture_renderer(CFG)

# load parameters for encoder and decoder
checkpoint_path = os.path.join('./experiments', CFG['geometry']['exp_name']) +\
                  '/checkpoints/checkpoint_epoch_{}.tar'.format(CFG['geometry']['checkpoint'])
print('Load checkpoint from: {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path, map_location=device)
print('Load learned parameters for geometry encoder and geometry decoder')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

trainer = training_texture.Trainer(
    encoder,
    texture_encoder,
    decoder,
    texture_decoder,
    texture_renderer,
    CFG,
    device,
    train_dataset,
    val_dataset,
    CFG['texture']['exp_name'],
    optimizer='Adam'
)
trainer.train_model(8000)
