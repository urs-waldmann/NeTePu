import argparse
import os
import yaml

import torch
import numpy as np

from data.synthDataset import get_synthetic_texture_dataset
from models.VAE import get_texture_encoder
from data.utils import view_dataset

parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-checkpoint', required=True, type=int)
parser.add_argument('-exp_name_pose', type=str, default=None)
parser.add_argument('-data', required=True, type=str)
parser.add_argument('-res_factor', type=float, default=1.0)
parser.add_argument('-npixels_per_batch', type=int, default=50000)
parser.add_argument('-novel_view_cfg_path', type=str)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

exp_dir = './experiments/{}/'.format(args.exp_name)
fname = exp_dir + 'configs.yaml'
with open(fname, 'r') as f:
    print('Loading config file from: ' + fname)
    CFG = yaml.safe_load(f)

print('[CONFIG]', CFG)

radius = CFG['data']['radius']
CAMS = [2, 6]
nkps = CFG['data']['nkps']

texture_encoder = get_texture_encoder(CFG)

CFG['training']['npoints_decoder'] = 10
print('Set new value for training.batch_size in CONFIG:')
CFG['training']['batch_size'] = 1
print('[INFO]', str(CFG['training']['batch_size']))
mode = 'track_texture_occlusions'
dataset = get_synthetic_texture_dataset(
    data_type=args.data,
    mode=mode,
    sup_distr='uniform',
    cfg=CFG,
    cams=CAMS,
    exp_name_pos_enc=args.exp_name_pose
)

# visualize n track_texture_novel_views data set samples from random camera views
# n_samples = 50
# print('Show {} samples from "track texture occlusions" set.'.format(n_samples))
# view_dataset(
#     data_loader=dataset.get_loader(shuffle=False),
#     n_samples=n_samples,
#     n_cams=len(CAMS),
#     masked_image=True
# )

device = torch.device("cuda")
texture_encoder = texture_encoder.to(device)
texture_encoder.eval()

# load parameters
checkpoint_path = exp_dir + 'checkpoints/checkpoint_epoch_{}.tar'.format(args.checkpoint)
print('Load checkpoint from: {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path, map_location=device)
print('Load learned parameters for texture encoder')
texture_encoder.load_state_dict(checkpoint['texture_encoder_state_dict'])

#iterate through dataset and store images
loader = dataset.get_loader(shuffle=False)

number_frames = len(dataset.steps)
number_textures = len(dataset.path)
number_cams = len(CAMS)
z_texture_all = np.empty([number_frames, number_textures, number_cams, CFG['encoder']['lat_dim']])
with torch.no_grad():
    for i, data in enumerate(loader):
        print('Rendering example {}/{}'.format(i+1, len(loader)))
        frame = data.get('frame').item()
        print('[INFO] frame', frame)
        texture = data.get('texture')
        print('[INFO] texture', texture.item())
        gt_masked_color = [d.to(device).squeeze() for d in data.get('masked_color_maps')]
        pos_enc = [d.to(device).squeeze() for d in data.get('pos_enc_maps')]

        texture_encoder_input = [
            torch.cat((masked_color_map.permute(2, 0, 1), pos_enc_map.permute(2, 0, 1)), dim=0)
            for masked_color_map, pos_enc_map in zip(gt_masked_color, pos_enc)
        ]

        for c_idx, cam_name in enumerate(CAMS):
            # texture encoding
            _, _, z_texture = texture_encoder(texture_encoder_input[c_idx].unsqueeze(0))
            z_texture_all[frame - 1, texture.item(), c_idx, :] = z_texture.squeeze().detach().cpu().numpy()

    np.save(
        os.path.join(
            exp_dir,
            'z_texture_' + mode[mode.rfind('occlusions'):] + '.npy'
        ),
        z_texture_all
    )
