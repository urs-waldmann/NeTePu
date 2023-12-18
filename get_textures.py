import argparse
import os
import yaml

import torch
import numpy as np

from time import time

from data.synthDataset import get_synthetic_texture_dataset
from data.h36m import get_h36m_texture_dataset
from models.VAE import get_texture_encoder
from models.NePu import get_encoder, get_decoder, get_renderer
from models.utils import get2Dkps

parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name_texture', required=True, type=str)
parser.add_argument('-checkpoint_texture', required=True, type=int)
parser.add_argument('-exp_name_geometry', required=True, type=str)
parser.add_argument('-checkpoint_geometry', required=True, type=int)
parser.add_argument('-data', required=True, type=str)
parser.add_argument('-res_factor', type=float, default=1.0)
parser.add_argument('-npixels_per_batch', type=int, default=50000)
parser.add_argument('-novel_view_cfg_path', type=str)
parser.add_argument("-load_pos_enc", action='store_true')

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

exp_dir = './experiments/{}/'.format(args.exp_name_texture)
exp_dir_pose = './experiments/{}/'.format(args.exp_name_geometry)
fname = exp_dir + 'configs.yaml'
with open(fname, 'r') as f:
    print('Loading config file from: ' + fname)
    CFG = yaml.safe_load(f)
CFG_geometry = yaml.safe_load(open(os.path.join('./experiments', CFG['geometry']['exp_name'], 'configs.yaml'), 'r'))

print('[CONFIG]', CFG)

radius = CFG['data']['radius']
CAMS = list(range(CFG['data']['ncams']))
nkps = CFG['data']['nkps']

encoder = get_encoder(CFG_geometry)
texture_encoder = get_texture_encoder(CFG)
decoder = get_decoder(CFG_geometry)
renderer = get_renderer(CFG_geometry)

CFG['training']['npoints_decoder'] = 10
print('Set new value for training.batch_size in CONFIG:')
CFG['training']['batch_size'] = 1
print('[INFO]', str(CFG['training']['batch_size']))
mode = 'track_texture'
if args.data == 'h36m':
    dataset = get_h36m_texture_dataset(
        data_type=args.data,
        mode=mode,
        sup_distr='uniform',
        cfg=CFG,
        cams=['54138969', '55011271', '58860488', '60457274']
    )
else:
    dataset = get_synthetic_texture_dataset(
        data_type=args.data,
        mode=mode,
        sup_distr='uniform',
        cfg=CFG,
        cams=CAMS,
        exp_name_pos_enc=args.exp_name_geometry
    )

device = torch.device("cuda")
encoder = encoder.to(device)
texture_encoder = texture_encoder.to(device)
decoder = decoder.to(device)
renderer = renderer.to(device)
encoder.eval()
decoder.eval()
renderer.eval()
texture_encoder.eval()

# load parameters for pose models
checkpoint_path = exp_dir_pose + 'checkpoints/checkpoint_epoch_{}.tar'.format(args.checkpoint_geometry)
print('Load checkpoint from: {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path, map_location=device)
print('Load learned parameters for "pose" encoder, decoder and renderer')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
if not CFG['renderer']['type'] == 'lfn':
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
renderer.load_state_dict(checkpoint['decoder_impl_state_dict'])
# load parameters for texture encoder
checkpoint_path = exp_dir + 'checkpoints/checkpoint_epoch_{}.tar'.format(args.checkpoint_texture)
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
time_total = 0
with torch.no_grad():
    for i, data in enumerate(loader):
        print('Rendering example {}/{}'.format(i+1, len(loader)))
        frame = data.get('frame').item()
        print('[INFO] frame', frame)
        texture = data.get('texture')
        print('[INFO] texture', texture.item())
        inp_pos = data.get('input_pos').to(device)
        inp_feats = data.get('input_feats').to(device)
        camera_params_tmp = data.get('camera_params')
        camera_params = [{k: v.to(device) for (k, v) in zip(c_params.keys(), c_params.values())}
                         for c_params in camera_params_tmp]
        gt_masked_color = [d.to(device).squeeze() for d in data.get('masked_color_maps')]
        if args.data != 'h36m':
            pos_enc = [d.to(device).squeeze() for d in data.get('pos_enc_maps')]

        for c_idx in CAMS:
            start_time = time()
            if args.data != 'h36m':
                if not args.load_pos_enc:
                    xres = int(gt_masked_color[c_idx].shape[1] * args.res_factor)
                    yres = int(gt_masked_color[c_idx].shape[0] * args.res_factor)
                    xx, yy = np.meshgrid(np.arange(xres), np.arange(yres))
                    if not CFG['renderer']['type'] == 'lfn':
                        xx = xx / xres
                        yy = yy / yres
                    img_coords = torch.from_numpy(np.stack([xx, yy], axis=-1))\
                        .float().reshape(-1, 2).unsqueeze(0).to(device)

                    start_time = time()

                    z = encoder(inp_pos, inp_feats)
                    encoding = decoder(z)
                    if 'anchors' in encoding:
                        encoding['anchors'] *= 2 * radius
                    kps_2d_ = get2Dkps(camera_params[c_idx], inp_pos * 2 * radius, gt_masked_color[c_idx])

                    coord_chunks = torch.split(img_coords, args.npixels_per_batch, dim=1)
                    logit_chunks = []
                    pos_enc_chunks = []
                    for coords in coord_chunks:
                        chunk, _, chunk_pe = renderer(
                            coords,
                            encoding,
                            camera_params[c_idx],
                            kps_2d_
                        )
                        logit_chunks.append(chunk.squeeze().detach())
                        pos_enc_chunks.append(chunk_pe.squeeze().detach())

                    logits = torch.cat(logit_chunks, dim=0)
                    pos_enc = torch.cat(pos_enc_chunks, dim=0)
                    rec_img = torch.sigmoid(logits.reshape(yres, xres))
                    red_pos_enc = pos_enc[:, 0].reshape(yres, xres)
                    green_pos_enc = pos_enc[:, 1].reshape(yres, xres)
                    blue_pos_enc = pos_enc[:, 2].reshape(yres, xres)
                    rec_pos_enc = torch.clamp(torch.stack([red_pos_enc, green_pos_enc, blue_pos_enc], dim=-1) * 255, 0, 255)

                    threshold = 0.5
                    rec_pos_enc[rec_img <= threshold] = 0
                    rec_pos_enc = rec_pos_enc / 255
                else:
                    rec_pos_enc = pos_enc[c_idx]

                texture_encoder_input = torch.cat(
                    (gt_masked_color[c_idx].permute(2, 0, 1), rec_pos_enc.permute(2, 0, 1)), dim=0
                )
            else:
                texture_encoder_input = gt_masked_color[c_idx].permute(2, 0, 1)

            # texture encoding
            _, _, z_texture = texture_encoder(texture_encoder_input.unsqueeze(0))
            z_texture_all[frame - 1, texture.item(), c_idx, :] = z_texture.squeeze().detach().cpu().numpy()

            end_time = time()
            time_total += (end_time - start_time)

    # calculate runtime
    runtime = (number_frames * number_textures * number_cams) / time_total
    print('Runtime: {} fps'.format(runtime))

    np.save(
        os.path.join(
            exp_dir,
            'z_texture_{}.npy'.format(args.data)
        ),
        z_texture_all
    )
