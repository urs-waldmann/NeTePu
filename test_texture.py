import argparse
import os
import yaml

import torch
import numpy as np

from data.synthDataset import get_synthetic_texture_dataset
from data.h36m import get_h36m_texture_dataset
from models.NePu import get_encoder, get_decoder, get_texture_decoder, get_texture_renderer
from models.VAE import get_texture_encoder
from models.utils import get2Dkps

parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-checkpoint', required=True, type=int)

parser.add_argument('-res_factor', type=float, default=1.0)
parser.add_argument('-npixels_per_batch', type=int, default=50000)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

exp_dir = './experiments/{}/'.format(args.exp_name)
fname = exp_dir + 'configs.yaml'
with open(fname, 'r') as f:
    print('Loading config file from: ' + fname)
    CFG = yaml.safe_load(f)
CFG_geometry = yaml.safe_load(open(os.path.join('./experiments', CFG['geometry']['exp_name'], 'configs.yaml'), 'r'))
exp_dir_geometry = './experiments/{}/'.format(CFG['geometry']['exp_name'])

print('[CONFIG]', CFG)

radius = CFG['data']['radius']
CAMS = list(range(CFG['data']['ncams']))
nkps = CFG['data']['nkps']

encoder = get_encoder(CFG_geometry)
texture_encoder = get_texture_encoder(CFG)
decoder = get_decoder(CFG_geometry)
texture_decoder = get_texture_decoder(CFG)
texture_renderer = get_texture_renderer(CFG)

CFG['training']['npoints_renderer'] = 10
CFG['training']['npoints_object_renderer'] = 10
print('Set new value for training.batch_size in CONFIG:')
CFG['training']['batch_size'] = 1
print('[INFO]', str(CFG['training']['batch_size']))
mode = 'test'
if CFG['texture']['data'] == "h36m":
    dataset = get_h36m_texture_dataset(
        data_type=CFG['texture']['data'],
        mode=mode,
        sup_distr='uniform',
        cfg=CFG,
        cams=['54138969', '55011271', '58860488', '60457274']
    )
else:
    dataset = get_synthetic_texture_dataset(
        data_type=CFG['texture']['data'],
        mode='test',
        sup_distr='uniform',
        cfg=CFG,
        cams=CAMS,
        exp_name_pos_enc=CFG['geometry']['exp_name']
    )

device = torch.device("cuda")
encoder = encoder.to(device)
texture_encoder = texture_encoder.to(device)
decoder = decoder.to(device)
texture_decoder = texture_decoder.to(device)
texture_renderer = texture_renderer.to(device)
encoder.eval()
texture_encoder.eval()
decoder.eval()
texture_decoder.eval()
texture_renderer.eval()

# load parameters for geometry models
checkpoint_path = exp_dir_geometry + 'checkpoints/checkpoint_epoch_{}.tar'.format(CFG['geometry']['checkpoint'])
print('Load checkpoint from: {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path, map_location=device)
print('Load learned parameters for geometry encoder and geometry decoder')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
if not CFG['renderer']['type'] == 'lfn':
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
# load parameters for texture models
checkpoint_path = exp_dir + 'checkpoints/checkpoint_epoch_{}.tar'.format(args.checkpoint)
print('Load checkpoint from: {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path, map_location=device)
print('Load learned parameters for texture encoder, decoder & renderer')
texture_encoder.load_state_dict(checkpoint['texture_encoder_state_dict'])
if not CFG['renderer']['type'] == 'lfn':
    texture_decoder.load_state_dict(checkpoint['texture_decoder_state_dict'])
texture_renderer.load_state_dict(checkpoint['texture_decoder_impl_state_dict'])

#iterate through dataset and store images
loader = dataset.get_loader(shuffle=False)

rec_dir = exp_dir + 'recs_texture_' + mode + '/'

os.makedirs(rec_dir, exist_ok=True)

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
        mask = [m.to(device).squeeze() for m in data.get('mask')]
        gt_color = [d.to(device).squeeze() for d in data.get('color_maps')]
        gt_masked_color = [d.to(device).squeeze() for d in data.get('masked_color_maps')]
        if CFG['texture']['data'] != "h36m":
            pos_enc = [d.to(device).squeeze() for d in data.get('pos_enc_maps')]
        camera_params = [{k: v.to(device) for (k, v) in zip(c_params.keys(), c_params.values())}
                         for c_params in camera_params_tmp]

        if CFG['texture']['data'] != "h36m":
            texture_encoder_input = [
                torch.cat((masked_color_map.permute(2, 0, 1), pos_enc_map.permute(2, 0, 1)), dim=0)
                for masked_color_map, pos_enc_map in zip(gt_masked_color, pos_enc)
            ]
        else:
            texture_encoder_input = [
                masked_color_map.permute(2, 0, 1)
                for masked_color_map in gt_masked_color
            ]

        for c_idx in CAMS:
            xres = int(mask[c_idx].shape[1] * args.res_factor)
            yres = int(mask[c_idx].shape[0] * args.res_factor)
            xx, yy = np.meshgrid(np.arange(xres), np.arange(yres))
            if not CFG['renderer']['type'] == 'lfn':
                xx = xx / xres
                yy = yy / yres
            img_coords = torch.from_numpy(np.stack([xx, yy], axis=-1)).float().reshape(-1, 2).unsqueeze(0).to(device)

            # pose encoding & decoding
            z = encoder(inp_pos, inp_feats)
            encoding = decoder(z)
            if 'anchors' in encoding:
                encoding['anchors'] *= 2 * radius
            kps_2d_ = get2Dkps(camera_params[c_idx], inp_pos * 2 * radius, mask[c_idx])

            # texture encoding & decoding
            _, _, z_texture = texture_encoder(texture_encoder_input[c_idx].unsqueeze(0))
            texture_encoding = texture_decoder(z_texture, encoding)

            coord_chunks = torch.split(img_coords, args.npixels_per_batch, dim=1)
            color_chunks = []
            for coords in coord_chunks:
                chunk_c = texture_renderer(
                    coords,
                    encoding,
                    texture_encoding,
                    camera_params[c_idx],
                    kps_2d_
                )
                color_chunks.append(chunk_c.squeeze().detach())

            colors = torch.cat(color_chunks, dim=0)
            red = colors[:, 0].reshape(yres, xres)
            green = colors[:, 1].reshape(yres, xres)
            blue = colors[:, 2].reshape(yres, xres)
            rec_color = torch.clamp(torch.stack([red, green, blue], dim=-1)*255, 0, 255)

            rec_color[mask[c_idx] == 0] = 0

            np.save(
                rec_dir + '/rec_color_frame{}_texture{}_camera{}.npy'.format(frame, texture.item(), c_idx),
                rec_color.detach().cpu().numpy().astype(np.uint8)
            )
            np.save(
                rec_dir + '/gt_color_frame{}_texture{}_camera{}.npy'.format(frame, texture.item(), c_idx),
                (gt_color[c_idx].detach().cpu().numpy()*255).astype(np.uint8)
            )
