import argparse
import os
import yaml

import torch
import numpy as np

from data.synthDataset import get_synthetic_texture_dataset_novel_views
from models.NePu import get_encoder, get_decoder, get_texture_decoder, get_texture_renderer
from models.VAE import get_texture_encoder
from models.utils import get2Dkps

parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-checkpoint', required=True, type=int)
parser.add_argument('-novel_views_cfg', type=str)

parser.add_argument('-image_width', type=int, default=1024)
parser.add_argument('-image_height', type=int, default=768)
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
exp_dir_pose = './experiments/{}/'.format(CFG['geometry']['exp_name'])

print('[CONFIG]', CFG)

radius = CFG['data']['radius']
CAMS = [i + 1 for i in list(range(0, 360, 10))]  # test' ('train', 'val')
# CAMS = [2]  # 'track_texture_real_world'
nkps = CFG['data']['nkps']

encoder = get_encoder(CFG_geometry)
texture_encoder = get_texture_encoder(CFG)
decoder = get_decoder(CFG_geometry)
texture_decoder = get_texture_decoder(CFG)
texture_renderer = get_texture_renderer(CFG)

print('Set new value for data.VCA_nneigh in CONFIG:')
CFG['data']['VCA_nneigh'] = nkps
print('[INFO]', str(CFG['data']['VCA_nneigh']))
print('Set new value for training.batch_size in CONFIG:')
CFG['training']['batch_size'] = 1
print('[INFO]', str(CFG['training']['batch_size']))

mode = 'test'  # 'test' ('train', 'val'), 'track_texture_real_world'
dataset = get_synthetic_texture_dataset_novel_views(
    data_type=CFG['texture']['data'],
    mode=mode,
    cfg=CFG,
    cams=CAMS,
    cfg_novel_views=args.novel_views_cfg,
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

# load parameters for pose models
checkpoint_path = exp_dir_pose + 'checkpoints/checkpoint_epoch_{}.tar'.format(CFG['geometry']['checkpoint'])
print('Load checkpoint from: {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path, map_location=device)
print('Load learned parameters for geometry encoder & geometry decoder')
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

# iterate through dataset and store images
loader = dataset.get_loader(shuffle=False)

if mode == 'track_texture_real_world':
    rec_dir = exp_dir + 'recs_texture_real_world/'
else:
    rec_dir = exp_dir + 'recs_texture_novel_views/'

os.makedirs(rec_dir, exist_ok=True)

with torch.no_grad():
    for i, data in enumerate(loader):
        print('Rendering example {}/{}'.format(i + 1, len(loader)))
        frame = data.get('frame').item()
        print('[INFO] frame', frame)
        texture = data.get('texture')
        print('[INFO] texture', texture.item())
        inp_pos = data.get('input_pos').to(device)
        inp_feats = data.get('input_feats').to(device)
        camera_params_tmp = data.get('camera_params')
        random_masked_color_map = [d.to(device).squeeze() for d in data.get('random_masked_color_map')]
        random_pos_enc_map = [d.to(device).squeeze() for d in data.get('random_pos_enc_map')]
        camera_params = [{k: v.to(device) for (k, v) in zip(c_params.keys(), c_params.values())}
                         for c_params in camera_params_tmp]
        empty_img_array = np.empty((args.image_height, args.image_width))

        texture_encoder_input = [
            torch.cat((masked_color_map.permute(2, 0, 1), pos_enc_map.permute(2, 0, 1)), dim=0)
            for masked_color_map, pos_enc_map in zip(random_masked_color_map, random_pos_enc_map)
        ]

        for j, c_idx in enumerate(CAMS):
            xres = int(empty_img_array.shape[1] * args.res_factor)
            yres = int(empty_img_array.shape[0] * args.res_factor)
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
            kps_2d_ = get2Dkps(camera_params[j], inp_pos * 2 * radius, empty_img_array)

            # texture encoding & decoding
            _, _, z_texture = texture_encoder(texture_encoder_input[0].unsqueeze(0))
            texture_encoding = texture_decoder(z_texture, encoding)

            coord_chunks = torch.split(img_coords, args.npixels_per_batch, dim=1)

            # render color map
            color_chunks = []
            for coords in coord_chunks:
                chunk_c = texture_renderer(
                    coords,
                    encoding,
                    texture_encoding,
                    camera_params[j],
                    kps_2d_
                )
                color_chunks.append(chunk_c.squeeze().detach())

            colors = torch.cat(color_chunks, dim=0)
            red = colors[:, 0].reshape(yres, xres)
            green = colors[:, 1].reshape(yres, xres)
            blue = colors[:, 2].reshape(yres, xres)
            rec_color = torch.clamp(torch.stack([red, green, blue], dim=-1) * 255, 0, 255)

            # get reconstructed mask
            rec_mask = np.load(
                os.path.join(
                    './experiments',
                    CFG['geometry']['exp_name'],
                    'recs_novel_views',
                    'rec_mask_frame' + str(frame) + '_camera' + str(c_idx) + '.npy'
                )
            )

            rec_color[rec_mask <= 0.5] = 0

            np.save(
                rec_dir + '/rec_color_frame{}_texture{}_camera{}.npy'.format(frame, texture.item(), c_idx),
                rec_color.detach().cpu().numpy().astype(np.uint8)
            )
