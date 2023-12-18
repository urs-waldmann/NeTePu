import argparse
import os
import yaml

import torch
import numpy as np

from data.synthDataset import get_synthetic_dataset_novel_views
from models.NePu import get_encoder, get_decoder, get_renderer
from models.utils import get2Dkps

parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-checkpoint', required=True, type=int)
parser.add_argument('-novel_views_cfg', required=True, type=str)

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

print('[CONFIG]', CFG)

radius = CFG['data']['radius']
CAMS = [i + 1 for i in list(range(0, 360, 10))]  # test' ('train', 'val')
# CAMS = [1, 3, 6, 10, 12, 13, 15, 19, 20, 22]  # 'track_texture_novel_views'
# CAMS = [6]  # 'track_texture_real_world'
nkps = CFG['data']['nkps']

encoder = get_encoder(CFG)
decoder = get_decoder(CFG)
renderer = get_renderer(CFG)

print('Set new value for data.VCA_nneigh in CONFIG:')
CFG['data']['VCA_nneigh'] = nkps
print('[INFO]', str(CFG['data']['VCA_nneigh']))
print('Set new value for training.batch_size in CONFIG:')
CFG['training']['batch_size'] = 1
print('[INFO]', str(CFG['training']['batch_size']))

mode = 'test'  # 'test' ('train', 'val'), 'track_texture_novel_views', 'track_texture_real_world'
dataset = get_synthetic_dataset_novel_views(
    data_type=CFG['geometry']['data'],
    mode=mode,
    cfg=CFG,
    cams=CAMS,
    cfg_novel_views=args.novel_views_cfg
)

device = torch.device("cuda")
encoder = encoder.to(device)
decoder = decoder.to(device)
renderer = renderer.to(device)
encoder.eval()
decoder.eval()
renderer.eval()

# load params
checkpoint_path = exp_dir + 'checkpoints/checkpoint_epoch_{}.tar'.format(args.checkpoint)
checkpoint = torch.load(checkpoint_path, map_location=device)
print('Loaded checkpoint from: {}'.format(checkpoint_path))
encoder.load_state_dict(checkpoint['encoder_state_dict'])
if not CFG['renderer']['type'] == 'lfn':
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
renderer.load_state_dict(checkpoint['decoder_impl_state_dict'])

# iterate through dataset and store images
loader = dataset.get_loader(shuffle=False)

if mode == 'track_texture_novel_views':
    rec_dir = exp_dir + 'recs_track_texture_novel_views/'
elif mode == 'track_texture_real_world':
    rec_dir = exp_dir + 'recs_track_texture_real_world/'
else:
    rec_dir = exp_dir + 'recs_novel_views/'

os.makedirs(rec_dir, exist_ok=True)

with torch.no_grad():
    for i, data in enumerate(loader):
        print('Rendering example {}/{}'.format(i+1, len(loader)))
        frame = data.get('frame').item()
        print('[INFO] frame', frame)
        inp_pos = data.get('input_pos').to(device)
        inp_feats = data.get('input_feats').to(device)
        camera_params_tmp = data.get('camera_params')
        camera_params = [{k: v.to(device) for (k, v) in zip(c_params.keys(), c_params.values())}
                         for c_params in camera_params_tmp]
        empty_img_array = np.empty((args.image_height, args.image_width))

        for j, c_idx in enumerate(CAMS):
            xres = int(empty_img_array.shape[1] * args.res_factor)
            yres = int(empty_img_array.shape[0] * args.res_factor)
            xx, yy = np.meshgrid(np.arange(xres), np.arange(yres))
            if not CFG['renderer']['type'] == 'lfn':
                xx = xx / xres
                yy = yy / yres
            img_coords = torch.from_numpy(np.stack([xx, yy], axis=-1)).float().reshape(-1, 2).unsqueeze(0).to(device)

            z = encoder(inp_pos, inp_feats)
            encoding = decoder(z)
            if 'anchors' in encoding:
                encoding['anchors'] *= 2*radius
            kps_2d_ = get2Dkps(camera_params[j], inp_pos*2*radius, empty_img_array)

            coord_chunks = torch.split(img_coords, args.npixels_per_batch, dim=1)
            logit_chunks = []
            depth_chunks = []
            pos_enc_chunks = []
            for coords in coord_chunks:
                chunk, chunk_d, chunk_pe = renderer(
                    coords,
                    encoding,
                    camera_params[j],
                    kps_2d_
                )
                logit_chunks.append(chunk.squeeze().detach())
                depth_chunks.append(chunk_d.squeeze().detach())
                pos_enc_chunks.append(chunk_pe.squeeze().detach())

            logits = torch.cat(logit_chunks, dim=0)
            dephts = torch.cat(depth_chunks, dim=0)
            pos_enc = torch.cat(pos_enc_chunks, dim=0)
            rec_img = torch.sigmoid(logits.reshape(yres, xres))
            rec_depth = dephts.reshape(yres, xres)
            red_pos_enc = pos_enc[:, 0].reshape(yres, xres)
            green_pos_enc = pos_enc[:, 1].reshape(yres, xres)
            blue_pos_enc = pos_enc[:, 2].reshape(yres, xres)
            rec_pos_enc = torch.clamp(torch.stack([red_pos_enc, green_pos_enc, blue_pos_enc], dim=-1) * 255, 0, 255)

            keyPos_hom = torch.cat(
                [
                    inp_pos*2*radius,
                    torch.ones([inp_pos.shape[0], inp_pos.shape[1], 1], device=inp_pos.device, dtype=torch.float)
                ],
                dim=2).permute(0, 2, 1)
            tmp = torch.bmm(camera_params[j]['extrinsics'], keyPos_hom).permute(0, 2, 1)
            d = tmp[:, :, -1].squeeze()

            threshold = 0.5
            rec_depth[rec_img > threshold] = rec_depth[rec_img > threshold] * max(d-torch.min(d).item())\
                                             + torch.min(d).item()

            rec_depth[rec_img <= threshold] = -1.0
            rec_pos_enc[rec_img <= threshold] = 0

            np.save(
                rec_dir + '/rec_depth_frame{}_camera{}.npy'.format(frame, c_idx),
                rec_depth.detach().cpu().numpy()
            )
            np.save(
                rec_dir + '/rec_pos_enc_frame{}_camera{}.npy'.format(frame, c_idx),
                rec_pos_enc.detach().cpu().numpy().astype(np.uint8)
            )
            np.save(
                rec_dir + '/rec_mask_frame{}_camera{}.npy'.format(frame, c_idx),
                rec_img.detach().cpu().numpy()
            )
