import argparse
import os
import yaml
import cv2
import torch
import numpy as np

from data.synthDataset import get_synthetic_dataset
from data.h36m import get_h36m_dataset
from models.NePu import get_encoder, get_decoder, get_renderer
from models.utils import get2Dkps

parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-checkpoint', required=True, type=int)

parser.add_argument('-res_factor', type=float, default=1.0)
parser.add_argument('-npixels_per_batch', type=int, default=50000)
parser.add_argument('-plot_only_kpts', type=bool, default=False)

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
CAMS = list(range(CFG['data']['ncams']))
nkps = CFG['data']['nkps']

encoder = get_encoder(CFG)
decoder = get_decoder(CFG)
renderer = get_renderer(CFG)

CFG['training']['npoints_renderer'] = 10
CFG['training']['npoints_object_renderer'] = 10
print('Set new value for training.batch_size in CONFIG:')
CFG['training']['batch_size'] = 1
print('[INFO]', str(CFG['training']['batch_size']))
mode = 'test'
if CFG['geometry']['data'] == "h36m":
    dataset = get_h36m_dataset(
        data_type=CFG['geometry']['data'],
        mode=mode,
        sup_distr='uniform',
        cfg=CFG,
        cams=['54138969', '55011271', '58860488', '60457274']
    )
else:
    dataset = get_synthetic_dataset(
        data_type=CFG['geometry']['data'],
        mode=mode,
        sup_distr='uniform',
        cfg=CFG,
        cams=CAMS
    )

device = torch.device("cuda")
encoder = encoder.to(device)
decoder = decoder.to(device)
renderer = renderer.to(device)
encoder.eval()
decoder.eval()
renderer.eval()

#load params
checkpoint_path = exp_dir + 'checkpoints/checkpoint_epoch_{}.tar'.format(args.checkpoint)
print('Load checkpoint from: {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path, map_location=device)
print('Load learned parameters for encoder, decoder & renderer')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
if not CFG['renderer']['type'] == 'lfn':
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
renderer.load_state_dict(checkpoint['decoder_impl_state_dict'])

#iterate through dataset and store images
loader = dataset.get_loader(shuffle=False)

rec_dir = exp_dir + 'recs_' + mode + '/'

os.makedirs(rec_dir, exist_ok=True)

with torch.no_grad():
    for i, data in enumerate(loader):
        print('Rendering example {}/{}'.format(i+1, len(loader)))
        frame = data.get('frame').item()
        print('[INFO] frame', frame)
        subject = data.get('subject').item()
        print('[INFO] subject', subject)
        inp_pos = data.get('input_pos').to(device)
        inp_feats = data.get('input_feats').to(device)
        camera_params_tmp = data.get('camera_params')
        mask = [m.to(device).squeeze() for m in data.get('mask')]
        if CFG['geometry']['data'] != 'h36m':
            gt_depth = [d.to(device).squeeze() for d in data.get('depth_maps')]
            gt_pos_enc = [d.to(device).squeeze() for d in data.get('pos_enc_maps')]
        camera_params = [{k: v.to(device) for (k, v) in zip(c_params.keys(), c_params.values())}
                         for c_params in camera_params_tmp]

        for c_idx in CAMS:
            xres = int(mask[c_idx].shape[1] * args.res_factor)
            yres = int(mask[c_idx].shape[0] * args.res_factor)
            xx, yy = np.meshgrid(np.arange(xres), np.arange(yres))
            if not CFG['renderer']['type'] == 'lfn':
                xx = xx / xres
                yy = yy / yres
            img_coords = torch.from_numpy(np.stack([xx, yy], axis=-1)).float().reshape(-1, 2).unsqueeze(0).to(device)

            z = encoder(inp_pos, inp_feats)
            encoding = decoder(z)
            if 'anchors' in encoding:
                encoding['anchors'] *= 2*radius
            kps_2d_ = get2Dkps(camera_params[c_idx], inp_pos*2*radius, mask[c_idx])

            coord_chunks = torch.split(img_coords, args.npixels_per_batch, dim=1)
            logit_chunks = []
            depth_chunks = []
            pos_enc_chunks = []
            for coords in coord_chunks:
                chunk, chunk_d, chunk_pe = renderer(
                    coords,
                    encoding,
                    camera_params[c_idx],
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
            tmp = torch.bmm(camera_params[c_idx]['extrinsics'], keyPos_hom).permute(0, 2, 1)
            d = tmp[:, :, -1].squeeze()

            threshold = 0.5
            rec_depth[rec_img > threshold] = rec_depth[rec_img > threshold] * max(d-torch.min(d).item())\
                                             + torch.min(d).item()
            if CFG['geometry']['data'] != 'h36m':
                gt_depth[c_idx][mask[c_idx] > threshold] = gt_depth[c_idx][mask[c_idx] > threshold]\
                                                            * max(d-torch.min(d).item()) + torch.min(d).item()

            rec_depth[rec_img <= threshold] = -1.0
            if CFG['geometry']['data'] != 'h36m':
                gt_depth[c_idx][mask[c_idx] <= threshold] = -1.0
            rec_pos_enc[rec_img <= threshold] = 0

            # plot keypoints on mask
            if args.plot_only_kpts:
                # predicted mask
                rec_mask_vis = np.stack(
                    [
                        rec_img.detach().cpu().numpy(),
                        rec_img.detach().cpu().numpy(),
                        rec_img.detach().cpu().numpy()
                    ],
                    axis=2
                ) * 255
                # gt mask
                mask_vis = np.stack(
                    [
                        mask[c_idx].detach().cpu().numpy(),
                        mask[c_idx].detach().cpu().numpy(),
                        mask[c_idx].detach().cpu().numpy()
                    ],
                    axis=2
                ) * 255
                # scale kpts to frame size
                # gt kpts
                kps_2d_[0, :, 0] *= mask[c_idx].shape[1]
                kps_2d_[0, :, 1] *= mask[c_idx].shape[0]
                # predicted kpts
                pred_kps_2d_ = get2Dkps(camera_params[c_idx], encoding['anchors'], mask[c_idx])
                pred_kps_2d_[0, :, 0] *= mask[c_idx].shape[1]
                pred_kps_2d_[0, :, 1] *= mask[c_idx].shape[0]
                # draw gt kpts on gt mask
                for p in kps_2d_[0, :, :]:
                    point = (int(p[0]), int(p[1]))
                    cv2.drawMarker(
                        mask_vis,
                        point,
                        color=(255, 0, 0),  # blue (BGR)
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=2
                    )
                # draw predicted kpts on gt mask
                for p in pred_kps_2d_[0, :, :]:
                    point = (int(p[0]), int(p[1]))
                    cv2.drawMarker(
                        mask_vis,
                        point,
                        color=(0, 0, 255),  # red (BGR)
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=1
                    )
                # draw gt kpts on predicted mask
                for p in kps_2d_[0, :, :]:
                    point = (int(p[0]), int(p[1]))
                    cv2.drawMarker(
                        rec_mask_vis,
                        point,
                        color=(255, 0, 0),  # blue (BGR)
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=2
                    )
                # draw predicted kpts on predicted mask
                for p in pred_kps_2d_[0, :, :]:
                    point = (int(p[0]), int(p[1]))
                    cv2.drawMarker(
                        rec_mask_vis,
                        point,
                        color=(0, 0, 255),  # red (BGR)
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=1
                    )
                # show image
                window_name = 'camera ' + str(c_idx) + ', subject ' + str(subject) + ', frame ' + str(frame) \
                              + ', left: gt mask, right: predicted mask, blue: gt, red: predictions'
                final = np.concatenate((mask_vis, rec_mask_vis), axis=1)
                cv2.imshow(window_name, final)
                # show until any key is pressed
                key = cv2.waitKey(0) & 0xFF
                # destroy all windows
                cv2.destroyAllWindows()
            else:
                # save files
                if CFG['geometry']['data'] != 'h36m':
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
                    np.save(
                        rec_dir + '/gt_depth_frame{}_camera{}.npy'.format(frame, c_idx),
                        gt_depth[c_idx].detach().cpu().numpy()
                    )
                    np.save(
                        rec_dir + '/gt_pos_enc_frame{}_camera{}.npy'.format(frame, c_idx),
                        (gt_pos_enc[c_idx].detach().cpu().numpy() * 255).astype(np.uint8)
                    )
                    np.save(
                        rec_dir + '/gt_mask_frame{}_camera{}.npy'.format(frame, c_idx),
                        mask[c_idx].detach().cpu().numpy()
                    )
                else:
                    np.save(
                        rec_dir + '/rec_mask_frame{}_subject{}_camera{}.npy'.format(frame, subject, c_idx),
                        rec_img.detach().cpu().numpy()
                    )
                    np.save(
                        rec_dir + '/gt_mask_frame{}_subject{}_camera{}.npy'.format(frame, subject, c_idx),
                        mask[c_idx].detach().cpu().numpy()
                    )
