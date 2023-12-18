from math import floor
import numpy as np
import numpy.random
from torch.utils.data import Dataset
import torch
import os
import imageio.v2 as imageio
from scipy.signal import convolve2d
import cdflib
import json
import random

from .utils import proj


class H36MDataset(Dataset):
    def __init__(self, species, mode, n_supervision_points, n_supervision_points_object, supervision_distr, batch_size,
                 nkps, num_datapoints, radius, std_near, std_far,
                 cams=None, normalize_pixel_coords=True, noise_aug=None, cam_aug=True):

        self.type = species
        self.mode = mode
        self.nkps = nkps
        self.cam_ids = ['54138969', '55011271', '58860488', '60457274']
        self.subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

        self.noise_aug = noise_aug
        self.cam_aug = cam_aug
        self.std_near = std_near
        self.std_far = std_far

        self.N = num_datapoints

        self.radius = radius
        np.random.seed(0)

        # compute splits
        n_splits = int(num_datapoints/10)
        n_train = floor(n_splits*0.7)
        n_val = floor(n_splits*0.1)
        n_test = n_splits - n_train - n_val

        A = np.arange(num_datapoints)

        chunks = np.stack(np.split(A, n_splits), axis=0)

        perm = np.random.permutation(n_splits)

        chunks = chunks[perm, :]

        chunks_train = chunks[:n_train, :]
        chunks_val = chunks[n_train:n_train+n_val, :]
        chunks_test = chunks[-n_test:, :]

        train_steps = np.reshape(chunks_train, [-1])
        val_steps = np.reshape(chunks_val, [-1])
        test_steps = np.reshape(chunks_test, [-1])

        steps = {'train': train_steps, 'val': val_steps, 'test': test_steps}
        self.steps = steps[mode]

        self.path = './data/h36m/'
        self.batch_size = batch_size
        self.n_supervision_points = n_supervision_points
        self.n_supervision_points_object = n_supervision_points_object
        self.supervision_distr = supervision_distr
        self.NORM = normalize_pixel_coords

        self.cams = cams

        # read json file with camera parameters
        with open(os.path.join(self.path, 'human36m-camera-parameters/camera-parameters.json'), 'r') as f:
            cam_param = json.load(f)

        self.keyPos3d = []
        self.cameraExtrinsics = []
        for subject in self.subjects:
            # 3D keypoints path
            path_x = os.path.join(
                self.path,
                'data/Poses_D3_Positions_Posing/',
                subject,
                'MyPoseFeatures/D3_Positions/Posing.cdf'
            )
            # read 3D keypoints
            X = np.squeeze(cdflib.CDF(path_x).varget(variable='Pose'))[:self.N, :]
            X = X.reshape((X.shape[0], -1, 3))
            self.keyPos3d.append(X)

            # obtain extrinsic parameters
            for cam_id in self.cam_ids:
                R = cam_param['extrinsics'][subject][cam_id]['R']
                t = cam_param['extrinsics'][subject][cam_id]['t']
                RT = np.hstack([R, t])
                self.cameraExtrinsics.append(RT)

        self.keyPos3d = np.concatenate(self.keyPos3d, axis=0)
        print('[INFO] Shape of', species, 'keypoint array (#frames, #kpts, #dim):', self.keyPos3d.shape)

        # obtain intrinsic parameters
        self.cameraIntrinsics = []
        for cam_id in self.cam_ids:
            K = cam_param['intrinsics'][cam_id]['calibration_matrix']
            K = np.stack(K, axis=0)
            self.cameraIntrinsics.append(K)

        # we do NOT augment cameras for human36m
        # thus the model does NOT learn rotation equivariance
        print('[INFO] No rotation augmentation for {}!'.format(self.type))

    def __len__(self):
        # 7 human subjects
        return len(self.steps) * 7

    def sample_points_concentrated(self, n_samps, idx, mask, std):
        res = mask.shape
        samps = np.random.randint(0, len(idx[0]), n_samps)
        samps_x = idx[0][samps]
        samps_y = idx[1][samps]

        samps_y01 = samps_y/res[1] + np.random.randn(samps_y.shape[0])*std
        samps_x01 = samps_x/res[0] + np.random.randn(samps_x.shape[0])*std
        samps_x = np.clip((samps_x01*res[0]).astype(np.int32), 0, res[0]-1)
        samps_y = np.clip((samps_y01*res[1]).astype(np.int32), 0, res[1]-1)

        if not self.NORM:
            samps_y01 = samps_y
            samps_x01 = samps_x

        return mask[samps_x, samps_y], np.stack([samps_y01, samps_x01], axis=1)

    def sample_points_uniform(self, n_samps, idx, mask):
        res = mask.shape
        samps = np.random.randint(0, len(idx[0]), n_samps)
        samps_x = idx[0][samps]
        samps_y = idx[1][samps]

        if self.NORM:
            samps_pos = np.stack([samps_y/res[1], samps_x/res[0]], axis=1)
        else:
            samps_pos = np.stack([samps_y, samps_x], axis=1)

        return mask[samps_x, samps_y],\
            samps_pos

    def __getitem__(self, idx):
        # assign idx to subject (numbering is different from h36m convention)
        texture = floor(idx / len(self.steps))
        # get subject with h36m convention
        subject = int(self.subjects[texture][1:])

        idx -= texture * len(self.steps)
        idx = self.steps[idx]

        # TODO: update once dataset is updated
        keyPos3d = self.keyPos3d[(texture * self.N) + idx, :, :]

        onehot = np.eye(self.nkps)
        masks = []
        sup_pos = []
        sup_gt = []
        sup_pos_foreground = []
        cams = []
        kps_2ds = []

        cams_curr = self.cams
        if cams_curr is None:
            cams_curr = [random.choice(self.cam_ids)]

        keyPos3d_unperturbed = keyPos3d
        if self.noise_aug is not None:
            keyPos3d = keyPos3d + np.random.randn(*list(keyPos3d.shape)) * self.noise_aug

        for camera in cams_curr:
            path = os.path.join(
                self.path,
                'data/h36m',
                str(self.subjects[texture]),
                'Posing/mask_cihp',
                camera
            )

            camera_idx = self.cam_ids.index(camera)

            intrinsics = self.cameraIntrinsics[camera_idx]
            extrinsics = self.cameraExtrinsics[camera_idx + texture * len(self.cam_ids)]

            mask = imageio.imread(path + '/{:06d}.png'.format(idx))
            mask[mask > 0] = 255

            camera_params = {'intrinsics': intrinsics, 'extrinsics': extrinsics}
            kps_2d = proj(camera_params, keyPos3d)

            if self.NORM:
                kps_2d[:, 0] /= mask.shape[1]
                kps_2d[:, 1] /= mask.shape[0]

            mask[mask == 255] = 1
            n_far = int(self.n_supervision_points/2)
            n_near = self.n_supervision_points - n_far
            # do uniform sampling for silhouette supervision if
            # a) it is specified, which is the case for inverse rendering
            # b) LFN baseline during training (for this baseline concentrated is too complex)
            if self.supervision_distr == 'uniform' or (self.supervision_distr == 'training' and not self.NORM):
                background_idx = np.where(mask == 0)
                foreground_idx = np.where(mask == 1)
                sup_gt_fore, sup_pos_fore = self.sample_points_uniform(
                    n_far,
                    foreground_idx,
                    mask
                )
                sup_gt_back, sup_pos_back = self.sample_points_uniform(
                    n_near, background_idx, mask
                )
                sup_pos_ = np.concatenate([sup_pos_fore, sup_pos_back], axis=0)
                sup_gt_ = np.concatenate([sup_gt_fore, sup_gt_back], axis=0)
            elif self.supervision_distr == 'training':
                # obtain boundary pixels
                filtered = convolve2d(mask.astype(np.int32), np.ones([3, 3], dtype=np.int32), mode='same')
                interesting_bool = np.logical_not(np.logical_or(filtered == 0, filtered == 255*9))
                interesting_pos = np.where(interesting_bool)
                sup_gt_far, sup_pos_far = self.sample_points_concentrated(n_far, interesting_pos, mask, self.std_far)
                sup_gt_near, sup_pos_near = self.sample_points_concentrated(
                    n_near,
                    interesting_pos,
                    mask,
                    self.std_near
                )
                sup_pos_ = np.concatenate([sup_pos_near, sup_pos_far], axis=0)
                sup_gt_ = np.concatenate([sup_gt_near, sup_gt_far], axis=0)
                _, sup_pos_fore = self.sample_points_uniform(
                    self.n_supervision_points_object,
                    np.where(mask == 1),
                    mask
                )

            cams.append({'intrinsics': torch.from_numpy(intrinsics).float(),
                         'extrinsics': torch.from_numpy(extrinsics).float()})
            masks.append(torch.from_numpy(mask))
            sup_pos.append(torch.from_numpy(sup_pos_).float())
            sup_gt.append(torch.from_numpy(sup_gt_).float())
            sup_pos_foreground.append(torch.from_numpy(sup_pos_fore).float())
            kps_2ds.append(torch.from_numpy(kps_2d).float())

        return {'input_pos': torch.from_numpy(keyPos3d_unperturbed).float() / (2*self.radius),
                'input_pos_pert': torch.from_numpy(keyPos3d).float() / (2*self.radius),
                'radius': torch.tensor(self.radius).float(),
                'input_feats': torch.from_numpy(onehot).float(),
                'supervision_pos': sup_pos,
                'supervision_gt': sup_gt,
                'supervision_pos_fore': sup_pos_foreground,
                'camera_params': cams,
                'mask': masks,
                'kps_2ds': kps_2ds,
                'frame': idx,
                'subject': subject}

    def get_loader(self, shuffle=True):
        #random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=10, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)


class H36MTextureDataset(Dataset):
    def __init__(self, species, mode, n_supervision_points, n_supervision_points_object, supervision_distr, batch_size,
                 nkps, num_datapoints, radius, std_near, std_far,
                 cams=None, normalize_pixel_coords=True, noise_aug=None, cam_aug=True):

        self.type = species
        self.mode = mode
        self.nkps = nkps
        self.cam_ids = ['54138969', '55011271', '58860488', '60457274']
        self.subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

        self.noise_aug = noise_aug
        self.cam_aug = cam_aug
        self.std_near = std_near
        self.std_far = std_far

        self.N = num_datapoints

        self.radius = radius
        np.random.seed(0)

        # compute splits
        n_splits = int(num_datapoints / 10)
        n_train = floor(n_splits * 0.7)
        n_val = floor(n_splits * 0.1)
        n_test = n_splits - n_train - n_val

        A = np.arange(num_datapoints)

        track_texture_steps = A

        chunks = np.stack(np.split(A, n_splits), axis=0)

        perm = np.random.permutation(n_splits)

        chunks = chunks[perm, :]

        chunks_train = chunks[:n_train, :]
        chunks_val = chunks[n_train:n_train + n_val, :]
        chunks_test = chunks[-n_test:, :]

        train_steps = np.reshape(chunks_train, [-1])
        val_steps = np.reshape(chunks_val, [-1])
        test_steps = np.reshape(chunks_test, [-1])

        steps = {
            'train': train_steps,
            'val': val_steps,
            'test': test_steps,
            'track_texture': track_texture_steps
        }
        self.steps = steps[mode]

        paths = {'h36m': ['./data/h36m/data/h36m/S1/Posing/',
                          './data/h36m/data/h36m/S5/Posing/',
                          './data/h36m/data/h36m/S6/Posing/',
                          './data/h36m/data/h36m/S7/Posing/',
                          './data/h36m/data/h36m/S8/Posing/',
                          './data/h36m/data/h36m/S9/Posing/',
                          './data/h36m/data/h36m/S11/Posing/'
                          ]
                 }
        self.path = paths[species]
        print('[INFO] Number of textures: {}'.format(len(self.path)))
        self.batch_size = batch_size
        self.n_supervision_points = n_supervision_points
        self.n_supervision_points_object = n_supervision_points_object
        self.supervision_distr = supervision_distr
        self.NORM = normalize_pixel_coords

        self.cams = cams

        # read json file with camera parameters
        with open('./data/h36m/human36m-camera-parameters/camera-parameters.json', 'r') as f:
            cam_param = json.load(f)

        self.keyPos3d = []
        self.cameraExtrinsics = []
        for subject in self.subjects:
            # 3D keypoints path
            path_x = os.path.join(
                './data/h36m/data/Poses_D3_Positions_Posing/',
                subject,
                'MyPoseFeatures/D3_Positions/Posing.cdf'
            )
            # read 3D keypoints
            X = np.squeeze(cdflib.CDF(path_x).varget(variable='Pose'))[:self.N, :]
            X = X.reshape((X.shape[0], -1, 3))
            self.keyPos3d.append(X)

            # obtain extrinsic parameters
            for cam_id in self.cam_ids:
                R = cam_param['extrinsics'][subject][cam_id]['R']
                t = cam_param['extrinsics'][subject][cam_id]['t']
                RT = np.hstack([R, t])
                self.cameraExtrinsics.append(RT)

        self.keyPos3d = np.concatenate(self.keyPos3d, axis=0)
        print('[INFO] Shape of', species, 'keypoint array (#frames, #kpts, #dim):', self.keyPos3d.shape)

        # obtain intrinsic parameters
        self.cameraIntrinsics = []
        for cam_id in self.cam_ids:
            K = cam_param['intrinsics'][cam_id]['calibration_matrix']
            K = np.stack(K, axis=0)
            self.cameraIntrinsics.append(K)

    def __len__(self):
        return len(self.steps) * len(self.path)

    def sample_points_uniform(self, n_samps, idx, mask, color):
        res = mask.shape
        samps = np.random.randint(0, len(idx[0]), n_samps)
        samps_x = idx[0][samps]
        samps_y = idx[1][samps]

        if self.NORM:
            samps_pos = np.stack([samps_y / res[1], samps_x / res[0]], axis=1)
        else:
            samps_pos = np.stack([samps_y, samps_x], axis=1)

        return color[samps_x, samps_y, :], samps_pos

    def __getitem__(self, idx):
        # assign idx to subject (numbering is different from h36m convention)
        texture = floor(idx / len(self.steps))
        # get subject with h36m convention
        subject = int(self.subjects[texture][1:])

        idx -= texture * len(self.steps)
        idx = self.steps[idx]

        # TODO: update once dataset is updated
        keyPos3d = self.keyPos3d[(texture * self.N) + idx, :, :]

        onehot = np.eye(self.nkps)
        masks = []
        color_maps = []
        masked_color_maps = []
        sup_col = []
        sup_pos_foreground = []
        cams = []
        kps_2ds = []

        cams_curr = self.cams
        if cams_curr is None:
            cams_curr = [random.choice(self.cam_ids)]

        keyPos3d_unperturbed = keyPos3d
        if self.noise_aug is not None:
            keyPos3d = keyPos3d + np.random.randn(*list(keyPos3d.shape)) * self.noise_aug

        for camera in cams_curr:
            path = os.path.join(
                self.path[texture],
                camera
            )
            path_masks = os.path.join(
                self.path[texture],
                'mask_cihp',
                camera
            )

            camera_idx = self.cam_ids.index(camera)

            intrinsics = self.cameraIntrinsics[camera_idx]
            extrinsics = self.cameraExtrinsics[camera_idx + texture * len(self.cam_ids)]

            # color
            color = imageio.imread(path + '/{:06d}.jpg'.format(idx)).astype(np.float32)
            # mask
            mask = imageio.imread(path_masks + '/{:06d}.png'.format(idx))
            mask[mask > 0] = 255
            color = color / 255
            # create masked color map
            masked_color = color.copy()
            masked_color[mask < 255] = 0

            camera_params = {'intrinsics': intrinsics, 'extrinsics': extrinsics}
            kps_2d = proj(camera_params, keyPos3d)

            if self.NORM:
                kps_2d[:, 0] /= mask.shape[1]
                kps_2d[:, 1] /= mask.shape[0]

            mask[mask == 255] = 1
            n_far = int(self.n_supervision_points / 2)
            # do uniform sampling for silhouette supervision if
            # a) it is specified, which is the case for inverse rendering
            # b) LFN baseline during training (for this baseline concentrated is too complex)
            if self.supervision_distr == 'uniform' or (self.supervision_distr == 'training' and not self.NORM):
                foreground_idx = np.where(mask == 1)
                color_gt, sup_pos_fore = self.sample_points_uniform(
                    n_far,
                    foreground_idx,
                    mask,
                    color
                )
            elif self.supervision_distr == 'training':
                # TODO add parameters for n_sup_points for color!
                color_gt, sup_pos_fore = self.sample_points_uniform(
                    self.n_supervision_points_object,
                    np.where(mask == 1),
                    mask,
                    color
                )

            cams.append({'intrinsics': torch.from_numpy(intrinsics).float(),
                         'extrinsics': torch.from_numpy(extrinsics).float()})
            masks.append(torch.from_numpy(mask))
            sup_col.append(torch.from_numpy(color_gt).float())
            sup_pos_foreground.append(torch.from_numpy(sup_pos_fore).float())
            color_maps.append(torch.from_numpy(color).float())
            masked_color_maps.append(torch.from_numpy(masked_color).float())
            kps_2ds.append(torch.from_numpy(kps_2d).float())

        return {'input_pos': torch.from_numpy(keyPos3d_unperturbed).float() / (2*self.radius),
                'input_pos_pert': torch.from_numpy(keyPos3d).float() / (2 * self.radius),
                'radius': torch.tensor(self.radius).float(),
                'input_feats': torch.from_numpy(onehot).float(),
                'supervision_gt_color': sup_col,
                'supervision_pos_fore': sup_pos_foreground,
                'mask': masks,
                'camera_params': cams,
                'color_maps': color_maps,
                'masked_color_maps': masked_color_maps,
                'texture': texture,
                'subject': subject,
                'frame': idx,
                'kps_2ds': kps_2ds}

    def get_loader(self, shuffle=True):
        #random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=10, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)


def get_h36m_dataset(data_type, mode, sup_distr, cfg, cams=None):
    return H36MDataset(species=data_type,
                       mode=mode,
                        n_supervision_points=cfg['training']['npoints_renderer'],
                        n_supervision_points_object=cfg['training']['npoints_object_renderer'],
                        supervision_distr=sup_distr,
                        batch_size=cfg['training']['batch_size'],
                        nkps=cfg['data']['nkps'],
                        num_datapoints=cfg['data']['num_datapoints'],
                        radius=cfg['data']['radius'],
                        cams=cams,
                        normalize_pixel_coords=True,
                        noise_aug=cfg['data']['noise_aug_kpts'],
                        cam_aug=True,
                        std_near=cfg['data']['std_near'],
                        std_far=cfg['data']['std_far']
                       )


def get_h36m_texture_dataset(data_type, mode, sup_distr, cfg, cams=None, exp_name_pos_enc=None):
    return H36MTextureDataset(species=data_type,
                              mode=mode,
                              n_supervision_points=cfg['training']['npoints_renderer'],
                              n_supervision_points_object=cfg['training']['npoints_object_renderer'],
                              supervision_distr=sup_distr,
                              batch_size=cfg['training']['batch_size'],
                              nkps=cfg['data']['nkps'],
                              num_datapoints=cfg['data']['num_datapoints'],
                              radius=cfg['data']['radius'],
                              cams=cams,
                              normalize_pixel_coords=True,
                              noise_aug=cfg['data']['noise_aug_kpts'],
                              cam_aug=True,
                              std_near=cfg['data']['std_near'],
                              std_far=cfg['data']['std_far']
                              )
