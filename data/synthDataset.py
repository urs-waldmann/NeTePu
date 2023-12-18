import math
from math import floor
import numpy as np
import numpy.random
from torch.utils.data import Dataset
import torch
import os
import configparser
import cv2
import imageio.v2 as imageio
from scipy.signal import convolve2d
from scipy.spatial.transform import Rotation as rotation
import sys

from .utils import get_intrinsic_params, get_extrinsic_params, proj


class SyntheticDataset(Dataset):
    def __init__(self, species, mode, n_supervision_points, n_supervision_points_object, supervision_distr, batch_size,
                 nkps, num_datapoints, radius, ncams, std_near, std_far,
                 cams=None, normalize_pixel_coords=True, noise_aug=None, cam_aug=True):

        self.type = species
        self.mode = mode
        self.nkps = nkps
        self.ncams = ncams

        self.noise_aug = noise_aug
        self.cam_aug = cam_aug
        self.std_near = std_near
        self.std_far = std_far

        self.N = num_datapoints

        self.radius = radius
        np.random.seed(0)

        # compute splits
        if species == 'humans':
            perm = np.random.permutation(num_datapoints)
            time_steps = np.arange(num_datapoints) + 1
            perm_steps = time_steps[perm]
            n_train = int(num_datapoints*0.70)
            n_val = int(num_datapoints*0.1)
            n_test = num_datapoints - n_train - n_val
            train_steps = perm_steps[:n_train]
            val_steps = perm_steps[n_train:n_train+n_val]
            test_steps = perm_steps[-n_test:]
            print(len(test_steps))
        else:
            n_splits = int(num_datapoints/10)
            n_train = floor(n_splits*0.7)
            n_val = floor(n_splits*0.1)
            n_test = n_splits - n_train - n_val

            A = np.arange(num_datapoints)
            # TODO: update once dataset is updated!
            if species != 'pigeons':
                A += 1

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

        paths = {'pigeons': './data/multiview_pigeon/2021.11.26 - pigeonanimation 56cams/',
                 'humans': './data/multiview_smplx/male/pose/',
                 'cows': './data/multiview_cow/cow_animation_alltextures.blend/Cow_Holstein.tif/',
                 'giraffes': './data/multiview_giraffe/giraffe_animation.blend/'
                 }
        self.path = paths[species]
        self.batch_size = batch_size
        self.n_supervision_points = n_supervision_points
        self.n_supervision_points_object = n_supervision_points_object
        self.supervision_distr = supervision_distr
        self.NORM = normalize_pixel_coords

        self.keyPos3d = np.genfromtxt(self.path + 'keypoints.csv', delimiter=',')[1:, 1:]
        #TODO n_cams is different for novel views!!!
        self.n_cams = 24
        # for pigeons remove the unfolded wing keypoints and use 56 cameras
        if species == 'pigeons':
            self.keyPos3d = np.concatenate([self.keyPos3d[:, :7*3], self.keyPos3d[:, 9*3:]], axis=1)
            self.n_cams = 56

        self.keyPos3d = self.keyPos3d.reshape((self.keyPos3d.shape[0], -1, 3))
        print('[INFO] Shape of', species, 'keypoint array (#frames, #kpts, #dim):', self.keyPos3d.shape)

        self.cams = cams

        # obtain intrinsic and extrinsic parameters
        parser = configparser.ConfigParser()
        cfgs = parser.read(self.path + 'parameters.cfg')

        self.cameraIntrinsics = get_intrinsic_params(parser)

        self.cameraExtrinsics = []
        self.trans = []

        for i in range(self.n_cams):
            RT, R, t = get_extrinsic_params(parser, cam_id=i)
            self.cameraExtrinsics.append(RT)
            self.trans.append(t)

    def __len__(self):
        return len(self.steps)

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

    def sample_points_uniform(self, n_samps, idx, mask, depth, pos_enc):
        res = mask.shape
        samps = np.random.randint(0, len(idx[0]), n_samps)
        samps_x = idx[0][samps]
        samps_y = idx[1][samps]

        if self.NORM:
            samps_pos = np.stack([samps_y/res[1], samps_x/res[0]], axis=1)
        else:
            samps_pos = np.stack([samps_y, samps_x], axis=1)

        return mask[samps_x, samps_y],\
            depth[samps_x, samps_y],\
            samps_pos,\
            pos_enc[samps_x, samps_y, :]

    def __getitem__(self, idx):
        idx = self.steps[idx]

        # TODO: update once dataset is updated
        if self.type == 'pigeons':
            keyPos3d = self.keyPos3d[idx, :, :]
        else:
            keyPos3d = self.keyPos3d[idx-1, :, :]
        onehot = np.eye(self.nkps)
        masks = []
        pos_enc_maps = []
        depth_maps = []
        sup_pos = []
        sup_gt = []
        sup_pos_enc = []
        sup_dep = []
        sup_pos_foreground = []
        cams = []
        kps_2ds = []

        cams_curr = self.cams
        if cams_curr is None:
            cams_curr = [np.random.randint(0, self.n_cams)]

        # when training we augment by a cyclic shift of the cameras and accordingly rotating the 3D keypoints
        # This helps the model to learn rotation equivariance
        if self.supervision_distr == 'training' and self.cam_aug and not self.mode == 'test':
            rot_steps = np.random.randint(0, 8)
        else:
            rot_steps = 0
            print('[INFO] No rotation augmentation for frame {}!'.format(idx))
        # rotated 3D keypoints into view of "virtual camera"
        r = rotation.from_euler('z', 45*rot_steps, degrees=True).as_matrix()
        keyPos3d_unperturbed = keyPos3d
        if self.noise_aug is not None:
            keyPos3d = keyPos3d + np.random.randn(*list(keyPos3d.shape)) * self.noise_aug
        keyPos3d_unperturbed = (r @ keyPos3d_unperturbed.T).T
        keyPos3d = (r @ keyPos3d.T).T

        for camera_idx in cams_curr:
            # calculate index of "virtual camera"
            ring = math.floor(camera_idx/8)
            camera_idx_rot = ((camera_idx % 8) + rot_steps) % 8 + ring*8

            path = self.path + '{:03d}/'.format(camera_idx)

            intrinsics = self.cameraIntrinsics
            extrinsics = self.cameraExtrinsics[camera_idx_rot]
            t_inv = -extrinsics[:3, :3].T @ extrinsics[:3, 3]
            R_inv = extrinsics[:3, :3].T
            ex_inv = np.zeros((3, 4))
            ex_inv[:3, :3] = R_inv
            ex_inv[:3, 3] = t_inv
            trans = self.trans[camera_idx_rot]

            if self.type == 'pigeons':
                mask = imageio.imread(path + 'objectID{:04d}.{:03d}.png'.format(idx, camera_idx))[:, :, 0]
            else:
                mask = imageio.imread(path + 'objectID{:04d}.{:03d}.png'.format(idx, camera_idx))
                mask[mask < 255] = 0
            depth = cv2.imread(
                path + 'depth{:04d}.{:03d}.exr'.format(idx, camera_idx),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
            )
            depth = depth[:, :, 0]
            # get positional encoding map
            pos_enc_path = os.path.join('./data/multiview_cow/cow_positional_encoding', '{:03d}/'.format(camera_idx))
            pos_enc = imageio.imread(
                os.path.join(pos_enc_path, 'Image{:04d}.{:03d}.png'.format(idx, camera_idx))
            ).astype(np.float32)
            pos_enc[mask < 255] = 0
            pos_enc /= 255

            camera_params = {'intrinsics': intrinsics, 'extrinsics': extrinsics}
            kps_2d = proj(camera_params, keyPos3d)

            if self.NORM:
                kps_2d[:, 0] /= mask.shape[1]
                kps_2d[:, 1] /= mask.shape[0]

            # normalize depth map, such that depth of closest key point is 0 and depth of farthest keypoint is 1
            keyPos_hom = np.transpose(np.concatenate([keyPos3d, np.ones([keyPos3d.shape[0], 1])], axis=1))
            tmp = np.transpose(extrinsics @ keyPos_hom)
            d = tmp[:, -1]
            min_d = np.min(d, axis=0)
            depth -= min_d
            depth = depth / np.max(d-min_d, axis=0)
            depth[mask != 255] = -1.0

            mask[mask == 255] = 1
            n_far = int(self.n_supervision_points/2)
            n_near = self.n_supervision_points - n_far
            # do uniform sampling for silhouette supervision if
            # a) it is specified, which is the case for inverse rendering
            # b) LFN baseline during training (for this baseline concentrated is too complex)
            if self.supervision_distr == 'uniform' or (self.supervision_distr == 'training' and not self.NORM):
                background_idx = np.where(mask == 0)
                foreground_idx = np.where(mask == 1)
                sup_gt_fore, depth_gt, sup_pos_fore, pos_enc_gt = self.sample_points_uniform(
                    n_far,
                    foreground_idx,
                    mask,
                    depth,
                    pos_enc
                )
                sup_gt_back, _, sup_pos_back, _ = self.sample_points_uniform(
                    n_near, background_idx, mask, depth, pos_enc
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
                _, depth_gt, sup_pos_fore, pos_enc_gt = self.sample_points_uniform(
                    self.n_supervision_points_object,
                    np.where(mask == 1),
                    mask,
                    depth,
                    pos_enc
                )

            cams.append({'intrinsics': torch.from_numpy(intrinsics).float(),
                         'extrinsics': torch.from_numpy(extrinsics).float(),
                         'ex_inv': torch.from_numpy(ex_inv).float(),
                         'trans': torch.from_numpy(trans).float()})
            masks.append(torch.from_numpy(mask))
            pos_enc_maps.append(torch.from_numpy(pos_enc))
            sup_pos.append(torch.from_numpy(sup_pos_).float())
            sup_gt.append(torch.from_numpy(sup_gt_).float())
            sup_pos_enc.append(torch.from_numpy(pos_enc_gt).float())
            sup_dep.append(torch.from_numpy(depth_gt).float())
            sup_pos_foreground.append(torch.from_numpy(sup_pos_fore).float())
            kps_2ds.append(torch.from_numpy(kps_2d).float())
            depth_maps.append(torch.from_numpy(depth).float())

        r = rotation.from_euler('z', np.random.uniform(0, 360), degrees=True).as_matrix()
        keyPos3d_rot = (r @ keyPos3d.T).T
        keyPos3d_rot_unperturbed = (r @ keyPos3d_unperturbed.T).T

        return {'input_pos': torch.from_numpy(keyPos3d_unperturbed).float() / (2*self.radius),
                'input_pos_pert': torch.from_numpy(keyPos3d).float() / (2*self.radius),
                'radius': torch.tensor(self.radius).float(),
                'input_feats': torch.from_numpy(onehot).float(),
                'supervision_pos': sup_pos,
                'supervision_gt': sup_gt,
                'supervision_gt_depth': sup_dep,
                'supervision_gt_pos_enc': sup_pos_enc,
                'supervision_pos_fore': sup_pos_foreground,
                'camera_params': cams,
                'mask': masks,
                'pos_enc_maps': pos_enc_maps,
                'kps_2ds': kps_2ds,
                'depth_maps': depth_maps,
                'frame': idx,
                'input_pos_rot': torch.from_numpy(keyPos3d_rot_unperturbed).float() / (2*self.radius),
                'input_pos_pert_rot': torch.from_numpy(keyPos3d_rot).float() / (2*self.radius)}

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


class SyntheticDatasetNovelViews(Dataset):
    def __init__(self, species, mode, batch_size, nkps, num_datapoints, radius, cfg_novel_views,
                 cams=None):

        self.type = species
        self.mode = mode
        self.nkps = nkps

        self.cfg_novel_views = cfg_novel_views

        np.random.seed(0)
        self.N = num_datapoints
        self.radius = radius

        if species == 'humans':
            perm = np.random.permutation(num_datapoints)
            time_steps = np.arange(num_datapoints) + 1
            perm_steps = time_steps[perm]
            n_train = int(num_datapoints*0.70)
            n_val = int(num_datapoints*0.1)
            n_test = num_datapoints - n_train - n_val
            train_steps = perm_steps[:n_train]
            val_steps = perm_steps[n_train:n_train+n_val]
            test_steps = perm_steps[-n_test:]
            print(len(test_steps))
            track_texture_steps = time_steps
        else:
            n_splits = int(num_datapoints/10)
            n_train = floor(n_splits*0.7)
            n_val = floor(n_splits*0.1)
            n_test = n_splits - n_train - n_val

            A = np.arange(num_datapoints)
            if species != 'pigeons':
                A += 1

            track_texture_steps = A

            chunks = np.stack(np.split(A, n_splits), axis=0)

            perm = np.random.permutation(n_splits)

            chunks = chunks[perm, :]

            chunks_train = chunks[:n_train, :]
            chunks_val = chunks[n_train:n_train+n_val, :]
            chunks_test = chunks[-n_test:, :]

            train_steps = np.reshape(chunks_train, [-1])
            val_steps = np.reshape(chunks_val, [-1])
            test_steps = np.reshape(chunks_test, [-1])

        steps = {
            'train': train_steps,
            'val': val_steps,
            'test': test_steps,
            'track_texture_novel_views': track_texture_steps[0::90],
            'track_texture_real_world': np.array([895])  # 188, 548, 895
        }

        self.steps = steps[mode]
        if mode == 'track_texture_novel_views':
            paths = {
                'cows': './data/multiview_cow/cow_animation_alltextures_10newcameras.blend/Cow_Holstein.tif/'
            }
        elif mode == 'track_texture_real_world':
            paths = {
                'cows': './data/multiview_cow/cow_real_world/'
            }
        else:
            paths = {'pigeons': './data/multiview_pigeon/2021.11.26 - pigeonanimation 56cams/',
                     'humans': './data/multiview_smplx/male/pose/',
                     'cows': './data/multiview_cow/cow_animation_alltextures.blend/Cow_Holstein.tif/',
                     'giraffes': './data/multiview_giraffe/giraffe_animation.blend/'
                     }
        self.path = paths[species]
        self.batch_size = batch_size

        self.keyPos3d = np.genfromtxt(self.path + 'keypoints.csv', delimiter=',')[1:, 1:]
        n_cams = 360
        if species == 'pigeons':
            self.keyPos3d = np.concatenate([self.keyPos3d[:, :7*3], self.keyPos3d[:, 9*3:]], axis=1)

        self.keyPos3d = self.keyPos3d.reshape((self.keyPos3d.shape[0], -1, 3))
        print('[INFO] Shape of', species, 'keypoint array (#frames, #kpts, #dim):', self.keyPos3d.shape)

        if cams is None:
            self.cams = range(n_cams)
        else:
            self.cams = cams

        # the maximum and minimum values per spatial dimension are:
        # [0.1713163  0.16732147 0.1758557 ]
        # [-0.1597985  -0.15980852 -0.01288748]
        #TODO: provide normalized and unnormalized for projection
        #TODO: or find global normalization
        # self.keyPos3d = self.keyPos3d * np.expand_dims(np.array([5, 5, 5]), axis=[0, 1])

        parser = configparser.ConfigParser()

        cfgs = parser.read(self.cfg_novel_views)

        self.cameraIntrinsics = get_intrinsic_params(parser)
        self.cameraExtrinsics = []
        self.trans = []

        scale = {
            'cows': 8.7
        }
        offset = {
            'cows': 0.66
        }

        for i in self.cams:
            if mode == 'track_texture_novel_views' or mode == 'track_texture_real_world':
                RT, _, t = get_extrinsic_params(parser, cam_id=i)
            else:
                RT, _, t = get_extrinsic_params(
                    parser, cam_id=i, scale_distance=scale[species], z_offset=offset[species]
                )

            self.cameraExtrinsics.append(RT)
            self.trans.append(t)

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        idx = self.steps[idx]

        if self.mode == 'track_texture_novel_views':
            keyPos3d = self.keyPos3d[int((idx - 1) / 90), :, :]
        else:
            if self.type == 'pigeons':
                keyPos3d = self.keyPos3d[idx, :, :]
            else:
                keyPos3d = self.keyPos3d[idx-1, :, :]
        onehot = np.eye(self.nkps)
        cams = []

        for j, camera_idx in enumerate(self.cams):
            intrinsics = self.cameraIntrinsics
            extrinsics = self.cameraExtrinsics[j]
            t_inv = -extrinsics[:3, :3].T @ extrinsics[:3, 3]
            R_inv = extrinsics[:3, :3].T
            ex_inv = np.zeros((3, 4))
            ex_inv[:3, :3] = R_inv
            ex_inv[:3, 3] = t_inv
            trans = self.trans[j]

            cams.append({'intrinsics': torch.from_numpy(intrinsics).float(),
                         'extrinsics': torch.from_numpy(extrinsics).float(),
                         'ex_inv': torch.from_numpy(ex_inv).float(),
                         'trans': torch.from_numpy(trans).float()})

        return {'input_pos': torch.from_numpy(keyPos3d).float() / (2*self.radius),
                'input_feats': torch.from_numpy(onehot).float(),
                'camera_params': cams,
                'frame': idx}

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


class SyntheticTextureDataset(Dataset):
    def __init__(self, species, mode, n_supervision_points, n_supervision_points_object, supervision_distr, batch_size,
                 nkps, num_datapoints, radius, ncams, std_near, std_far,
                 cams=None, normalize_pixel_coords=True, noise_aug=None, cam_aug=True, exp_name_pos_enc=None):
        self.exp_name_pos_enc = exp_name_pos_enc

        self.type = species
        self.mode = mode
        self.nkps = nkps
        self.ncams = ncams

        self.noise_aug = noise_aug
        self.cam_aug = cam_aug
        self.std_near = std_near
        self.std_far = std_far

        self.N = num_datapoints

        self.radius = radius
        np.random.seed(0)

        # compute splits
        if species == 'humans':
            perm = np.random.permutation(num_datapoints)
            time_steps = np.arange(num_datapoints) + 1
            perm_steps = time_steps[perm]
            n_train = int(num_datapoints * 0.70)
            n_val = int(num_datapoints * 0.1)
            n_test = num_datapoints - n_train - n_val
            train_steps = perm_steps[:n_train]
            val_steps = perm_steps[n_train:n_train + n_val]
            test_steps = perm_steps[-n_test:]
            print(len(test_steps))
            track_texture_steps = time_steps
        else:
            n_splits = int(num_datapoints / 10)
            n_train = floor(n_splits * 0.7)
            n_val = floor(n_splits * 0.1)
            n_test = n_splits - n_train - n_val

            A = np.arange(num_datapoints)
            # TODO: update once dataset is updated!
            if species != 'pigeons':
                A += 1

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
            'track_texture': track_texture_steps,
            'track_texture_novel_views': track_texture_steps[0::90],
            'track_texture_occlusions': track_texture_steps[:50],
            'track_texture_occlusions_complete_masks': track_texture_steps[:50],
            'track_texture_real_world': np.array([193])  # np.array([896])
        }
        self.steps = steps[mode]

        if mode == 'track_texture_novel_views':
            paths = {
                'cows': [
                    './data/multiview_cow/cow_animation_alltextures_10newcameras.blend/Cow_Holstein.tif/',
                    './data/multiview_cow/cow_animation_alltextures_10newcameras.blend/Cow_brown.png/',
                    './data/multiview_cow/cow_animation_alltextures_10newcameras.blend/Cow_brown2.png/',
                    './data/multiview_cow/cow_animation_alltextures_10newcameras.blend/Cow_freckles.png/',
                    './data/multiview_cow/cow_animation_alltextures_10newcameras.blend/Cow_freckles2.png/',
                    './data/multiview_cow/cow_animation_alltextures_10newcameras.blend/Cow_holstein2.png/',
                    './data/multiview_cow/cow_animation_alltextures_10newcameras.blend/Cow_holstein3.png/',
                    './data/multiview_cow/cow_animation_alltextures_10newcameras.blend/Cow_holstein4.png/',
                    './data/multiview_cow/cow_animation_alltextures_10newcameras.blend/Cow_holstein5.png/',
                    './data/multiview_cow/cow_animation_alltextures_10newcameras.blend/Cow_Limousine_Spec.tif/',
                    './data/multiview_cow/cow_animation_alltextures_10newcameras.blend/Cow_white.png/',
                    './data/multiview_cow/cow_animation_alltextures_10newcameras.blend/Cow_Limousine.tif/'
                ]
            }
        elif mode == 'track_texture_occlusions' or mode == 'track_texture_occlusions_complete_masks':
            paths = {
                'cows': [
                    './data/multiview_cow/cow_occlusions/'
                ]
            }
        elif mode == 'track_texture_real_world':
            paths = {
                'cows': [
                    './data/multiview_cow/cow_real_world/'
                ]
            }
        else:
            paths = {'pigeons': ['./data/multiview_pigeon/2021.11.26 - pigeonanimation 56cams/'],
                     'humans': ['./data/multiview_smplx/male/pose/'],
                     'cows': [
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_Holstein.tif/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_brown.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_brown2.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_freckles.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_freckles2.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_holstein2.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_holstein3.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_holstein4.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_holstein5.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_Limousine_Spec.tif/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_white.png/',
                         './data/multiview_cow/cow_animation_Limousine.blend/'
                     ],
                     'giraffes': ['./data/multiview_giraffe/giraffe_animation.blend/']
                     }
        self.path = paths[species]
        print('[INFO] Number of textures: {}'.format(len(self.path)))
        self.batch_size = batch_size
        self.n_supervision_points = n_supervision_points
        self.n_supervision_points_object = n_supervision_points_object
        self.supervision_distr = supervision_distr
        self.NORM = normalize_pixel_coords

        self.keyPos3d = np.genfromtxt(self.path[0] + 'keypoints.csv', delimiter=',')[1:, 1:]
        # TODO n_cams is different for novel views!!!
        self.n_cams = 24
        # for pigeons remove the unfolded wing keypoints and use 56 cameras
        if species == 'pigeons':
            self.keyPos3d = np.concatenate([self.keyPos3d[:, :7 * 3], self.keyPos3d[:, 9 * 3:]], axis=1)
            self.n_cams = 56

        self.keyPos3d = self.keyPos3d.reshape((self.keyPos3d.shape[0], -1, 3))
        print('[INFO] Shape of', species, 'keypoint array (#frames, #kpts, #dim):', self.keyPos3d.shape)

        self.cams = cams

        # obtain intrinsic and extrinsic parameters
        parser = configparser.ConfigParser()
        cfgs = parser.read(self.path[0] + 'parameters.cfg')

        self.cameraIntrinsics = get_intrinsic_params(parser)

        self.cameraExtrinsics = []
        self.trans = []

        if mode == 'track_texture_novel_views' or mode == 'track_texture_occlusions'\
                or mode == 'track_texture_occlusions_complete_masks' or mode == 'track_texture_real_world':
            for i in cams:
                RT, _, t = get_extrinsic_params(parser, cam_id=i)
                self.cameraExtrinsics.append(RT)
                self.trans.append(t)
        else:
            for i in range(self.n_cams):
                RT, _, t = get_extrinsic_params(parser, cam_id=i)
                self.cameraExtrinsics.append(RT)
                self.trans.append(t)

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
        texture = floor(idx / len(self.steps))  # assign idx to texture

        idx -= texture * len(self.steps)
        idx = self.steps[idx]

        if self.mode == 'track_texture_novel_views':
            keyPos3d = self.keyPos3d[int((idx - 1) / 90), :, :]
        else:
            # TODO: update once dataset is updated
            if self.type == 'pigeons':
                keyPos3d = self.keyPos3d[idx, :, :]
            else:
                keyPos3d = self.keyPos3d[idx - 1, :, :]
        onehot = np.eye(self.nkps)
        masks = []
        color_maps = []
        masked_color_maps = []
        sup_col = []
        pos_enc_maps = []
        sup_pos_foreground = []
        cams = []
        kps_2ds = []

        cams_curr = self.cams
        if cams_curr is None:
            cams_curr = [np.random.randint(0, self.n_cams)]

        keyPos3d_unperturbed = keyPos3d
        if self.noise_aug is not None:
            keyPos3d = keyPos3d + np.random.randn(*list(keyPos3d.shape)) * self.noise_aug

        for camera_idx, cam_name in enumerate(cams_curr):
            path = self.path[texture] + '{:03d}/'.format(cam_name)
            path_masks = self.path[0] + '{:03d}/'.format(cam_name)

            intrinsics = self.cameraIntrinsics
            extrinsics = self.cameraExtrinsics[camera_idx]
            t_inv = -extrinsics[:3, :3].T @ extrinsics[:3, 3]
            R_inv = extrinsics[:3, :3].T
            ex_inv = np.zeros((3, 4))
            ex_inv[:3, :3] = R_inv
            ex_inv[:3, 3] = t_inv
            trans = self.trans[camera_idx]

            if self.type == 'pigeons':
                color = imageio.imread(path + 'Image{:04d}.{:03d}.png'.format(idx, cam_name)).astype(np.float32)
                mask = imageio.imread(path_masks + 'objectID{:04d}.{:03d}.png'.format(idx, cam_name))[:, :, 0]
            else:
                # color
                if self.mode == 'track_texture_real_world':
                    color = imageio.imread(path + 'Image{:04d}.{:03d}.png'.format(idx, cam_name))[:, :, :3]\
                        .astype(np.float32)
                else:
                    color = imageio.imread(path + 'Image{:04d}.{:03d}.jpg'.format(idx, cam_name)).astype(np.float32)
                # mask
                if self.mode == 'track_texture_occlusions_complete_masks':
                    mask = imageio.imread(path_masks + 'objectID_complete_mask_{:04d}.{:03d}.png'.format(idx, cam_name))
                else:
                    mask = imageio.imread(path_masks + 'objectID{:04d}.{:03d}.png'.format(idx, cam_name))
                    if self.mode == 'track_texture_real_world':
                        mask = mask[:, :, 0]
                mask[mask < 255] = 0
            color = color / 255
            # create masked color map
            masked_color = color.copy()
            masked_color[mask < 255] = 0
            # get positional encoding map
            # learned positional encoding
            if self.exp_name_pos_enc is not None:
                if self.mode == 'track_texture' or self.mode == 'track_texture_occlusions'\
                        or self.mode == 'track_texture_occlusions_complete_masks':
                    for mode in ['train', 'test', 'val']:
                        try:
                            pos_enc_path = os.path.join('./experiments', self.exp_name_pos_enc, 'recs_' + mode)
                            pos_enc = np.load(
                                os.path.join(pos_enc_path, 'rec_pos_enc_frame{}_camera{}.npy'.format(idx, cam_name))
                            )
                            # use intersection of gt and learned mask
                            pos_enc[mask < 255] = 0
                        except FileNotFoundError:
                            pass
                else:
                    pos_enc_path = os.path.join('./experiments', self.exp_name_pos_enc, 'recs_' + self.mode)
                    pos_enc = np.load(
                        os.path.join(pos_enc_path, 'rec_pos_enc_frame{}_camera{}.npy'.format(idx, cam_name))
                    )
                    if self.mode == 'track_texture_novel_views':
                        # use intersection of gt and learned mask
                        pos_enc[mask < 255] = 0
                pos_enc = pos_enc.astype(np.float32)
                if self.mode == 'track_texture_real_world':
                    pos_enc[mask < 255] = 0
            # ground truth positional encoding
            else:
                if self.mode == 'track_texture_novel_views':
                    pos_enc_path = os.path.join('./data/multiview_cow/'
                                                'cow_animation_positional_encoding_10newcameras.blend',
                                                '{:03d}/'.format(cam_name))
                elif self.mode == 'track_texture_real_world':
                    print("!!!ATTENTION: There is no pos. enc. ground truth for our real-world examples!!!")
                    sys.exit()
                else:
                    pos_enc_path = os.path.join('./data/multiview_cow/cow_positional_encoding',
                                                '{:03d}/'.format(cam_name))
                pos_enc = imageio.imread(
                    os.path.join(pos_enc_path, 'Image{:04d}.{:03d}.png'.format(idx, cam_name))
                ).astype(np.float32)
                pos_enc[mask < 255] = 0
            pos_enc = pos_enc / 255
            if self.mode in [
                'track_texture', 'track_texture_novel_views', 'track_texture_occlusions',
                'track_texture_occlusions_complete_masks'
            ]:
                # use intersection of gt and learned mask for masked color map
                masked_color[pos_enc == 0] = 0

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
                         'extrinsics': torch.from_numpy(extrinsics).float(),
                         'ex_inv': torch.from_numpy(ex_inv).float(),
                         'trans': torch.from_numpy(trans).float()})
            masks.append(torch.from_numpy(mask))
            sup_col.append(torch.from_numpy(color_gt).float())
            pos_enc_maps.append(torch.from_numpy(pos_enc))
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
                'frame': idx,
                'pos_enc_maps': pos_enc_maps,
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


class SyntheticTextureDatasetNovelViews(Dataset):
    def __init__(self, species, mode, batch_size, nkps, num_datapoints, radius, cfg_novel_views,
                 cams=None, exp_name_pos_enc=None):
        self.exp_name_pos_enc = exp_name_pos_enc

        self.type = species
        self.mode = mode
        self.nkps = nkps

        self.cfg_novel_views = cfg_novel_views

        np.random.seed(0)
        self.N = num_datapoints
        self.radius = radius

        if species == 'humans':
            perm = np.random.permutation(num_datapoints)
            time_steps = np.arange(num_datapoints) + 1
            perm_steps = time_steps[perm]
            n_train = int(num_datapoints*0.70)
            n_val = int(num_datapoints*0.1)
            n_test = num_datapoints - n_train - n_val
            train_steps = perm_steps[:n_train]
            val_steps = perm_steps[n_train:n_train+n_val]
            test_steps = perm_steps[-n_test:]
            print(len(test_steps))
        else:
            n_splits = int(num_datapoints/10)
            n_train = floor(n_splits*0.7)
            n_val = floor(n_splits*0.1)
            n_test = n_splits - n_train - n_val

            A = np.arange(num_datapoints)
            if species != 'pigeons':
                A += 1

            chunks = np.stack(np.split(A, n_splits), axis=0)

            perm = np.random.permutation(n_splits)

            chunks = chunks[perm, :]

            chunks_train = chunks[:n_train, :]
            chunks_val = chunks[n_train:n_train+n_val, :]
            chunks_test = chunks[-n_test:, :]

            train_steps = np.reshape(chunks_train, [-1])
            val_steps = np.reshape(chunks_val, [-1])
            test_steps = np.reshape(chunks_test, [-1])

        steps = {
            'train': train_steps,
            'val': val_steps,
            'test': test_steps,
            'track_texture_real_world': np.array([895])  # 188, 548, 895
        }

        self.steps = steps[mode]

        if mode == 'track_texture_real_world':
            paths = {
                'cows': ['./data/multiview_cow/cow_real_world/']
            }
        else:
            paths = {'pigeons': ['./data/multiview_pigeon/2021.11.26 - pigeonanimation 56cams/'],
                     'humans': ['./data/multiview_smplx/male/pose/'],
                     'cows': [
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_Holstein.tif/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_brown.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_brown2.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_freckles.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_freckles2.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_holstein2.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_holstein3.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_holstein4.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_holstein5.png/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_Limousine_Spec.tif/',
                         './data/multiview_cow/cow_animation_alltextures.blend/Cow_white.png/',
                         './data/multiview_cow/cow_animation_Limousine.blend/'
                     ],
                     'giraffes': ['./data/multiview_giraffe/giraffe_animation.blend/']
                     }
        self.path = paths[species]
        print('[INFO] Number of textures: {}'.format(len(self.path)))
        self.batch_size = batch_size

        self.keyPos3d = np.genfromtxt(self.path[0] + 'keypoints.csv', delimiter=',')[1:, 1:]
        n_cams = 360
        if species == 'pigeons':
            self.keyPos3d = np.concatenate([self.keyPos3d[:, :7 * 3], self.keyPos3d[:, 9 * 3:]], axis=1)

        self.keyPos3d = self.keyPos3d.reshape((self.keyPos3d.shape[0], -1, 3))
        print('[INFO] Shape of', species, 'keypoint array (#frames, #kpts, #dim):', self.keyPos3d.shape)

        n_cams = 360

        if cams is None:
            self.cams = range(n_cams)
        else:
            self.cams = cams

        # the maximum and minimum values per spatial dimension are:
        # [0.1713163  0.16732147 0.1758557 ]
        # [-0.1597985  -0.15980852 -0.01288748]
        #TODO: provide normalized and unnormalized for projection
        #TODO: or find global normalization
        # self.keyPos3d = self.keyPos3d * np.expand_dims(np.array([5, 5, 5]), axis=[0, 1])

        parser = configparser.ConfigParser()

        cfgs = parser.read(self.cfg_novel_views)

        self.cameraIntrinsics = get_intrinsic_params(parser)
        self.cameraExtrinsics = []
        self.trans = []

        scale = {
            'cows': 8.7
        }
        offset = {
            'cows': 0.66
        }

        for i in self.cams:
            if mode == 'track_texture_real_world':
                RT, _, t = get_extrinsic_params(parser, cam_id=i)
            else:
                RT, _, t = get_extrinsic_params(
                    parser, cam_id=i, scale_distance=scale[species], z_offset=offset[species]
                )

            self.cameraExtrinsics.append(RT)
            self.trans.append(t)

    def __len__(self):
        return len(self.steps) * len(self.path)

    def __getitem__(self, idx):
        texture = floor(idx / len(self.steps))  # assign idx to texture

        idx -= texture * len(self.steps)
        idx = self.steps[idx]

        if self.type == 'pigeons':
            keyPos3d = self.keyPos3d[idx, :, :]
        else:
            keyPos3d = self.keyPos3d[idx-1, :, :]
        onehot = np.eye(self.nkps)
        cams = []

        # randomly select one gt camera view
        if self.type == 'pigeons':
            cam_n = 56
        else:
            cam_n = 24

        if self.mode == 'track_texture_real_world':
            camera_idx = self.cams[0]
        else:
            camera_idx = np.random.randint(0, cam_n)

        path = self.path[texture] + '{:03d}/'.format(camera_idx)
        path_masks = self.path[0] + '{:03d}/'.format(camera_idx)

        if self.type == 'pigeons':
            color = imageio.imread(path + 'Image{:04d}.{:03d}.png'.format(idx, camera_idx)).astype(np.float32)
            mask = imageio.imread(path_masks + 'objectID{:04d}.{:03d}.png'.format(idx, camera_idx))[:, :, 0]
        else:
            # color
            if self.mode == 'track_texture_real_world':
                color = imageio.imread(path + 'Image{:04d}.{:03d}.png'.format(idx, camera_idx))[:, :, :3] \
                    .astype(np.float32)
            else:
                color = imageio.imread(path + 'Image{:04d}.{:03d}.jpg'.format(idx, camera_idx)).astype(np.float32)
            # mask
            mask = imageio.imread(path_masks + 'objectID{:04d}.{:03d}.png'.format(idx, camera_idx))
            if self.mode == 'track_texture_real_world':
                mask = mask[:, :, 0]
            mask[mask < 255] = 0
        color = color / 255
        # create masked color map
        masked_color = color.copy()
        masked_color[mask < 255] = 0
        # get positional encoding map
        # learned positional encoding
        if self.exp_name_pos_enc is not None:
            pos_enc_path = os.path.join('./experiments', self.exp_name_pos_enc, 'recs_' + self.mode)
            pos_enc = np.load(
                os.path.join(pos_enc_path, 'rec_pos_enc_frame{}_camera{}.npy'.format(idx, camera_idx))
            )
            pos_enc = pos_enc.astype(np.float32)
            if self.mode == 'track_texture_real_world':
                pos_enc[mask < 255] = 0
        # TODO Urs: Delete gt pos. enc. part for novel views?!
        # ground truth positional encoding
        else:
            pos_enc_path = os.path.join('./data/multiview_cow/cow_positional_encoding', '{:03d}/'.format(camera_idx))
            pos_enc = imageio.imread(
                os.path.join(pos_enc_path, 'Image{:04d}.{:03d}.png'.format(idx, camera_idx))
            ).astype(np.float32)
            pos_enc[mask < 255] = 0
        pos_enc = pos_enc / 255

        for j, camera_idx in enumerate(self.cams):
            intrinsics = self.cameraIntrinsics
            extrinsics = self.cameraExtrinsics[j]
            t_inv = -extrinsics[:3, :3].T @ extrinsics[:3, 3]
            R_inv = extrinsics[:3, :3].T
            ex_inv = np.zeros((3, 4))
            ex_inv[:3, :3] = R_inv
            ex_inv[:3, 3] = t_inv
            trans = self.trans[j]

            cams.append({'intrinsics': torch.from_numpy(intrinsics).float(),
                         'extrinsics': torch.from_numpy(extrinsics).float(),
                         'ex_inv': torch.from_numpy(ex_inv).float(),
                         'trans': torch.from_numpy(trans).float()})

        return {'input_pos': torch.from_numpy(keyPos3d).float() / (2*self.radius),
                'input_feats': torch.from_numpy(onehot).float(),
                'camera_params': cams,
                'random_masked_color_map': masked_color,
                'random_pos_enc_map': pos_enc,
                'frame': idx,
                'texture': texture}

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


#TODO finish up final params!!
def get_synthetic_dataset(data_type, mode, sup_distr, cfg, cams=None):
    return SyntheticDataset(species=data_type,
                            mode=mode,
                            n_supervision_points=cfg['training']['npoints_renderer'],
                            n_supervision_points_object=cfg['training']['npoints_object_renderer'],
                            supervision_distr=sup_distr,
                            batch_size=cfg['training']['batch_size'],
                            nkps=cfg['data']['nkps'],
                            num_datapoints=cfg['data']['num_datapoints'],
                            radius=cfg['data']['radius'],
                            ncams=cfg['data']['ncams'],
                            cams=cams,
                            normalize_pixel_coords=True,
                            noise_aug=cfg['data']['noise_aug_kpts'],
                            cam_aug=True,
                            std_near=cfg['data']['std_near'],
                            std_far=cfg['data']['std_far']
                            )


def get_synthetic_dataset_novel_views(data_type, mode, cfg, cfg_novel_views, cams=None):
    return SyntheticDatasetNovelViews(species=data_type,
                                      mode=mode,
                                      batch_size=cfg['training']['batch_size'],
                                      nkps=cfg['data']['nkps'],
                                      num_datapoints=cfg['data']['num_datapoints'],
                                      radius=cfg['data']['radius'],
                                      cams=cams,
                                      cfg_novel_views=cfg_novel_views
                                      )


def get_synthetic_texture_dataset(data_type, mode, sup_distr, cfg, cams=None, exp_name_pos_enc=None):
    return SyntheticTextureDataset(species=data_type,
                                   mode=mode,
                                   n_supervision_points=cfg['training']['npoints_renderer'],
                                   n_supervision_points_object=cfg['training']['npoints_object_renderer'],
                                   supervision_distr=sup_distr,
                                   batch_size=cfg['training']['batch_size'],
                                   nkps=cfg['data']['nkps'],
                                   num_datapoints=cfg['data']['num_datapoints'],
                                   radius=cfg['data']['radius'],
                                   ncams=cfg['data']['ncams'],
                                   cams=cams,
                                   normalize_pixel_coords=True,
                                   noise_aug=cfg['data']['noise_aug_kpts'],
                                   cam_aug=True,
                                   std_near=cfg['data']['std_near'],
                                   std_far=cfg['data']['std_far'],
                                   exp_name_pos_enc=exp_name_pos_enc
                                   )


def get_synthetic_texture_dataset_novel_views(data_type, mode, cfg, cfg_novel_views, cams=None, exp_name_pos_enc=None):
    return SyntheticTextureDatasetNovelViews(species=data_type,
                                             mode=mode,
                                             batch_size=cfg['training']['batch_size'],
                                             nkps=cfg['data']['nkps'],
                                             num_datapoints=cfg['data']['num_datapoints'],
                                             radius=cfg['data']['radius'],
                                             cams=cams,
                                             cfg_novel_views=cfg_novel_views,
                                             exp_name_pos_enc=exp_name_pos_enc
                                             )
