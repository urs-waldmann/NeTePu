import numpy as np
import cv2
import random
from math import floor

from scipy.spatial.transform import Rotation as rotation


def get_intrinsic_params(parser):

    intrinsics = parser['intrinsics']
    pixelsize = np.array([int(intrinsics.get('image_resolution_x_px')), int(intrinsics.get('image_resolution_y_px'))])
    pixelsize = pixelsize / float(intrinsics.get('sensor_size_mm'))

    f = float(intrinsics.get('focal_length_mm')) * np.max(pixelsize)

    c = float(intrinsics.get('sensor_size_mm'))/2. * pixelsize + 0.5

    cameraMatrix = np.array([[f, 0., c[0]],
                             [0., f, c[1]],
                             [0., 0., 1.0]])
    return cameraMatrix


def get_extrinsic_params(parser, cam_id, scale_distance=1, z_offset=0):
    ex_params = parser['extrinsics_Camera.{:03d}'.format(cam_id)]
    # bcam stands for blender camera
    R_bcam2cv = np.array(
        [[1, 0,  0],
         [0, -1, 0],
         [0, 0, -1]])

    sign = np.array([[1., -1., 1.],
                     [-1., 1., -1.],
                     [1., -1, 1.]])


    R_world2bcam = rotation.from_euler('zyx', (
        ex_params.get('center_cam_rz_rad'), ex_params.get('center_cam_ry_rad'), ex_params.get('center_cam_rx_rad'))
                                ).as_matrix()
    R_world2bcam = sign * R_world2bcam


    if scale_distance != 1:
        print('Att: Scaling camera ' + str(cam_id) + ' to sphere')

        location = np.array([float(ex_params.get('center_cam_x_m')),
                         float(ex_params.get('center_cam_y_m')),
                         float(ex_params.get('center_cam_z_m'))])
        norm = np.linalg.norm(location)
        location = (location / norm) * scale_distance
        location[-1] += z_offset
    else:
        location = np.array([float(ex_params.get('center_cam_x_m'))*scale_distance,
                             float(ex_params.get('center_cam_y_m'))*scale_distance,
                             float(ex_params.get('center_cam_z_m'))*scale_distance+z_offset])


    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1.0 * R_world2bcam @ location

    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    RT = np.concatenate([R_world2cv, np.expand_dims(T_world2cv, 1)], axis=1)
    ###RT = np.concatenate([rotm, np.expand_dims(location, 1)], axis=1)
    return RT, R_world2cv, location


def proj(camera_params, kps):
    A = camera_params['intrinsics'] @ camera_params['extrinsics']

    tmp = A @ np.concatenate([kps, np.ones([kps.shape[0], 1])], axis=1).T
    tmp = tmp.T

    tmp = tmp / tmp[:, 2:]
    return tmp[:, :2]


def view_dataset(data_loader, n_samples, n_cams=None, masked_image=False, show_kpts=False, show_mask=False):
    # calculate which samples to view
    # case: one sample to view
    if n_samples == 1:
        view_every_nth_sample = len(data_loader)
    # case: more than one sample to view
    else:
        view_every_nth_sample = len(data_loader) / (n_samples - 1)
        # case: there is no remainder
        if view_every_nth_sample % 1 == 0:
            view_every_nth_sample -= 1
        # case: there is a remainder
        else:
            view_every_nth_sample = floor(view_every_nth_sample)
        # view all samples if data set contains less than "n_samples"
        if view_every_nth_sample == 0:
            view_every_nth_sample = 1
            print('INFO: Cannot show {} samples. Show all samples, ie. {}.'.format(n_samples, len(data_loader)))
        else:
            pass
    # loop over data loader
    for i, data_point in enumerate(iter(data_loader)):
        # condition whether to view data point or not
        if i % view_every_nth_sample == 0:
            # test split: we have all camera views available
            if n_cams is not None:
                # choose random camera view
                cam_view = random.randrange(n_cams)
                if show_mask:
                    # get mask to view
                    image = (data_point['mask'][cam_view][0].numpy() * 255).astype(np.uint8)
                    image = np.stack([image, image, image], axis=2)
                    # create window name for visualization
                    window_name = 'camera ' + str(cam_view) + ', frame ' + str(data_point['frame'].item())
                else:
                    # get image to view
                    image = (data_point['color_maps'][cam_view][0].numpy() * 255).astype(np.uint8)
                    # mask image
                    if masked_image:
                        image = (data_point['masked_color_maps'][cam_view][0].numpy() * 255).astype(np.uint8)
                    # create window name for visualization
                    window_name = 'camera ' + str(cam_view) + ', texture ' + str(data_point['texture'].item()) \
                                  + ', frame ' + str(data_point['frame'].item())
            # train/val split: we have one random camera view available
            else:
                # choose sample from batch randomly
                batch = random.randrange(data_point['mask'][0].shape[0])
                if show_mask:
                    # get mask to view
                    image = (data_point['mask'][0][batch].numpy() * 255).astype(np.uint8)
                    image = np.stack([image, image, image], axis=2)
                    # create window name for visualization
                    window_name = 'random camera' + ', frame ' + str(data_point['frame'][batch].item())
                else:
                    # get image to view
                    image = (data_point['color_maps'][0][batch].numpy() * 255).astype(np.uint8)
                    # mask image
                    if masked_image:
                        image = (data_point['masked_color_maps'][0][batch].numpy() * 255).astype(np.uint8)
                    # create window name for visualization
                    window_name = 'random camera' + ', texture ' + str(data_point['texture'][batch].item())\
                                  + ', frame ' + str(data_point['frame'][batch].item())
                if show_kpts:
                    # get 2D kpts
                    kpts = data_point['kps_2ds'][0][batch].numpy()
                    kpts[:, 0] *= image.shape[1]
                    kpts[:, 1] *= image.shape[0]
                    # draw kpts on image
                    for p in kpts:
                        point = (int(p[0]), int(p[1]))
                        cv2.drawMarker(
                            image,
                            point,
                            color=(255, 0, 0),  # red (RGB)
                            markerType=cv2.MARKER_CROSS,
                            markerSize=20,
                            thickness=1
                        )
            # convert image from RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # show image
            cv2.imshow(window_name, image)
            # show until any key is pressed
            key = cv2.waitKey(0) & 0xFF
            # destroy all windows
            cv2.destroyAllWindows()
            # if "q" key is pressed exit from function
            if key == ord('q'):
                break
