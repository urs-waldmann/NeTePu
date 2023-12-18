# Neural Texture Puppeteer: A Framework for Neural Geometry and Texture Rendering of Articulated Shapes, Enabling Re-Identification at Interactive Speed
This repository provides code for NeTePu (WACVW "CV4Smalls" 2024, oral).

**Abstract**

In this paper, we present a neural rendering pipeline for textured articulated shapes that we call Neural Texture Puppeteer. Our method separates geometry and texture encoding. The geometry pipeline learns to capture spatial relationships on the surface of the articulated shape from ground truth data that provides this geometric information. A texture auto-encoder makes use of this information to encode textured images into a global latent code. This global texture embedding can be efficiently trained separately from the geometry, and used in a downstream task to identify individuals. The neural texture rendering and the identification of individuals run at interactive speeds. To the best of our knowledge, we are the first to offer a promising alternative to CNN- or transformer-based approaches for re-identification of articulated individuals based on neural rendering. Realistic looking novel view and pose synthesis for different synthetic cow textures further demonstrate the quality of our method. Restricted by the availability of ground truth data for the articulated shape’s geometry, the quality for real-world data synthesis is reduced. We further demonstrate the flexibility of our model for real-world data by applying a synthetic to real-world texture domain shift where we reconstruct the texture from a real-world 2D RGB image. Thus, our method can be applied to endangered species where data is limited. Our novel synthetic texture dataset NePuMoo is publicly available to inspire further development in the field of neural rendering-based re-identification.

If you find a bug, have a question or know how to improve the code, please open an issue.

## Conda environment
Set up a conda environment with `conda env create -f environment.yml`.

## NePuMoo dataset
Our novel NePuMoo data set can be downloaded [here](). Unzip and copy the "multiview_cow" folder to `./data/`.

## Pre-trained weights
Pre-trained weights for cows (our novel NePuMoo data set), the [Human3.6M dataset](http://vision.imar.ro/human3.6m/description.php), and our ablation studies can be downloaded [here](https://zenodo.org/records/10402116). Unzip and copy the "experiments" folder to `./`.

## Rendering

### Geometry
The pipeline for geometry can render depth maps, masks, and [NNOPCS maps](https://arxiv.org/abs/2311.17109).

**Novel pose synthesis**

To render multiple views of the NePuMoo test set, run:

    python test.py -exp_name geometry_cows -checkpoint 1950

where `-exp_name` specifies the name of the experiment, and `-checkpoint` the epoch of the trained weights.

**Novel pose and view synthesis**

To render multiple novel views of the NePuMoo test set, run:

    python test_nv.py -exp_name geometry_cows -checkpoint 1950 -novel_views_cfg configs/parameters_ref.cfg

where `-novel_views_cfg` specifies the path to the configuration file containing the novel views, and the other command line arguments are the same as above.

### Texture
The texture pipeline can render color maps.

**Novel pose synthesis**

To render multiple views of the NePuMoo test set, run:

    python test_texture.py -exp_name texture_cows -checkpoint 535

where the command line arguments are the same as above.

**Novel pose and view synthesis**

To render multiple novel views of the NePuMoo test set, run:

    python test_texture_nv.py -exp_name texture_cows -checkpoint 535 -novel_views_cfg configs/parameters_ref.cfg

where the command line arguments are the same as above.

### Human3.6M
**Preliminary tasks**

Clone [this repository](https://github.com/karfly/human36m-camera-parameters) into `./data/h36m/`.

[Download](http://vision.imar.ro/human3.6m/filebrowser.php) and unzip the 3D poses (positions, not angles) of the action "Posing" for all subjects (you need a login to download). If you picked correctly, the folder you downloaded is named "Poses_D3_Positions_Posing". Copy that folder to `.data/h36m/data/`.

We use the processed data from [AniNeRF](https://github.com/zju3dv/animatable_nerf/blob/master/INSTALL.md#set-up-datasets) in `./data/h36m/data/`.

**Novel pose synthesis**

To render multiple views of the Human3.6M test set, run:

    python test_texture.py -exp_name texture_h36m -checkpoint 2080

where the command line arguments are the same as above.
