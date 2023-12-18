from __future__ import division
import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
from time import time

from .utils import vae_loss


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer(object):

    def __init__(self, encoder, texture_encoder, decoder, texture_decoder, texture_decoder_impl, cfg, device,
                 train_dataset, val_dataset, exp_name,
                 optimizer='Adam'):
        self.encoder = encoder.to(device)
        self.texture_encoder = texture_encoder.to(device)
        self.decoder = decoder.to(device)
        self.texture_decoder = texture_decoder.to(device)
        self.texture_decoder_impl = texture_decoder_impl.to(device)
        print('[INFO] Number of Parameters in geometry encoder: {}'.format(count_parameters(self.encoder)))
        print('[INFO] Number of Parameters in texture encoder: {}'.format(count_parameters(self.texture_encoder)))
        print('[INFO] Number of Parameters in geometry decoder: {}'.format(count_parameters(self.decoder)))
        print('[INFO] Number of Parameters in texture decoder: {}'.format(count_parameters(self.texture_decoder)))
        print('[INFO] Number of Parameters in texture renderer: {}'.format(count_parameters(self.texture_decoder_impl)))
        self.cfg = cfg
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.AdamW(params=list(texture_encoder.parameters()) +
                                                list(texture_decoder.parameters()) +
                                                list(texture_decoder_impl.parameters()),
                                         lr=self.cfg['training']['lr'],
                                         weight_decay=self.cfg['training'].get('weight_decay'))
        self.lr = self.cfg['training']['lr']

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format(exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format(exp_name)
        if not os.path.exists(self.checkpoint_path):
            print('Creating checkpoint dir: {}'.format(self.checkpoint_path))
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.val_data_loader = self.val_dataset.get_loader()

    def reduce_lr(self, epoch):
        if (epoch > 0 and self.cfg['training']['lr_decay_interval'] is not None and
                epoch % self.cfg['training']['lr_decay_interval'] == 0):
            decay_steps = int(epoch/self.cfg['training']['lr_decay_interval'])
            lr = self.cfg['training']['lr'] * self.cfg['training']['lr_decay_factor']**decay_steps
            print('Reducing LR to {}'.format(lr))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr * self.cfg['training']['lr_decay_factor']**decay_steps

    def train_step(self, batch):
        #t0_idk = time()
        self.texture_encoder.train()
        self.texture_decoder.train()
        self.texture_decoder_impl.train()
        self.optimizer.zero_grad()
        loss_color, loss_kld = self.compute_loss(batch)

        loss_tot = loss_color + loss_kld
        loss_tot.backward()
        if self.cfg['training']['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(
                self.texture_encoder.parameters(),
                max_norm=self.cfg['training']['grad_clip']
            )
            torch.nn.utils.clip_grad_norm_(
                self.texture_decoder.parameters(),
                max_norm=self.cfg['training']['grad_clip']
            )
            torch.nn.utils.clip_grad_norm_(
                self.texture_decoder_impl.parameters(),
                max_norm=self.cfg['training']['grad_clip']
            )
        self.optimizer.step()
        return loss_tot.item(), loss_color.item(), loss_kld.item()

    def compute_loss(self, batch):
        device = self.device

        inp_pos_pert = batch.get('input_pos_pert').to(device)
        radius = batch.get('radius').to(device)
        inp_feats = batch.get('input_feats').to(device)
        sup_pos_fore = batch.get('supervision_pos_fore')[0].to(device)
        sup_color = batch.get('supervision_gt_color')[0].to(device)
        kps_2d = batch.get('kps_2ds')[0].to(device)

        # get masked color map
        masked_color_map = batch.get('masked_color_maps')[0].to(device)
        masked_color_map = masked_color_map.permute(0, 3, 1, 2)
        if self.cfg['texture']['data'] != 'h36m':
            # get positional encoding map
            pos_enc_map = batch.get('pos_enc_maps')[0].to(device)
            pos_enc_map = pos_enc_map.permute(0, 3, 1, 2)
            # concatenate masked color and positional encoding map as input to texture encoder
            texture_encoder_input = torch.cat((masked_color_map, pos_enc_map), dim=1)
        else:
            texture_encoder_input = masked_color_map

        camera_params_tmp = batch.get('camera_params')[0]
        camera_params = {k: v.to(device) for (k, v) in zip(camera_params_tmp.keys(), camera_params_tmp.values())}

        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            z = self.encoder(inp_pos_pert, inp_feats)
            encoding = self.decoder(z)

        mu, logvar, z_texture = self.texture_encoder(texture_encoder_input)
        texture_encoding = self.texture_decoder(z_texture, encoding)
        if 'anchors' in encoding:
            encoding['anchors'] = encoding['anchors']*2*radius.unsqueeze(1).unsqueeze(2)

        color_pred = self.texture_decoder_impl(
            sup_pos_fore,
            encoding,
            texture_encoding,
            camera_params,
            kps_2d
        )
        color_pred = color_pred.squeeze()
        loss_color, kl_divergence = vae_loss(color_pred, mu, logvar, sup_color)

        return self.cfg['training'].get('weight_color_loss') * loss_color,\
            self.cfg['training'].get('weight_KLD_loss') * kl_divergence

    def train_model(self, epochs, ckp_interval=5):
        loss = 0
        start = self.load_checkpoint()
        self.load_val_min()

        for epoch in range(start, epochs):
            self.reduce_lr(epoch)
            sum_loss = 0
            sum_loss_color = 0
            sum_loss_kld = 0
            train_data_loader = self.train_dataset.get_loader()

            for batch in train_data_loader:
                loss, loss_color, loss_kld, = self.train_step(batch)

                sum_loss += loss
                sum_loss_color += loss_color
                sum_loss_kld += loss_kld

            self.writer.add_scalar('training loss last batch', loss, epoch)
            self.writer.add_scalar('training loss batch avg', sum_loss / len(train_data_loader), epoch)
            self.writer.add_scalar('training loss color batch avg', sum_loss_color / len(train_data_loader), epoch)
            self.writer.add_scalar('training loss KLD batch avg', sum_loss_kld / len(train_data_loader), epoch)

            if epoch % ckp_interval == 0:
                self.save_checkpoint(epoch)
                val_loss, val_loss_color, val_loss_kld = self.compute_val_loss()

                if self.val_min is None:
                    self.val_min = val_loss

                if val_loss < self.val_min:
                    self.val_min = val_loss
                    for path in glob(self.exp_path + 'val_min=*'):
                        os.remove(path)
                    np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, val_loss])

                self.writer.add_scalar('val loss batch avg', val_loss, epoch)
                self.writer.add_scalar('val loss color batch avg', val_loss_color, epoch)
                self.writer.add_scalar('val loss KLD batch  avg', val_loss_kld, epoch)
                print("Epoch {:5d}: ColorLoss: {:06.6f} - {:06.6f}, KLDloss: {:06.6f} - {:06.6f}, "
                      "Total Loss: {:06.6f} - {:06.6f}"
                .format(
                    epoch,
                    sum_loss_color/len(train_data_loader),
                    val_loss_color,
                    sum_loss_kld/len(train_data_loader),
                    val_loss_kld,
                    sum_loss/len(train_data_loader),
                    val_loss
                ))

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):

            torch.save({'epoch': epoch,
                        'texture_encoder_state_dict': self.texture_encoder.state_dict(),
                        'texture_decoder_state_dict': self.texture_decoder.state_dict(),
                        'texture_decoder_impl_state_dict': self.texture_decoder_impl.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()},
                       path)

    def load_val_min(self):
        val_min = glob(self.exp_path + '/val_min*')
        if len(val_min) == 0:
            print('[INFO] No file with minimum validation loss found at {}'.format(self.exp_path))
        else:
            print('Load minimum validation loss value from {}'.format(val_min[0]))
            val_min_value = np.load(val_min[0])
            self.val_min = val_min_value[1]

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        if self.cfg['training']['ckpt'] is not None:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(self.cfg['training']['ckpt'])
        else:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.texture_encoder.load_state_dict(checkpoint['texture_encoder_state_dict'])
        self.texture_decoder.load_state_dict(checkpoint['texture_decoder_state_dict'])
        self.texture_decoder_impl.load_state_dict(checkpoint['texture_decoder_impl_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        for param_group in self.optimizer.param_groups:
            print('Setting LR to {}'.format(self.cfg['training']['lr']))
            param_group['lr'] = self.cfg['training']['lr']
        if self.cfg['training']['lr_decay_interval'] is not None:
            decay_steps = int(epoch/self.cfg['training']['lr_decay_interval'])
            lr = self.cfg['training']['lr'] * self.cfg['training']['lr_decay_factor']**decay_steps
            print('Reducing LR to {}'.format(lr))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr * self.cfg['training']['lr_decay_factor']**decay_steps
        return epoch

    def compute_val_loss(self):
        self.texture_encoder.eval()
        self.texture_decoder.eval()
        self.texture_decoder_impl.eval()

        sum_val_loss = 0
        sum_val_color = 0
        sum_val_kld = 0

        c = 0
        for val_batch in self.val_data_loader:
            l_color, l_kld = self.compute_loss(val_batch)
            sum_val_color += l_color.item()
            sum_val_kld += l_kld.item()
            sum_val_loss += l_color.item() + l_kld.item()
            c = c + 1

        return sum_val_loss / c, sum_val_color / c, sum_val_kld / c
