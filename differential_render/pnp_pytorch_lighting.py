import traceback
from typing import Any, Optional
import cv2
import time
import math
from tqdm import tqdm
import torch
import trimesh
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from .dataset import LoadDataset
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
import PIL.Image

import pytorch3d
import ast
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams, SoftPhongShader)

from .net import unet
import segmentation_models_pytorch as smp
from .loss import SSIMLoss, HistogramLoss, GradientLoss
from ..tools import paths, utils, vis_utils
from torchsummary import summary

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# torch.set_float32_matmul_precision('high')

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
else: device = torch.device("cpu")

def freeze_encoder(model):
    for child in model.encoder.children():
        for param in child.parameters():
            param.requires_grad = False
    return

def unfreeze(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True
    return

class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

class DifferentialRenderPL(pl.LightningModule):
    def __init__(self, config, eval_output_dir):
        super().__init__()
        self.config = config
        self.eval_output_dir = eval_output_dir
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.psnr = PeakSignalNoiseRatio()
        self.model = smp.UnetPlusPlus(encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=2,                      # model output channels (number of classes in your dataset)
                        activation='sigmoid',
                    ).to(device)

        # freeze the encoder
        freeze_encoder(self.model)
        if self.config['image_size'] == [1080, 1920]: self.crop_size = 512
        elif self.config['image_size'] == [540, 960]: self.crop_size = 256
        elif self.config['image_size'] == [270, 480]: self.crop_size = 128
        elif self.config['image_size'] == [108, 192]: self.crop_size = 64
        print(summary(self.model, (3, self.crop_size, self.crop_size)))
        self.model.train()

        self.errors = {'target_rot': [], 'target_tx': [], 'target_ty': [], 'target_tz': [], 'predict_rot': [], 'predict_tx': [], 'predict_ty': [], 'predict_tz': []}
        self.latlon = utils.load_latitude_longitude(paths.PKG_ROOT.parent / "data" / "latlon" /"ossiclesCoordinateMapping2.json")
    
        # create loss functions
        for loss_func in self.config['loss'].keys():
            try: setattr(self, f'{loss_func}', getattr(torch.nn, loss_func))
            except AttributeError: 
                if loss_func == 'SSIMLoss': setattr(self, f'{loss_func}', SSIMLoss)
                elif loss_func == 'HistLoss': setattr(self, f'{loss_func}', HistogramLoss)
                elif loss_func == 'GradientLoss': setattr(self, f'{loss_func}', GradientLoss)
                
        # self.automatic_optimization = False

    def cv2_pnp(self, latlon, bbox, bbox_offset, pose, verts, faces, seg_render, seg_mask, intrinsics):
        # coordinate system change (x, y, z) -> (-x, -y, z)
        coordinate_change = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        mesh = trimesh.Trimesh(vertices=verts.numpy(), faces=faces.numpy(), process=False)
        # get the color mask
        color_mask = seg_render[..., :3].numpy()
        binary_mask = utils.color2binary_mask(color_mask)
        # assert (binary_mask == seg_mask.numpy()).all(), "bianry mask and seg mask are not the same"
        idx = np.where(binary_mask == 1)
        # swap the points for opencv, maybe because they handle RGB image differently (RGB -> BGR in opencv)
        idx = idx[:2][::-1]
        
        x, y = idx[0], idx[1]
        pts = np.stack((x, y), axis=1)
        pts3d = []
        
        # Obtain the rg color
        color = color_mask[pts[:,1], pts[:,0]][..., :2]
        if np.max(color) > 1: color = color / 255
        gx = color[:, 0]
        gy = color[:, 1]

        lat = np.array(latlon[..., 0])
        lon = np.array(latlon[..., 1])
        lonf = lon[mesh.faces]
        msk = (np.sum(lonf>=0, axis=1)==3) & (np.sum(lat[mesh.faces]>=0, axis=1)==3)
        for i in range(len(pts)):
            pt = utils.latLon2xyz(mesh, lat, lonf, msk, gx[i], gy[i])
            pts3d.append(pt)
        
        pts3d = np.array(pts3d).reshape((len(pts3d), 3)).astype('float32')
        
        x, y = x + (np.ones((x.shape)) * (bbox[0].numpy() + bbox_offset[0].numpy())), y + (np.ones((y.shape)) * (bbox[1].numpy() + bbox_offset[1].numpy()))
        pts2d = np.stack((x, y), axis=1).astype('float32')

        # use EPnP to estimate the pose
        # success, rvec, tvec, _ = cv2.solvePnPRansac(pts3d, pts2d, intrinsics.numpy(), np.zeros((4, 1)), confidence=0.99, flags=cv2.SOLVEPNP_EPNP)
        # use Iterative pnp to estimate the pose
        success, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d, pts2d, intrinsics.numpy(), np.zeros((4, 1)), iterationsCount=100, reprojectionError=8.0, confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            rot = cv2.Rodrigues(rvec)[0]
            t = np.squeeze(tvec).reshape((-1, 1))
            rot = coordinate_change @ rot
            t = coordinate_change @ t
            estimated_pose = np.vstack((np.hstack((rot, t)), np.array([0, 0, 0, 1])))
            pts3d_inliers = pts3d[inliers]
            pts2d_inliers = pts2d[inliers]
        else:
            estimated_pose = np.eye(4)
            pts3d_inliers = None
            pts2d_inliers = None
        
        angular_distance = vis_utils.angler_distance(estimated_pose[:3, :3], pose[:3, :3].numpy())
        translation_error = np.sqrt((estimated_pose[:3, 3] - pose[:3, 3].numpy()) ** 2) # L2 norm for x, y, and z translation error specifically
          
        return estimated_pose, angular_distance, translation_error, pts3d, pts2d, pts3d_inliers, pts2d_inliers

    def cv2_pnp_batch(self, batch_size, latlon, bbox, bbox_offset, pose, verts, faces, seg_render, seg_mask, intrinsics):
        estimated_poses = np.zeros((batch_size, 4, 4))
        angular_distances = np.zeros((batch_size, 1))
        translation_errors = np.zeros((batch_size, 3))
        pts3ds = {'original': [], 'inliers': []}
        pts2ds = {'original': [], 'inliers': []}
        for batch_idx in range(batch_size):
            estimated_pose, angular_distance, translation_error, pts3d, pts2d, pts3d_inliers, pts2d_inliers = self.cv2_pnp(latlon, bbox[batch_idx], bbox_offset[batch_idx], pose[batch_idx], verts[batch_idx], faces[batch_idx], seg_render[batch_idx], seg_mask[batch_idx], intrinsics[batch_idx])
            estimated_poses[batch_idx] = estimated_pose
            angular_distances[batch_idx] = angular_distance
            translation_errors[batch_idx] = translation_error
            pts3ds['original'].append(pts3d)
            pts2ds['original'].append(pts2d)
            pts3ds['inliers'].append(pts3d_inliers)
            pts2ds['inliers'].append(pts2d_inliers)
        return estimated_poses, angular_distances, translation_errors, pts3ds, pts2ds

    def torch_render_batch(self, latlon, batch_size, verts, triangles, intrinsics, pose, image_size):

        h, w = image_size[1], image_size[2]
        # set mesh
        verts_rgb_colors = torch.tensor(latlon, dtype=torch.float32, device=device).unsqueeze(0)
        textures_batch = verts_rgb_colors.repeat(batch_size, 1, 1)
        textures = pytorch3d.renderer.TexturesVertex(verts_features=textures_batch)
        verts = verts.to(torch.float) # make sure the verts are float
        triangles = triangles.to(torch.float) # make sure the triangles are float
        mesh = Meshes(verts=[vert.to(device) for vert in verts], faces=[triangle.to(device) for triangle in triangles], textures=textures)

        # set camera
        R = pose[..., :3, :3].transpose(1, 2).to(device)
        T = pose[..., :3, 3].to(device)
        focal_length_batch = intrinsics[..., 0, 0]
        view_angle_batch = [np.degrees(2.0 * math.atan2(h/2.0, focal_length)) for focal_length in focal_length_batch]
        camera = pytorch3d.renderer.cameras.FoVPerspectiveCameras(device=device, R=R, T=T, znear=-100, zfar=3000, fov=view_angle_batch)

        # set the rasetrizer (with white background color)
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1, 1, 1))
        raster_settings = RasterizationSettings(image_size=(h, w))
        # add directional light
        lights = pytorch3d.renderer.lighting.DirectionalLights(ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), ), direction=((0, 0, 0), ), device=device)
        # set the shader
        softphong_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
                                    shader=SoftPhongShader(device=device, lights=lights, blend_params=blend_params))
        # render the target mesh providing the values of R and T
        rendered_image = softphong_renderer(meshes_world=mesh, cameras=camera)
        
        return rendered_image

    def save_output(self, iter, batch_size, ossicles_path, seg_render, output, target_pts3ds, predict_pts3ds, target_pts2ds, predict_pts2ds, gt_poses, target_poses, predict_poses, gt_pose_renders, target_pose_renders, predict_pose_renders):
        
        for idx in range(batch_size):
            name = pathlib.Path(ossicles_path[idx]).stem
            np.save(self.eval_output_dir / "pts3d" / "target" / "original" / f"{name}_{iter}_{idx}.npy", target_pts3ds['original'][idx])
            np.save(self.eval_output_dir / "pts3d" / "predict" / "original" / f"{name}_{iter}_{idx}.npy", predict_pts3ds['original'][idx])
            np.save(self.eval_output_dir / "pts3d" / "target" / "inliers" / f"{name}_{iter}_{idx}.npy", target_pts3ds['inliers'][idx])
            np.save(self.eval_output_dir / "pts3d" / "predict" / "inliers" / f"{name}_{iter}_{idx}.npy", predict_pts3ds['inliers'][idx])

            np.save(self.eval_output_dir / "pts2d" / "target" / "original" / f"{name}_{iter}_{idx}.npy", target_pts2ds['original'][idx])
            np.save(self.eval_output_dir / "pts2d" / "predict" / "original" / f"{name}_{iter}_{idx}.npy", predict_pts2ds['original'][idx])
            np.save(self.eval_output_dir / "pts2d" / "target" / "inliers" / f"{name}_{iter}_{idx}.npy", target_pts2ds['inliers'][idx])
            np.save(self.eval_output_dir / "pts2d" / "predict" / "inliers" / f"{name}_{iter}_{idx}.npy", predict_pts2ds['inliers'][idx])

            np.save(self.eval_output_dir / "poses" / "gt_poses" / f"{name}_{iter}_{idx}.npy", gt_poses[idx])
            np.save(self.eval_output_dir / "poses" / "target" / f"{name}_{iter}_{idx}.npy", target_poses[idx])
            np.save(self.eval_output_dir / "poses" / "predict" / f"{name}_{iter}_{idx}.npy", predict_poses[idx])

            # save mapping
            target_mapping = seg_render[idx].numpy()
            predict_mapping = output[idx].numpy()
            target_mapping_output = PIL.Image.fromarray((target_mapping * 255).astype('uint8'))
            predict_mapping_output = PIL.Image.fromarray((predict_mapping * 255).astype('uint8'))
            target_mapping_output.save(self.eval_output_dir / "mapping" / "target" / f"{name}_{iter}_{idx}.png")
            predict_mapping_output.save(self.eval_output_dir / "mapping" / "predict" / f"{name}_{iter}_{idx}.png")

            _, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
            axs[0].imshow(target_mapping)
            axs[1].imshow(predict_mapping)
            plt.savefig(self.eval_output_dir / "mapping" / "comparison" / f"{name}_{iter}_{idx}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # save renders
            gt_pose_render = np.clip(gt_pose_renders[idx].detach().cpu().numpy()[..., :3], 0, 1)
            target_pose_render = np.clip(target_pose_renders[idx].detach().cpu().numpy()[..., :3], 0, 1)
            predict_pose_render = np.clip(predict_pose_renders[idx].detach().cpu().numpy()[..., :3], 0, 1)
            gt_pose_render_output = PIL.Image.fromarray((gt_pose_render * 255).astype('uint8'))
            target_pose_render_output = PIL.Image.fromarray((target_pose_render * 255).astype('uint8'))
            predict_pose_render_output = PIL.Image.fromarray((predict_pose_render * 255).astype('uint8'))
            gt_pose_render_output.save(self.eval_output_dir / "renders" / "gt_pose" / f"{name}_{iter}_{idx}.png")
            target_pose_render_output.save(self.eval_output_dir / "renders" / "target" / f"{name}_{iter}_{idx}.png")
            predict_pose_render_output.save(self.eval_output_dir / "renders" / "predict" / f"{name}_{iter}_{idx}.png")
            
            _, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
            axs[0].imshow(gt_pose_render)
            axs[1].imshow(target_pose_render)
            axs[2].imshow(predict_pose_render)
            plt.savefig(self.eval_output_dir / "renders" / "comparison" / f"{name}_{iter}_{idx}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.model.parameters(), lr=self.config['lr'])
        scheduler = PolyLRScheduler(optimizer=optimizer, initial_lr=self.config['lr'], max_steps=self.config['epochs'])
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def compute_loss(self, output, target):
        container = {}
        loss_tot = 0
        for loss_func in self.config['loss'].keys():
            try:
                if loss_func == 'KLDivLoss': 
                    container[loss_func]= getattr(torch.nn, loss_func)(reduction='batchmean')(output, target) * self.config['loss'][loss_func]
                else:
                    container[loss_func]= getattr(torch.nn, loss_func)()(output, target) * self.config['loss'][loss_func]
            except AttributeError:
                if loss_func == 'GradientLoss': container[loss_func]= getattr(self, loss_func)()(output) * self.config['loss'][loss_func]
                else: container[loss_func]= getattr(self, loss_func)()(output, target) * self.config['loss'][loss_func]
            loss_tot += container[loss_func]
        container['loss_tot'] = loss_tot
        return container

    def training_step(self, batch, _):
        # opt = self.optimizers()
        _input, _target = batch['seg_image'], batch['seg_render']
        output = self.model(_input)
        assert torch.min(output) >= 0 and torch.max(output) <= 1, "range of output should be inside [0, 1]"
        container = self.compute_loss(output, _target)
        self.log('loss/train_loss', container['loss_tot'], on_step=False, on_epoch=True)
        # opt.zero_grad()
        # self.manual_backward(container['loss_tot'])
        # self.clip_gradients(opt, gradient_clip_val=10, gradient_clip_algorithm="norm")
        # opt.step()
        return container['loss_tot']
    
    def validation_step(self, batch, _):
        _input, _target = batch['seg_image'], batch['seg_render']
        output = self.model(_input)
        assert torch.min(output) >= 0 and torch.max(output) <= 1, "range of output should be inside [0, 1]"
        batch_size = output.shape[0]
        assert self.config['batch_size'] == batch_size, "batch size is not correct"

        # log loss
        container = self.compute_loss(output, _target)
        self.log('loss/val_loss', container['loss_tot'], prog_bar=True)
        for loss_func in self.config['loss'].keys(): 
            self.log(f'loss/val_loss_{loss_func}', container[loss_func], on_step=False, on_epoch=True)

        # log ssim and psnr
        self.val_accuracy = self.ssim(output, batch['seg_render'])
        self.log('acc/val_accuracy', self.val_accuracy, on_step=False, on_epoch=True)
        acc_psnr = self.psnr(output, batch['seg_render'])
        self.log('acc/psnr', acc_psnr, on_step=False, on_epoch=True)

        """
        # log the rotation and translation errors
        output = output.permute(0, 2, 3, 1).detach().cpu()
        batch['seg_render'] = batch['seg_render'].permute(0, 2, 3, 1).detach().cpu()
        batch['seg_mask'] = batch['seg_mask'].permute(0, 2, 3, 1).detach().cpu()
        pad_arr = torch.zeros((batch_size, int(self.crop_size), int(self.crop_size), 1))
        output = torch.concat((output, pad_arr), dim=-1)
        # make sure the output is within the segmentation mask
        output = output * batch['seg_mask']
        batch['seg_render'] = torch.concat((batch['seg_render'], pad_arr), dim=-1)
        _, predict_angular_distances, predict_translation_errors, _, _ = self.cv2_pnp_batch(batch_size, self.latlon, batch['bbox'].detach().cpu(), batch['bbox_offset'].detach().cpu(), batch['pose'].detach().cpu(), batch['ossicles'][:, :2454, :].detach().cpu(), batch['ossicles'][:, 2454:, :].detach().cpu(), output, batch['seg_mask'], batch['intrinsics'].detach().cpu())

        self.errors['predict_rot'].extend([d[0] for d in predict_angular_distances])
        self.errors['predict_tx'].extend([tx for tx in predict_translation_errors[:, 0]])
        self.errors['predict_ty'].extend([ty for ty in predict_translation_errors[:, 1]])
        self.errors['predict_tz'].extend([tz for tz in predict_translation_errors[:, 2]])

        self.log('predict/rot', np.median(np.array(self.errors['predict_rot'])))
        self.log('predict/tx', np.median(np.array(self.errors['predict_tx'])))
        self.log('predict/ty', np.median(np.array(self.errors['predict_ty'])))
        self.log('predict/tz', np.median(np.array(self.errors['predict_tz'])))
        """

    def test_step(self, batch, batch_idx):
        self.errors['predict_rot'].clear()
        self.errors['predict_tx'].clear()
        self.errors['predict_ty'].clear()
        self.errors['predict_tz'].clear()
        output = self.model(batch['seg_image'])
        assert torch.min(output) >= 0 and torch.max(output) <= 1, "range of output should be inside [0, 1]"
        batch_size = output.shape[0]
        output = output.permute(0, 2, 3, 1).detach().cpu()
        batch['seg_render'] = batch['seg_render'].permute(0, 2, 3, 1).detach().cpu()
        batch['seg_mask'] = batch['seg_mask'].permute(0, 2, 3, 1).detach().cpu()
        pad_arr = torch.zeros((batch_size, int(self.crop_size), int(self.crop_size), 1))
        output = torch.concat((output, pad_arr), dim=-1)
        # make sure the output is within the segmentation mask
        output = output * batch['seg_mask']
        batch['seg_render'] = torch.concat((batch['seg_render'], pad_arr), dim=-1)

        ossicles_verts = batch['ossicles'][:, :2454, :].detach().cpu()
        ossicles_faces = batch['ossicles'][:, 2454:, :].detach().cpu()
        gt_pose_renders = self.torch_render_batch(self.latlon, batch_size, ossicles_verts, ossicles_faces, batch['intrinsics'].detach().cpu(), batch['pose'].detach().cpu(), batch['image'].shape)
        target_poses, target_angular_distances, target_translation_errors, target_pts3ds, target_pts2ds = self.cv2_pnp_batch(batch_size, self.latlon, batch['bbox'].detach().cpu(), batch['bbox_offset'].detach().cpu(), batch['pose'].detach().cpu(), ossicles_verts, ossicles_faces, batch['seg_render'], batch['seg_mask'], batch['intrinsics'].detach().cpu())
        target_renders = self.torch_render_batch(self.latlon, batch_size, ossicles_verts, ossicles_faces, batch['intrinsics'].detach().cpu(), torch.from_numpy(target_poses), batch['image'].shape)
        predict_poses, predict_angular_distances, predict_translation_errors, predict_pts3ds, predict_pts2ds = self.cv2_pnp_batch(batch_size, self.latlon, batch['bbox'].detach().cpu(), batch['bbox_offset'].detach().cpu(), batch['pose'].detach().cpu(), ossicles_verts, ossicles_faces, output, batch['seg_mask'], batch['intrinsics'].detach().cpu())
        predict_renders = self.torch_render_batch(self.latlon, batch_size, ossicles_verts, ossicles_faces, batch['intrinsics'].detach().cpu(), torch.from_numpy(predict_poses), batch['image'].shape)
        
        self.errors['target_rot'].extend([d[0] for d in target_angular_distances])
        self.errors['target_tx'].extend([tx for tx in target_translation_errors[:, 0]])
        self.errors['target_ty'].extend([ty for ty in target_translation_errors[:, 1]])
        self.errors['target_tz'].extend([tz for tz in target_translation_errors[:, 2]])
        self.errors['predict_rot'].extend([d[0] for d in predict_angular_distances])
        self.errors['predict_tx'].extend([tx for tx in predict_translation_errors[:, 0]])
        self.errors['predict_ty'].extend([ty for ty in predict_translation_errors[:, 1]])
        self.errors['predict_tz'].extend([tz for tz in predict_translation_errors[:, 2]])

        # save the output
        self.save_output(batch_idx, batch_size, batch['ossicles_path'], batch['seg_render'], output, target_pts3ds, predict_pts3ds, target_pts2ds, predict_pts2ds, batch['pose'].detach().cpu(), target_poses, predict_poses, gt_pose_renders, target_renders, predict_renders)

    def on_test_epoch_end(self):
        # create the box plot for target and neural network output
        utils.create_box_plot(self.errors['target_rot'], self.errors['target_tx'], self.errors['target_ty'], self.errors['target_tz'], self.eval_output_dir / "plots" / "box_plot" / "target" / "target_box_plot.png")
        utils.create_box_plot(self.errors['predict_rot'], self.errors['predict_tx'], self.errors['predict_ty'], self.errors['predict_tz'], self.eval_output_dir / "plots" / "box_plot" / "predict" / "predict_box_plot.png")
        np.save(self.eval_output_dir / "plots" / "box_plot" / "target" / "target_rot_errors.npy", self.errors['target_rot'])
        np.save(self.eval_output_dir / "plots" / "box_plot" / "target" / "target_tx_errors.npy", self.errors['target_tx'])
        np.save(self.eval_output_dir / "plots" / "box_plot" / "target" / "target_ty_errors.npy", self.errors['target_ty'])
        np.save(self.eval_output_dir / "plots" / "box_plot" / "target" / "target_tz_errors.npy", self.errors['target_tz'])
        np.save(self.eval_output_dir / "plots" / "box_plot" / "predict" / "predict_rot_errors.npy", self.errors['predict_rot'])
        np.save(self.eval_output_dir / "plots" / "box_plot" / "predict" / "predict_tx_errors.npy", self.errors['predict_tx'])
        np.save(self.eval_output_dir / "plots" / "box_plot" / "predict" / "predict_ty_errors.npy", self.errors['predict_ty'])
        np.save(self.eval_output_dir / "plots" / "box_plot" / "predict" / "predict_tz_errors.npy", self.errors['predict_tz'])
        tqdm.write(f"\nsaved the evaliation results\n")