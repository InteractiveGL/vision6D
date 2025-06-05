import os
import cv2
import math
import traceback
from tqdm import tqdm
import torch.nn as nn
import torch
import trimesh
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from .dataset import LoadDataset
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from ignite.metrics import PSNR, SSIM
import PIL.Image

import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams, SoftPhongShader)

import segmentation_models_pytorch as smp
from .loss import SSIMLoss, HistogramLoss, GradientLoss, CrossCorrLoss
from ..tools import paths, utils, vis_utils
from torchsummary import summary

# fix the all seeds possible
seed_number = 42
random.seed(seed_number)
np.random.seed(seed_number)
torch.manual_seed(seed_number)
torch.cuda.manual_seed(seed_number)
torch.cuda.manual_seed_all(seed_number)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    Tensor = torch.cuda.FloatTensor
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
else: 
    device = torch.device("cpu")
    Tensor = torch.FloatTensor

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

class Discriminator(nn.Module):
    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        
        self.desc_model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.desc_model(img_flat)

        return validity

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

class GAN():
    def __init__(self, config, logs_dir, output_dir) -> None:
        self.config = config
        self.logs_dir = logs_dir
        self.output_dir = output_dir
        # early stopping
        self.early_stopping = False
        self.patience = 400
        self.delta = 0
        self.counter = 0

        if self.config['image_size'] == [1080, 1920]: self.crop_size = 512
        elif self.config['image_size'] == [540, 960]: self.crop_size = 256
        elif self.config['image_size'] == [270, 480]: self.crop_size = 128
        elif self.config['image_size'] == [108, 192]: self.crop_size = 64

        # self.model = unet.UNet().to(device)
        self.model = smp.UnetPlusPlus(encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=2,                      # model output channels (number of classes in your dataset)
                        activation='sigmoid',
                    ).to(device)
        # freeze the encoder
        freeze_encoder(self.model)
        
        self.discriminator = Discriminator((2, self.crop_size, self.crop_size)).to(device)
        self.adversarial_loss = torch.nn.BCELoss()

        print(summary(self.model, (3, self.crop_size, self.crop_size)))
        print(summary(self.discriminator, (2, self.crop_size, self.crop_size)))
        
        self.optimizer_G = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.config['lr'])

        self.scheduler_G = PolyLRScheduler(optimizer=self.optimizer_G, initial_lr=self.config['lr'], max_steps=self.config['epochs'])
        self.scheduler_D = PolyLRScheduler(optimizer=self.optimizer_D, initial_lr=self.config['lr_d'], max_steps=self.config['epochs'])

        self.save_top_k = 5
        self.best_models = []
        self.loss_save = 1e+5
        for loss_func in self.config['loss'].keys():
            try: setattr(self, f'{loss_func}', getattr(torch.nn, loss_func))
            except AttributeError: 
                if loss_func == 'SSIMLoss': setattr(self, f'{loss_func}', SSIMLoss)
                elif loss_func == 'HistLoss': setattr(self, f'{loss_func}', HistogramLoss)
                elif loss_func == 'GradientLoss': setattr(self, f'{loss_func}', GradientLoss)
                elif loss_func == "CrossCorrLoss": setattr(self, f'{loss_func}', CrossCorrLoss)
        
        self.psnr = PSNR(data_range=1.0)
        self.ssim = SSIM(data_range=1.0)
        
        self.writer = SummaryWriter(self.logs_dir)

        self.errors = {'target_rot': [], 'target_tx': [], 'target_ty': [], 'target_tz': [], 'predict_rot': [], 'predict_tx': [], 'predict_ty': [], 'predict_tz': []}
        self.latlon = utils.load_latitude_longitude(paths.PKG_ROOT.parent / "data" / "latlon" /"ossiclesCoordinateMapping2.json")

    def cv2_pnp(self, latlon, pose, verts, faces, seg_render, intrinsics):
        # coordinate system change (x, y, z) -> (-x, -y, z)
        coordinate_change = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        mesh = trimesh.Trimesh(vertices=verts.numpy(), faces=faces.numpy(), process=False)
        # get the color mask
        color_mask = seg_render[..., :3]
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
        
        # x, y = x + (np.ones((x.shape)) * (bbox[0].numpy() + bbox_offset[0].numpy())), y + (np.ones((y.shape)) * (bbox[1].numpy() + bbox_offset[1].numpy()))
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

    def cv2_pnp_batch(self, batch_size, latlon, pose, verts, faces, seg_render, intrinsics):
        estimated_poses = np.zeros((batch_size, 4, 4))
        angular_distances = np.zeros((batch_size, 1))
        translation_errors = np.zeros((batch_size, 3))
        pts3ds = {'original': [], 'inliers': []}
        pts2ds = {'original': [], 'inliers': []}
        for iter in range(batch_size):
            estimated_pose, angular_distance, translation_error, pts3d, pts2d, pts3d_inliers, pts2d_inliers = self.cv2_pnp(latlon, pose[iter], verts[iter], faces[iter], seg_render[iter], intrinsics[iter])
            estimated_poses[iter] = estimated_pose
            angular_distances[iter] = angular_distance
            translation_errors[iter] = translation_error
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
        T = (pose[..., :3, 3]).to(device)
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
  
    def compute_loss(self, output, target):
        container = {}
        loss_tot = 0
        for loss_func in self.config['loss'].keys():
            try:
                if loss_func == 'KLDivLoss': 
                    container[loss_func]= getattr(torch.nn, loss_func)(reduction='batchmean')(torch.nn.functional.log_softmax(output, dim=1), target) * self.config['loss'][loss_func]
                else:
                    container[loss_func]= getattr(torch.nn, loss_func)()(output, target) * self.config['loss'][loss_func]
            except AttributeError:
                if loss_func == 'GradientLoss': container[loss_func]= getattr(self, loss_func)()(output) * self.config['loss'][loss_func]
                else: container[loss_func]= getattr(self, loss_func)()(output, target) * self.config['loss'][loss_func]
            loss_tot += container[loss_func]
        container['loss_tot'] = loss_tot
        return container
    
    def compute_accuracy(self, output, target):
        self.psnr.update((output, target))
        self.ssim.update((output, target))
        res_psnr = self.psnr.compute()
        res_ssim = self.ssim.compute()
        return res_psnr, res_ssim
        
    def train_per_epoch(self, real, fake, train_dataloader, epoch):
        torch.cuda.empty_cache()
        self.model.train()
        self.discriminator.train()

        self.scheduler_G.step(epoch)
        self.scheduler_D.step(epoch)

        gen_loss = 0.0
        disc_loss = 0.0
        for _, data in enumerate(train_dataloader):

            # for the generator
            self.optimizer_G.zero_grad()
            output = self.model(data['seg_image'].to(device))
            assert torch.min(output) >= 0 and torch.max(output) <= 1, "output range should be inside [0, 1]"
            container = self.compute_loss(output, data['seg_render'].to(device))
            gen_tot = container['loss_tot']
            gen_tot.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer_G.step()
            gen_loss += gen_tot.detach().item()

            # for the discriminator
            self.optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            # pad the output with zeros

            real_loss = self.adversarial_loss(self.discriminator(data['seg_render'].to(device)), real)
            fake_loss = self.adversarial_loss(self.discriminator(output.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            self.optimizer_D.step()
            disc_loss += d_loss.detach().item()

        gen_loss_mean = gen_loss / len(train_dataloader)
        disc_loss_mean = disc_loss / len(train_dataloader)
        self.writer.add_scalar(f'train/GEN_Loss/loss_mean', gen_loss_mean, epoch)
        self.writer.add_scalar(f'train/DISC_Loss/loss_mean', disc_loss_mean, epoch)

        if epoch % 40 == 0:
            pad_arr = torch.zeros((data['seg_image'].shape[0], 1, int(self.crop_size), int(self.crop_size))).to(device)
            self.writer.add_images(f"train/input", data['seg_image'].to(device), epoch, dataformats='NCHW')
            self.writer.add_images(f"train/target", torch.concat((data['seg_render'].to(device), pad_arr), dim=1), epoch, dataformats='NCHW')
            self.writer.add_images(f"train/output", torch.concat((output, pad_arr), dim=1), epoch, dataformats='NCHW')
                    
    def eval_per_epoch(self, valid_dataloader, test_dataloader, epoch):
        torch.cuda.empty_cache()
        self.model.eval()
        for loss_func in self.config['loss'].keys(): setattr(self, f'{loss_func}_tot', 0.0)
            
        loss_tot = 0.0
        psnr_tot = 0.0
        ssim_tot = 0.0
        
        # for validation process, we don't need to compute the gradient
        with torch.no_grad(): 
            for _, data in enumerate(valid_dataloader):
                assert data['seg_image'].shape == (self.config['batch_size'], 3, self.crop_size, self.crop_size), "seg_image shape is not correct"
                output = self.model(data['seg_image'].to(device))
                assert torch.min(output) >= 0 and torch.max(output) <= 1, "output range should be inside [0, 1]"
                container = self.compute_loss(output, data['seg_render'].to(device))
                loss_tot += container['loss_tot'].detach().item()
                
                for loss_func in self.config['loss'].keys(): 
                    loss = getattr(self, f'{loss_func}_tot')
                    loss += container[loss_func]
                    setattr(self, f'{loss_func}_tot', loss)
                    
                # compute the accuracy
                res_psnr, res_ssim = self.compute_accuracy(output, data['seg_render'].to(device))
                psnr_tot += res_psnr
                ssim_tot += res_ssim
                
        self.loss_mean = loss_tot / len(valid_dataloader)
        self.psnr_mean = psnr_tot / len(valid_dataloader)
        self.ssim_mean = ssim_tot / len(valid_dataloader)
        
        self.writer.add_scalar(f'valid/Loss/loss_mean', self.loss_mean, epoch)
        for loss_func in self.config['loss'].keys(): self.writer.add_scalar(f'valid/Loss/{loss_func}_mean', getattr(self, f'{loss_func}_tot') / len(valid_dataloader), epoch)
        self.writer.add_scalar(f'valid/Accuracy/psnr_mean', self.psnr_mean, epoch)
        self.writer.add_scalar(f'valid/Accuracy/ssim_mean', self.ssim_mean, epoch)          
        tqdm.write(f"loss mean: {self.loss_mean} | psnr mean: {self.psnr_mean} | ssim mean: {self.ssim_mean}")
    
        if self.loss_mean < self.loss_save:
            self.counter = 0
            self.loss_save = self.loss_mean
            if len(self.best_models) >= self.save_top_k: 
                os.remove(self.best_models.pop(0))
            self.save_checkpoint(f'best_{epoch}', epoch)
            self.best_models.append(self.output_dir / "checkpoints" / f"best_{epoch}_checkpoint.pth")
            if epoch > 100: 
                self.test_per_epoch(test_dataloader, epoch)
                pad_arr = torch.zeros((data['seg_image'].shape[0], 1, int(self.crop_size), int(self.crop_size))).to(device)
                self.writer.add_images(f"valid/input", data['seg_image'].to(device), epoch, dataformats='NCHW')
                self.writer.add_images(f"valid/target", torch.concat((data['seg_render'].to(device), pad_arr), dim=1), epoch, dataformats='NCHW')
                self.writer.add_images(f"valid/output", torch.concat((output, pad_arr), dim=1), epoch, dataformats='NCHW')

        elif self.loss_mean > self.loss_save + self.delta:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stopping = True
    
    def test_per_epoch(self, test_dataloader, epoch):
        with torch.no_grad():  
            for _, data in enumerate(test_dataloader):
                output = self.model(data['seg_image'].to(device))
                assert torch.min(output) >= 0 and torch.max(output) <= 1, "output range should be inside [0, 1]"
                batch_size = output.shape[0]
                # make sure the dimension now is (batch_size, 512, 512, 3)
                output = output.permute(0, 2, 3, 1).detach().cpu()
                data['seg_render'] = data['seg_render'].permute(0, 2, 3, 1).detach().cpu()
                data['seg_mask'] = data['seg_mask'].permute(0, 2, 3, 1).detach().cpu()
                pad_arr = torch.zeros((batch_size, int(self.crop_size), int(self.crop_size), 1))
                output = torch.concat((output, pad_arr), dim=-1)
                # make sure the output is within the segmentation mask
                output = output * data['seg_mask']
                offsets_x = data['bbox'][:, 0].numpy() + data['bbox_offset'][:, 0].numpy()
                offsets_y = data['bbox'][:, 1].numpy() + data['bbox_offset'][:, 1].numpy()
                output_upscale = self.upscale_image(batch_size, output, offsets_x, offsets_y)
                _, predict_angular_distances, predict_translation_errors, _, _ = self.cv2_pnp_batch(batch_size, self.latlon, data['pose'], data['ossicles'][:, :2454, :], data['ossicles'][:, 2454:, :], output_upscale, data['intrinsics'])

                self.errors['predict_rot'].extend([d[0] for d in predict_angular_distances])
                self.errors['predict_tx'].extend([tx for tx in predict_translation_errors[:, 0]])
                self.errors['predict_ty'].extend([ty for ty in predict_translation_errors[:, 1]])
                self.errors['predict_tz'].extend([tz for tz in predict_translation_errors[:, 2]])

        self.writer.add_scalar(f'test/metric/predict_rot', np.median(np.array(self.errors['predict_rot'])), epoch)
        self.writer.add_scalar(f'test/metric/predict_tx', np.median(np.array(self.errors['predict_tx'])), epoch)
        self.writer.add_scalar(f'test/metric/predict_ty', np.median(np.array(self.errors['predict_ty'])), epoch)
        self.writer.add_scalar(f'test/metric/predict_tz', np.median(np.array(self.errors['predict_tz'])), epoch)

    def save_output(self, batch_idx, batch_size, ossicles_path, seg_image, seg_render, output, target_pts3ds, predict_pts3ds, target_pts2ds, predict_pts2ds, gt_poses, target_poses, predict_poses, gt_pose_renders, target_pose_renders, predict_pose_renders):
        
        for idx in range(batch_size):
            name = pathlib.Path(ossicles_path[idx]).stem
            np.save(self.output_dir / "pts3d" / "target" / "original" / f"{name}_{batch_idx}_{idx}.npy", target_pts3ds['original'][idx])
            np.save(self.output_dir / "pts3d" / "predict" / "original" / f"{name}_{batch_idx}_{idx}.npy", predict_pts3ds['original'][idx])
            np.save(self.output_dir / "pts3d" / "target" / "inliers" / f"{name}_{batch_idx}_{idx}.npy", target_pts3ds['inliers'][idx])
            np.save(self.output_dir / "pts3d" / "predict" / "inliers" / f"{name}_{batch_idx}_{idx}.npy", predict_pts3ds['inliers'][idx])

            np.save(self.output_dir / "pts2d" / "target" / "original" / f"{name}_{batch_idx}_{idx}.npy", target_pts2ds['original'][idx])
            np.save(self.output_dir / "pts2d" / "predict" / "original" / f"{name}_{batch_idx}_{idx}.npy", predict_pts2ds['original'][idx])
            np.save(self.output_dir / "pts2d" / "target" / "inliers" / f"{name}_{batch_idx}_{idx}.npy", target_pts2ds['inliers'][idx])
            np.save(self.output_dir / "pts2d" / "predict" / "inliers" / f"{name}_{batch_idx}_{idx}.npy", predict_pts2ds['inliers'][idx])

            np.save(self.output_dir / "poses" / "gt_poses" / f"{name}_{batch_idx}_{idx}.npy", gt_poses[idx])
            np.save(self.output_dir / "poses" / "target" / f"{name}_{batch_idx}_{idx}.npy", target_poses[idx])
            np.save(self.output_dir / "poses" / "predict" / f"{name}_{batch_idx}_{idx}.npy", predict_poses[idx])

            # save input images
            input_image = seg_image[idx]
            input_image_output = PIL.Image.fromarray((input_image * 255).astype('uint8'))
            input_image_output.save(self.output_dir / "inputs" / f"{name}_{batch_idx}_{idx}.png")

            # save mapping
            target_mapping = seg_render[idx]
            predict_mapping = output[idx]
            target_mapping_output = PIL.Image.fromarray((target_mapping * 255).astype('uint8'))
            predict_mapping_output = PIL.Image.fromarray((predict_mapping * 255).astype('uint8'))
            target_mapping_output.save(self.output_dir / "mapping" / "target" / f"{name}_{batch_idx}_{idx}.png")
            predict_mapping_output.save(self.output_dir / "mapping" / "predict" / f"{name}_{batch_idx}_{idx}.png")

            _, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
            axs[0].imshow(target_mapping)
            axs[1].imshow(predict_mapping)
            plt.savefig(self.output_dir / "mapping" / "comparison" / f"{name}_{batch_idx}_{idx}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # save renders
            gt_pose_render = np.clip(gt_pose_renders[idx].detach().cpu().numpy()[..., :3], 0, 1)
            target_pose_render = np.clip(target_pose_renders[idx].detach().cpu().numpy()[..., :3], 0, 1)
            predict_pose_render = np.clip(predict_pose_renders[idx].detach().cpu().numpy()[..., :3], 0, 1)
            gt_pose_render_output = PIL.Image.fromarray((gt_pose_render * 255).astype('uint8'))
            target_pose_render_output = PIL.Image.fromarray((target_pose_render * 255).astype('uint8'))
            predict_pose_render_output = PIL.Image.fromarray((predict_pose_render * 255).astype('uint8'))
            gt_pose_render_output.save(self.output_dir / "renders" / "gt_pose" / f"{name}_{batch_idx}_{idx}.png")
            target_pose_render_output.save(self.output_dir / "renders" / "target" / f"{name}_{batch_idx}_{idx}.png")
            predict_pose_render_output.save(self.output_dir / "renders" / "predict" / f"{name}_{batch_idx}_{idx}.png")
            
            _, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
            axs[0].imshow(gt_pose_render)
            axs[1].imshow(target_pose_render)
            axs[2].imshow(predict_pose_render)
            plt.savefig(self.output_dir / "renders" / "comparison" / f"{name}_{batch_idx}_{idx}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_checkpoint(self, name, epoch):
        checkpoint = {'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer_G.state_dict(),
                    'lr_scheduler_state_dict': self.scheduler_G.state_dict(),
                    'loss': self.loss_mean,
                    'psnr': self.psnr_mean,
                    'ssim': self.ssim_mean}
        torch.save(checkpoint, self.output_dir / "checkpoints" / f'{name}_checkpoint.pth')
        
    def resume_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.loss_save = checkpoint['loss']
        self.train_net(checkpoint['epoch'] + 1)
        
    def load_checkpoint(self, checkpoint_path, sample):
        checkpoint = torch.load(checkpoint_path)
        print(f"Loading checkpoint path {checkpoint_path} from epoch {checkpoint['epoch']} with loss {checkpoint['loss']}")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.eval_net(sample)

    def train_net(self, start_epoch=0):
        torch.cuda.empty_cache()
        real = Variable(Tensor(self.config['batch_size'], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(self.config['batch_size'], 1).fill_(0.0), requires_grad=False)
        train_dataloader, valid_dataloader, _, simple_test_dataloader = LoadDataset(self.config).create_dataloaders()
        try: 
            for epoch in tqdm(range(start_epoch, self.config['epochs']), position=0, total=self.config['epochs']-start_epoch):
                tqdm.write(f"\nCurrent epoch {epoch} | lr is {self.optimizer_G.state_dict()['param_groups'][0]['lr']} | ")
                self.train_per_epoch(real, fake, train_dataloader, epoch)
                self.eval_per_epoch(valid_dataloader, simple_test_dataloader, epoch)
                if self.early_stopping: tqdm.write(f"\nEarly stopping at epoch {epoch}"); break
        except Exception as e:
            if e != KeyboardInterrupt:
                tqdm.write(f"\nException {e} at epoch {epoch}. Exiting!")
                tqdm.write(f"{traceback.format_exc()}")
                # import pdb; pdb.set_trace()
        self.save_checkpoint('last', epoch)
        tqdm.write(f"\nSaved the last checkpoint at epoch {epoch}")
        return self.best_models[-1]

    def upscale_image(self, batch_size, image, offsets_x, offsets_y):
        color_mask_batch = image[..., :3].numpy()
        binary_mask = np.zeros(color_mask_batch.shape[:-1], dtype=np.uint8)
        binary_mask[(color_mask_batch == [0., 0., 0.]).all(axis=-1) == 0] = 1
        b, y, x = np.where(binary_mask == 1)
        x_upscale = x + offsets_x[b]
        y_upscale = y + offsets_y[b]
        upscaled_image = np.zeros((batch_size, *self.config['image_size'], 3))
        upscaled_image[b, y_upscale.astype(np.int16), x_upscale.astype(np.int16)] = color_mask_batch[b, y, x]
        return upscaled_image

    def eval_net(self, sample):
        # load the test dataloader
        if not sample: _, _, dataloader, _ = LoadDataset(self.config).create_dataloaders()
        else: _, _, _, dataloader = LoadDataset(self.config).create_dataloaders()
        self.model.eval()
        # Disable gradient computation for efficiency in evaluation
        with torch.no_grad():  
            for batch_idx, data in enumerate(dataloader):
                output = self.model(data['seg_image'].to(device))
                assert torch.min(output) >= 0 and torch.max(output) <= 1, "output range should be inside [0, 1]"
                batch_size = output.shape[0]
                # make sure the dimension now is (batch_size, 512, 512, 3)
                output = output.permute(0, 2, 3, 1).detach().cpu()
                data['seg_image'] = data['seg_image'].permute(0, 2, 3, 1).detach().cpu()
                data['seg_render'] = data['seg_render'].permute(0, 2, 3, 1).detach().cpu()
                data['seg_mask'] = data['seg_mask'].permute(0, 2, 3, 1).detach().cpu()
                pad_arr = torch.zeros((batch_size, int(self.crop_size), int(self.crop_size), 1))
                output = torch.concat((output, pad_arr), dim=-1)
                # make sure the output is within the segmentation mask
                output = output * data['seg_mask']
                data['seg_render'] = torch.concat((data['seg_render'], pad_arr), dim=-1)
                
                offsets_x = data['bbox'][:, 0].numpy() + data['bbox_offset'][:, 0].numpy()
                offsets_y = data['bbox'][:, 1].numpy() + data['bbox_offset'][:, 1].numpy()
                output_upscale = self.upscale_image(batch_size, output, offsets_x, offsets_y)
                seg_render_upscale = self.upscale_image(batch_size, data['seg_render'], offsets_x, offsets_y)
                
                gt_pose_renders = self.torch_render_batch(self.latlon, batch_size, data['ossicles'][:, :2454, :], data['ossicles'][:, 2454:, :], data['intrinsics'], data['pose'], data['image'].shape)
                target_poses, target_angular_distances, target_translation_errors, target_pts3ds, target_pts2ds = self.cv2_pnp_batch(batch_size, self.latlon, data['pose'], data['ossicles'][:, :2454, :], data['ossicles'][:, 2454:, :], seg_render_upscale, data['intrinsics'])
                target_renders = self.torch_render_batch(self.latlon, batch_size, data['ossicles'][:, :2454, :], data['ossicles'][:, 2454:, :], data['intrinsics'], torch.from_numpy(target_poses), data['image'].shape)
                predict_poses, predict_angular_distances, predict_translation_errors, predict_pts3ds, predict_pts2ds = self.cv2_pnp_batch(batch_size, self.latlon, data['pose'], data['ossicles'][:, :2454, :], data['ossicles'][:, 2454:, :], output_upscale, data['intrinsics'])
                predict_renders = self.torch_render_batch(self.latlon, batch_size, data['ossicles'][:, :2454, :], data['ossicles'][:, 2454:, :], data['intrinsics'], torch.from_numpy(predict_poses), data['image'].shape)
                
                self.errors['target_rot'].extend([d[0] for d in target_angular_distances])
                self.errors['target_tx'].extend([tx for tx in target_translation_errors[:, 0]])
                self.errors['target_ty'].extend([ty for ty in target_translation_errors[:, 1]])
                self.errors['target_tz'].extend([tz for tz in target_translation_errors[:, 2]])
                self.errors['predict_rot'].extend([d[0] for d in predict_angular_distances])
                self.errors['predict_tx'].extend([tx for tx in predict_translation_errors[:, 0]])
                self.errors['predict_ty'].extend([ty for ty in predict_translation_errors[:, 1]])
                self.errors['predict_tz'].extend([tz for tz in predict_translation_errors[:, 2]])

                # save the output
                self.save_output(batch_idx, batch_size, data['ossicles_path'], data['seg_image'].numpy(), seg_render_upscale, output_upscale, target_pts3ds, predict_pts3ds, target_pts2ds, predict_pts2ds, data['pose'], target_poses, predict_poses, gt_pose_renders, target_renders, predict_renders)

        # create the box plot for target and neural network output
        utils.create_box_plot(self.errors['target_rot'], self.errors['target_tx'], self.errors['target_ty'], self.errors['target_tz'], self.output_dir / "plots" / "box_plot" / "target" / "target_box_plot.png")
        utils.create_box_plot(self.errors['predict_rot'], self.errors['predict_tx'], self.errors['predict_ty'], self.errors['predict_tz'], self.output_dir / "plots" / "box_plot" / "predict" / "predict_box_plot.png")
        np.save(self.output_dir / "plots" / "box_plot" / "target" / "target_rot_errors.npy", self.errors['target_rot'])
        np.save(self.output_dir / "plots" / "box_plot" / "target" / "target_tx_errors.npy", self.errors['target_tx'])
        np.save(self.output_dir / "plots" / "box_plot" / "target" / "target_ty_errors.npy", self.errors['target_ty'])
        np.save(self.output_dir / "plots" / "box_plot" / "target" / "target_tz_errors.npy", self.errors['target_tz'])
        np.save(self.output_dir / "plots" / "box_plot" / "predict" / "predict_rot_errors.npy", self.errors['predict_rot'])
        np.save(self.output_dir / "plots" / "box_plot" / "predict" / "predict_tx_errors.npy", self.errors['predict_tx'])
        np.save(self.output_dir / "plots" / "box_plot" / "predict" / "predict_ty_errors.npy", self.errors['predict_ty'])
        np.save(self.output_dir / "plots" / "box_plot" / "predict" / "predict_tz_errors.npy", self.errors['predict_tz'])
        tqdm.write(f"\nsaved the evaliation results\n")