import os
import trimesh
import numpy as np
import cv2
import torch
import numpy as np
import albumentations as A
from ...tools import utils, vis_utils, paths

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
else: device = torch.device("cpu")

data_dir = paths.PKG_ROOT.parent / "data" / "dataset"

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, augmentations, augment_image, auxiliary_info=False):
        super(CustomDataset, self).__init__()
        self.dataset = dataset
        self.augment_image = augment_image
        self.auxiliary_info = auxiliary_info
        self.augmentations = augmentations

    def cv2_epnp(self, xyxy, offset, pose, ossicles, seg_render, seg_mask, intrinsics):
        
        latlon = utils.load_latitude_longitude(data_dir.parent / "latlon" /"ossiclesCoordinateMapping2.json")
        # coordinate system change (x, y, z) -> (-x, -y, z)
        coordinate_change = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        
        mesh = trimesh.Trimesh(vertices=ossicles[:2454], faces=ossicles[2454:], process=False)
        # get the color mask
        color_mask = seg_render[..., :3]
        binary_mask = utils.color2binary_mask(color_mask)
        if np.max(seg_mask) > 1: seg_mask = seg_mask / 255
        # assert (binary_mask == seg_mask).all(), "bianry mask and seg mask are not the same"
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
        
        x, y = x + (np.ones((x.shape)) * (xyxy[0] + offset[0])), y + (np.ones((y.shape)) * (xyxy[1] + offset[1]))
        pts2d = np.stack((x, y), axis=1).astype('float32')

        # use EPnP to predict the pose
        _, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d, pts2d, intrinsics, np.zeros((4, 1)), confidence=0.99, flags=cv2.SOLVEPNP_EPNP)

        # if len(inliers) != np.sum(seg_mask[..., 0]): print(f"inliers: {len(inliers)} and total pixels (seg mask): {np.sum(seg_mask[..., 0])} are not equal")

        # convert the rotation vector to rotation matrix and the translation vector to translation matrix
        rot = cv2.Rodrigues(rvec)[0]
        t = np.squeeze(tvec).reshape((-1, 1))
        
        rot = coordinate_change @ rot
        t = coordinate_change @ t
        estimated_pose = np.vstack((np.hstack((rot, t)), np.array([0, 0, 0, 1])))
        
        # print("\n")
        # print(estimated_pose)
        # print(pose)
        
        angular_distance = vis_utils.angler_distance(estimated_pose[:3, :3], pose[:3, :3])
        translation_error = np.linalg.norm(estimated_pose[:3, 3] - pose[:3, 3])
        
        assert angular_distance < 2, "angular distance is too large"
        assert translation_error < 50, "translation error is too large"

        return estimated_pose
        
    def __getitem__(self, index):
        # tic = time.perf_counter()
        container = np.load(self.dataset / f"{index}.npz")
        
        bbox = container['bbox']

        image = container['image']
        if self.augment_image:
            bright_contrast = A.transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, p=0.5)
            image = bright_contrast(image=image)['image']

        mask = container['mask']
        render = container['render']

        seg_image = image * mask
        seg_render = render * mask[..., :1]
        
        x1, y1, x2, y2 = bbox
        seg_image = seg_image[y1:y2, x1:x2]
        seg_mask = mask[y1:y2, x1:x2]
        seg_render = seg_render[y1:y2, x1:x2][..., :2]
        
        assert np.max(seg_image) <= 1, "max value of seg_image is greater than 1"
        assert np.max(seg_mask) <= 1, "max value of seg_mask is greater than 1"
        assert np.max(seg_render) <= 1, "max value of seg_render is greater than 1"

        # estimate_pose = self.cv2_epnp(bbox, container['bbox_offset'], container['pose'], container['ossicles'], seg_render, seg_mask, container['intrinsics'])
        # perform data augmentations
        if self.augmentations is not None:
            augmented = self.augmentations(image=seg_image, masks=[seg_mask, seg_render])
            seg_image = augmented['image']
            seg_mask = augmented['masks'][0]
            seg_render = augmented['masks'][1]

        # convert to tensor size (C, H, W)
        seg_image = torch.from_numpy(seg_image).permute((2, 0, 1)).float()
        seg_mask = torch.from_numpy(seg_mask).permute((2, 0, 1)).float()
        seg_render = torch.from_numpy(seg_render).permute((2, 0, 1)).float()

        if self.auxiliary_info:      
            auxiliary_output = {'ossicles_path': container['ossicles_path'].item(),
                    'ossicles': container['ossicles'],
                    'intrinsics': container['intrinsics'],
                    'pose': container['pose'],
                    'bbox': container['bbox'],
                    'bbox_offset': container['bbox_offset'],
                    'image': container['image'],
                    'seg_image': seg_image,
                    # 'seg_image': seg_mask, # try with just a binary mask to do image synthesis
                    'seg_mask': seg_mask,
                    'seg_render': seg_render}
            return auxiliary_output
        else:
            simple_output = {
                'seg_image': seg_image, 
                # 'seg_image': seg_mask, # try with just a binary mask to do image synthesis
                'seg_render': seg_render}
            return simple_output
        
    def __len__(self):
        return len(os.listdir(self.dataset))
    
class LoadDataset():
    def __init__(self, config):
        self.config = config
        if config['fix_zoom']: dataset = data_dir / f"fold_{self.config['fold_num']}_fix_zoom_{config['image_size'][0]}_{config['image_size'][1]}"
        else: dataset = data_dir / f"fold_{self.config['fold_num']}_zoom_in_out_{config['image_size'][0]}_{config['image_size'][1]}"
        self.train_dataset = dataset / 'train'
        self.valid_dataset = dataset / 'valid'
        self.test_dataset = dataset / 'test'
        self.simple_test_dataset = dataset / 'simple_test'

    def create_dataloaders(self):
        # data augmentations
        augmentations = A.Compose([# A.HorizontalFlip(p=0.5), # A.VerticalFlip(p=0.5),
                                A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                                A.ShiftScaleRotate(scale_limit=[0, 0], rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5)], p=0.5)
        
        # Creating dataset     
        train_dataset = CustomDataset(self.train_dataset, augmentations=augmentations, augment_image=False)
        valid_dataset = CustomDataset(self.valid_dataset, augmentations=augmentations, augment_image=False)
        test_dataset = CustomDataset(self.test_dataset, augmentations=None, augment_image=False, auxiliary_info=True)
        simple_test_dataset = CustomDataset(self.simple_test_dataset, augmentations=None, augment_image=False, auxiliary_info=True)

        # Creating dataloader, and for windows it is safer to set num_workers=0
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=3, persistent_workers=True, pin_memory=True, drop_last=True)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=3, persistent_workers=True, pin_memory=True, drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=3, drop_last=True)
        simple_test_dataloader = torch.utils.data.DataLoader(simple_test_dataset, batch_size=len(simple_test_dataset), shuffle=False, num_workers=0, drop_last=True)

        return train_dataloader, valid_dataloader, test_dataloader, simple_test_dataloader