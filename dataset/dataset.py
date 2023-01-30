import os
import torch
import pickle
from torch.utils.data import Dataset
from matplotlib import image
from util.util import get_bounding_box, get_pixel_coords
import torchvision.transforms as transforms

import numpy as np


DEFAULT_DATA_ROOT = os.path.abspath('/usr/wiss/hofherrf/gitRepos/pendulum_2d/data')


class Dataloader():
    def __init__(self, dataset, batch_size, shuffle, device=None, **kwargs):
        if device is not None:
            pixel_coords = []
            images = []
            masks = []
            for p in dataset.pixel_coords:
                pixel_coords.append(p.to(device))
            for i in dataset.images:
                images.append(i.to(device))
            for m in dataset.masks:
                masks.append(m.to(device))
            dataset.pixel_coords = pixel_coords
            dataset.images = images
            dataset.masks = masks

        self.dataset = dataset

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(dataset)
        self.n_batches = (self.n_samples + self.batch_size) // self.batch_size

        self.i = 0
        self.idxs = torch.arange(self.n_samples)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            self.idxs = torch.randperm(self.n_samples)
        self.i = 0
        return self

    def _get_next_batch_idxs(self):
        low = self.i * self.batch_size
        high = min((self.i + 1) * self.batch_size, self.n_samples)
        self.i += 1
        return self.idxs[low:high]

    def __next__(self):
        if self.i >= self.n_batches:
            raise StopIteration

        batch_idxs = self._get_next_batch_idxs()
        batch = self.dataset.__getitem__(batch_idxs)

        return batch


def get_split_dynamic_pixel_data(
    path_data,
    skip_timesteps=1,
    max_samples=1e6,
    max_samples_eval=1e6,
    start_idx_test=0,
    data_root=DEFAULT_DATA_ROOT,
    device_training="cpu",
    device_test="cpu",
    **kwargs
):
    train_data = DynamicPixelDataset(
        path_data=path_data,
        skip_timesteps=skip_timesteps,
        max_samples=max_samples,
        data_root=data_root,
        device=device_training
    )

    valid_data = DynamicPixelDataset(
        path_data=path_data,
        skip_timesteps=skip_timesteps,
        start_index=start_idx_test,
        max_samples=max_samples_eval,
        data_root=data_root,
        device=device_test
    )

    return train_data, valid_data


class DynamicPixelDataset(Dataset):
    def __init__(
        self,
        path_data,
        skip_timesteps=0,
        start_index=0,
        max_samples=1e6,
        data_root=DEFAULT_DATA_ROOT,
        device="cpu",
    ):
        path_data = os.path.join(data_root, path_data)

        # Load parameters
        self.parameters = torch.load(path_data + '/parameters.pt')

        # Save full time span and trajectory for trajectory evaluation
        self._full_tspan = self.parameters["tspan"]
        self._full_traj = self.parameters['traj_angles']

        # Adjust time indices
        indices = torch.arange(
            start_index,
            min(len(self.parameters["tspan"]), (1+skip_timesteps)*max_samples),
            1+skip_timesteps)
        if start_index != 0:
            indices = torch.cat([torch.tensor([0]), indices])

        self.parameters["tspan"] = self.parameters["tspan"][indices]
        self.parameters['traj_angles'] = self.parameters['traj_angles'][indices]

        # Load images and masks
        images = torch.tensor(
            [image.imread(f"{path_data}/{i}.jpg") for i in indices],
            dtype=torch.float32
        ) / 255.0
        self.wo_pendulum = torch.tensor([image.imread(os.path.join(path_data, 'wo_pendulum.jpg'))], dtype=torch.float32).squeeze() / 255.0
        masks = torch.load(os.path.join(path_data, 'masks.pt'))[indices, ...]

        # Get shape of images
        H, W, _ = images[0, ...].shape
        self.image_dim = [H, W]

        # Store images, masks and pixel coordinates
        self.images = images.view(len(indices), -1, 3).to(device)
        self.masks = masks.view(len(indices), -1).to(device)
        self.pixel_coords = get_pixel_coords(H, W).to(device)

    def get_image_dim(self):
        return self.image_dim

    def get_pixel_coords(self):
        return self.pixel_coords

    def get_full_images(self, indices=-1):
        # Indices can be a single integer or a list of intergers. If none given (or -1) return all images
        if indices == -1:
            return_ims = self.images[:, :, :]
        else:
            return_ims = self.images[indices, :, :]
        return torch.reshape(return_ims, (-1, *self.get_image_dim(), 3)).squeeze()

    def get_full_mask(self, indices=-1):
        # Indices can be a single integer or a list of intergers. If none given (or -1) return all images
        if indices == -1:
            return_masks = self.masks[:, :]
        else:
            return_masks = self.masks[indices, :]
        return torch.reshape(return_masks.float(), (-1, *self.get_image_dim())).squeeze()

    def __len__(self):
        return self.pixel_coords.shape[0]

    def __getitem__(self, idx: int):
        im_vals = self.images[:, idx, :].transpose(0, 1)
        masks = self.masks[:, idx].transpose(0, 1)

        item = {
            "coords": self.pixel_coords[idx, :],
            "im_vals": im_vals,
            "mask": masks
        }

        return item


def from_pickle(path):  # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


class ImageDataset_LagrangianVAE(Dataset):
    def __init__(
        self, path_data, T_pred, batch_idx, data_root=DEFAULT_DATA_ROOT,
        offset_x=0, offset_y=0,
        **kwargs
    ):
        path_data = os.path.join(data_root, path_data)
        data = from_pickle(path_data)
        # Data in the dataset has the shape (u_levels, t_steps, n_runs_per_u_level, H, W) where u_levels is differnt constant levels of
        # actuation (which is ignored in our case). At the 0 index for u_levels the actuation is 0
        masks = torch.tensor(data['x'][0, :, batch_idx, ...])

        _, dim_y, dim_x = masks.shape

        if offset_x > 0:
            masks = torch.cat([masks[:, :, -offset_x:], masks[:, :, :(dim_x - offset_x)]], dim=-1)
        elif offset_x < 0:
            masks = torch.cat([masks[:, :, -offset_x:], masks[:, :, :-offset_x]], dim=-1)

        if offset_y > 0:
            masks = torch.cat([masks[:, -offset_y:, :], masks[:, :(dim_y - offset_y), :]], dim=-2)
        elif offset_y < 0:
            masks = torch.cat([masks[:, -offset_y:, :], masks[:, :-offset_y, :]], dim=-2)

        [_, H, W] = masks.shape
        self.image_dim = [H, W]
        self.pixel_coords = get_pixel_coords(H, W)

        self.n_timesteps_for_training = T_pred + 1
        self.t_steps_eval = torch.tensor(data['t'], dtype=torch.float32)
        self.t_steps_train = self.t_steps_eval[:self.n_timesteps_for_training]

        self.masks_eval = masks.view(len(self.t_steps_eval), -1) > 0.5
        self.masks = self.masks_eval[:self.n_timesteps_for_training, :]

    def get_full_mask(self, indices=-1, eval_data=True):
        # Indices can be a single integer or a list of intergers. If none given (or -1) return all images
        if not eval_data:
            masks = self.masks
        else:
            masks = self.masks_eval

        if indices == -1:
            return_masks = masks[:, :]
        else:
            return_masks = masks[indices, :]
        return torch.reshape(return_masks, (-1, *self.image_dim)).squeeze()

    def get_pixel_coords(self, **kwargs):
        return self.pixel_coords

    def get_image_dim(self, **kwargs):
        return self.image_dim

    def __getitem__(self, idx):
        masks = self.masks[:, idx].transpose(0, 1)

        item = {
            "coords": self.pixel_coords[idx, :],
            "mask": masks
        }

        return item

    def __len__(self):
        return self.pixel_coords.shape[0]


class ImageDataset_paig(Dataset):
    def __init__(
        self,
        path_data,
        batch_idx,
        prediction_length=None,
        data_root=DEFAULT_DATA_ROOT,
        training_length=None,
        test_set=False,
        **kwargs
    ):
        path_data = os.path.join(data_root, path_data)
        data = torch.load(path_data)

        if training_length is None:
            training_length = 500     # Large Number to include everything

        # Use prediction length to overwrite training_length
        if test_set:
            training_length = prediction_length

        masks1 = data['masks1'][batch_idx, :training_length]
        masks2 = data['masks2'][batch_idx, :training_length]
        images = data['ims'][batch_idx, :training_length]

        if not test_set:
            # Ensure that we cannot accidentially use information other than on the first frame
            masks1[1:, ...] = torch.rand_like(masks1[1:, ...]) > 0.5
            masks2[1:, ...] = torch.rand_like(masks2[1:, ...]) > 0.5

            # Blur masks to create "blobb" for loss on first frame
            trafo = transforms.GaussianBlur(kernel_size=(35, 35), sigma=(6, 6))
            masks1 = trafo(masks1.unsqueeze(1)).squeeze() > 0.1
            masks2 = trafo(masks2.unsqueeze(1)).squeeze() > 0.1

        n_tsteps, H, W, _ = images.shape
        self.image_dim = [H, W]
        self.pixel_coords = get_pixel_coords(H, W)

        dt = 0.3    # From appendix of paper
        self.t_steps = dt*torch.arange(0, n_tsteps)
        n_tsteps = len(self.t_steps)

        self.masks1 = masks1.view(n_tsteps, -1)
        self.masks2 = masks2.view(n_tsteps, -1)
        self.images = images.view(n_tsteps, -1, 3)

        traj_mean1 = torch.tensor([])
        traj_mean2 = torch.tensor([])
        traj_bb1 = torch.tensor([])
        traj_bb2 = torch.tensor([])
        for i in range(n_tsteps):
            traj_mean1 = torch.cat([
                traj_mean1,
                torch.mean(self.pixel_coords[masks1[i].flatten() > 0], dim=0).unsqueeze(0)
                ],
                dim=0
            )
            traj_mean2 = torch.cat([
                traj_mean2,
                torch.mean(self.pixel_coords[masks2[i].flatten() > 0], dim=0).unsqueeze(0)
                ],
                dim=0
            )
            _, center_bb1 = get_bounding_box(masks1[i])
            _, center_bb2 = get_bounding_box(masks2[i])
            traj_bb1 = torch.cat([traj_bb1, center_bb1.unsqueeze(0)], dim=0)
            traj_bb2 = torch.cat([traj_bb2, center_bb2.unsqueeze(0)], dim=0)

        self.trajectory_mean1 = traj_mean1
        self.trajectory_mean2 = traj_mean2
        self.trajectory_bb_center1 = traj_bb1
        self.trajectory_bb_center2 = traj_bb2

    def get_image_dim(self):
        return self.image_dim

    def get_pixel_coords(self):
        return self.pixel_coords

    def get_full_mask(self, indices=-1, combined=False):
        # Indices can be a single integer or a list of intergers. If none given (or -1) return all images
        masks1 = self.masks1
        masks2 = self.masks2

        if indices == -1:
            return_masks1 = masks1[:, :]
            return_masks2 = masks2[:, :]
        else:
            return_masks1 = masks1[indices, :]
            return_masks2 = masks2[indices, :]

        if combined:
            return_values = torch.logical_or(
                torch.reshape(return_masks1, (-1, *self.get_image_dim())).squeeze(),
                torch.reshape(return_masks2, (-1, *self.get_image_dim())).squeeze()
            )
        else:
            return_values = {
                'masks1': torch.reshape(return_masks1, (-1, *self.get_image_dim())).squeeze(),
                'masks2': torch.reshape(return_masks2, (-1, *self.get_image_dim())).squeeze(),
            }

        return return_values

    def get_full_images(self, indices=-1):
        # Indices can be a single integer or a list of intergers. If none given (or -1) return all images
        if indices == -1:
            return_ims = self.images[:, :, :]
        else:
            return_ims = self.images[indices, :, :]
        return torch.reshape(return_ims, (-1, *self.get_image_dim(), 3)).squeeze()

    def __getitem__(self, idx: int):
        im_vals = self.images[:, idx, :].transpose(0, 1)
        masks1 = self.masks1[:, idx].transpose(0, 1)
        masks2 = self.masks2[:, idx].transpose(0, 1)

        item = {
            "coords": self.pixel_coords[idx, :],
            "im_vals": im_vals,
            "mask1": masks1,
            "mask2": masks2
        }

        return item

    def __len__(self):
        return self.pixel_coords.shape[0]


class ImageDataset_realData(Dataset):
    def __init__(
        self,
        path_data,
        skip_timesteps=0,
        start_index=0,
        max_samples=1e6,
        data_root=DEFAULT_DATA_ROOT,
        device="cpu",
        normalize_by_H=False,
        indices=None,
        **kwargs
    ):
        path_data = os.path.join(data_root, path_data)
        data = np.load(path_data)

        # Construct indices
        if indices is None:
            indices = torch.arange(
                start_index,
                min(len(data['ts']), (1+skip_timesteps)*max_samples),
                1+skip_timesteps)

        # Zero needs always needs to be part of the index set for the ode solver
        if indices[0] != 0:
            indices = torch.cat([torch.tensor([0]), indices])

        # Find unused indices
        all_inds = torch.arange(0, len(data['ts']))
        uniques, counts = torch.cat((indices, all_inds)).unique(return_counts=True)
        self.unused_inds = uniques[counts == 1]

        images = torch.tensor(data['imgs'], dtype=torch.float32)
        if 'masks' in data:
            masks = torch.tensor(data['masks'], dtype=torch.bool)
        else:
            masks = torch.zeros(*images.shape[:-1])
        tsteps = torch.tensor(data['ts'], dtype=torch.float32)

        masks = masks[indices]
        images = images[indices]
        tsteps = tsteps[indices]

        n_tsteps, H, W, _ = images.shape
        self.image_dim = [H, W]
        self.pixel_coords = get_pixel_coords(H, W, normalize_by_H=normalize_by_H).to(device)

        self.t_steps = tsteps

        self.masks = masks.view(n_tsteps, -1).to(device)
        self.images = images.view(n_tsteps, -1, 3).to(device)

    def get_image_dim(self):
        return self.image_dim

    def get_pixel_coords(self):
        return self.pixel_coords

    def get_full_mask(self, indices=-1):
        # Indices can be a single integer or a list of intergers. If none given (or -1) return all images
        masks = self.masks.float()

        if indices == -1:
            return_masks = masks[:, :]
        else:
            return_masks = masks[indices, :]

        return torch.reshape(return_masks, (-1, *self.get_image_dim())).squeeze()

    def get_full_images(self, indices=-1):
        # Indices can be a single integer or a list of intergers. If none given (or -1) return all images
        if indices == -1:
            return_ims = self.images[:, :, :]
        else:
            return_ims = self.images[indices, :, :]
        return torch.reshape(return_ims, (-1, *self.get_image_dim(), 3)).squeeze()

    def __getitem__(self, idx: int):
        im_vals = self.images[:, idx, :].transpose(0, 1)
        masks = self.masks[:, idx].transpose(0, 1)

        item = {
            "coords": self.pixel_coords[idx, :],
            "im_vals": im_vals,
            "mask": masks
        }

        return item

    def __len__(self):
        return self.pixel_coords.shape[0]
