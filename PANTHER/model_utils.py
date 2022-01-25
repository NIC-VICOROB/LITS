import sys
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import niclib as nl

import torch.utils.data

INIT_PRELU = 0.0
BN_MOMENTUM = 0.01

from niclib.generator.patch import PatchSampling, _norm_patch, PatchInstruction, _get_patch_slice


class PatchSet_LATS(torch.utils.data.Dataset):
    """
    Creates a torch dataset that returns patches extracted from images either from predefined extraction centers or
    using a predefined patch sampling strategy.

    :param List[np.ndarray] images: list of images with shape (CH, X, Y, Z)
    :param PatchSampling sampling: An object of type PatchSampling defining the patch sampling strategy.
    :param normalize: one of ``'none'``, ``'patch'``, ``'image'``.
    :param dtype: the desired output data type (default: torch.float)
    :param List[List[tuple]] centers: (optional) a list containing a list of centers for each provided image.
        If provided it ignores the given sampling and directly uses the centers to extract the patches.
    """

    def  __init__(self, images, lesion_masks, patch_shape, centers, dtype=torch.float, norm_type='minmax'):
        assert all([img.ndim == 4 for img in images]), 'Images must be numpy ndarrays with dimensions (C, X, Y, Z)'
        assert len(patch_shape) == 3, 'len({}) != 3'.format(patch_shape)
        assert len(centers) == len(images)

        self.images, self.lesion_masks, self.dtype = images, lesion_masks, dtype

        # Build all instructions according to centers and normalize
        self.instructions = []
        images_centers = centers

        for image_idx, image_centers in enumerate(images_centers):
            # Compute normalize function for this image's patches in the normalize_ROI
            normROI_img = copy.deepcopy(self.images[image_idx][0:1]) # Assuming two channel input
            normROI_img[normROI_img == 0.0] = np.nan
            if norm_type == 'normal':
                means = np.nanmean(normROI_img, axis=(1,2,3), keepdims=True, dtype=np.float64)
                stds = np.nanstd(normROI_img, axis=(1,2,3), keepdims=True, dtype=np.float64)
            elif norm_type == 'minmax':
                pmin, pmax = np.nanpercentile(normROI_img, [0.1, 99.9], axis=(1,2,3), keepdims=True)

                means = (pmax - pmin) / 2.0
                stds = (pmax - pmin) / 2.0
            else:
                raise ValueError
            del normROI_img

            ## Generate instructions
            lesion_keys = list(self.lesion_masks[image_idx].keys())
            for n, center in enumerate(image_centers):
                self.instructions.append({
                    'image_idx': image_idx,
                    'lesion_key': lesion_keys[n % len(lesion_keys)] if len(lesion_keys) > 0 else None,
                    'center': center,
                    'shape': patch_shape,
                    'norm_stats': {'m': means, 's': stds}
                })

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        # Unpack instruction contents
        instr = self.instructions[index]
        image = self.images[instr['image_idx']]
        patch_slice = _get_patch_slice(instr['center'], instr['shape'])
        norm_stats = instr['norm_stats']

        # Extract image patch
        image_patch = copy.deepcopy(image[patch_slice])

        # Extract lesion patch
        if instr['lesion_key'] is not None:
            lesion_mask = self.lesion_masks[instr['image_idx']][instr['lesion_key']]
            lesion_patch = copy.deepcopy(lesion_mask.get_dense(patch_slice)).astype(image_patch.dtype)
        else:
            lesion_patch = np.zeros_like(image_patch)

        # Apply lesion mask to image patch
        image_patch = image_patch * (1.0 - lesion_patch)

        # Normalize after applying mask
        image_patch = (image_patch - norm_stats['m']) / norm_stats['s']

        # Concatenate patches
        x_patch = np.concatenate([image_patch, lesion_patch], axis=0)
        return torch.tensor(np.ascontiguousarray(x_patch), dtype=self.dtype)



class PatchSet_LATS_test(torch.utils.data.Dataset):
    """
    Creates a torch dataset that returns patches extracted from images either from predefined extraction centers or
    using a predefined patch sampling strategy.

    :param List[np.ndarray] images: list of images with shape (CH, X, Y, Z)
    :param PatchSampling sampling: An object of type PatchSampling defining the patch sampling strategy.
    :param normalize: one of ``'none'``, ``'patch'``, ``'image'``.
    :param dtype: the desired output data type (default: torch.float)
    :param List[List[tuple]] centers: (optional) a list containing a list of centers for each provided image.
        If provided it ignores the given sampling and directly uses the centers to extract the patches.
    """

    def  __init__(self, image, patch_shape, centers, dtype=torch.float, norm_type='minmax'):
        self.image, self.dtype = image, dtype

        # Build all instructions according to centers and normalize

        # Compute normalize function for this image's patches in the normalize_ROI
        normROI_img = copy.deepcopy(self.image[0:1]) # Assuming two channel input
        normROI_img[normROI_img == 0.0] = np.nan
        if norm_type == 'normal':
            means = np.nanmean(normROI_img, axis=(1,2,3), keepdims=True, dtype=np.float64)
            stds = np.nanstd(normROI_img, axis=(1,2,3), keepdims=True, dtype=np.float64)
        elif norm_type == 'minmax':
            pmin, pmax = np.nanpercentile(normROI_img, [0.1, 99.9], axis=(1,2,3), keepdims=True)

            means = (pmax - pmin) / 2.0
            stds = (pmax - pmin) / 2.0
        else:
            raise ValueError
        del normROI_img

        self.instructions = []
        ## Generate instructions
        for center in centers:
            self.instructions.append({
                'center': center,
                'shape': patch_shape,
                'norm_stats': {'m': means, 's': stds}
            })

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        # Unpack instruction contents
        instr = self.instructions[index]
        image = self.image
        patch_slice = _get_patch_slice(instr['center'], instr['shape'])
        norm_stats = instr['norm_stats']

        image_patch = copy.deepcopy(image[patch_slice])
        image_patch[0:1] = (image_patch[0:1] - norm_stats['m']) / norm_stats['s']

        x_patch = image_patch
        return torch.tensor(np.ascontiguousarray(x_patch), dtype=self.dtype)



class SUNETx4(nn.Module):
    def __init__(self, in_ch=5, out_ch=2, nfilts=32, ndims=3, dropout_rate=0.0, activation=None):
        super(SUNETx4, self).__init__()
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if ndims is 2 else nn.ConvTranspose3d

        self.inconv = Conv(in_ch, 1 * nfilts, 3, padding=1)

        self.dual1 = DualRes(1 * nfilts, ndims)
        self.dual2 = DualRes(2 * nfilts, ndims, dropout_rate=dropout_rate)
        self.dual3 = DualRes(4 * nfilts, ndims, dropout_rate=dropout_rate)
        self.dual4 = DualRes(8 * nfilts, ndims, dropout_rate=dropout_rate)

        self.down1 = DownStep(1 * nfilts, ndims)
        self.down2 = DownStep(2 * nfilts, ndims)
        self.down3 = DownStep(4 * nfilts, ndims)

        self.mono3 = MonoRes(4 * nfilts, ndims)
        self.mono2 = MonoRes(2 * nfilts, ndims)
        self.mono1 = MonoRes(1 * nfilts, ndims)

        self.up4 = ConvTranspose(in_channels=8 * nfilts, out_channels=4 * nfilts, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.up3 = ConvTranspose(in_channels=4 * nfilts, out_channels=2 * nfilts, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.up2 = ConvTranspose(in_channels=2 * nfilts, out_channels=1 * nfilts, kernel_size=3, padding=1, output_padding=1, stride=2)

        self.outconv = Conv(nfilts, out_ch, 3, padding=1)
        self.activation_out = activation

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nparams = sum([np.prod(p.size()) for p in model_parameters])
        print("SUNETx4_{}D_f{} network with {} parameters".format(ndims, nfilts, nparams))

    def forward(self, x_in):
        l1_start = self.inconv(x_in)

        l1_end = self.dual1(l1_start)
        l2_start = self.down1(l1_end)

        l2_end = self.dual2(l2_start)
        l3_start = self.down2(l2_end)

        l3_end = self.dual3(l3_start)
        l4_start = self.down3(l3_end)

        l4_latent = self.dual4(l4_start)
        r4_up = self.up4(l4_latent)

        r3_start = l3_end + r4_up
        r3_end = self.mono3(r3_start)
        r3_up = self.up3(r3_end)

        r2_start = l2_end + r3_up
        r2_end = self.mono2(r2_start)
        r2_up = self.up2(r2_end)

        r1_start = l1_end + r2_up
        r1_end = self.mono1(r1_start)

        pred = self.outconv(r1_end)

        if self.activation_out is not None:
            pred = self.activation_out(pred)

        return pred


class DualRes(nn.Module):
    def __init__(self, num_ch, ndims=3, dropout_rate=0.0):
        super(DualRes, self).__init__()
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if ndims is 2 else nn.BatchNorm3d
        Dropout = nn.AlphaDropout

        self.conv_path = nn.Sequential(
            BatchNorm(num_ch, momentum=BN_MOMENTUM, eps=0.001),
            nn.PReLU(num_ch, init=INIT_PRELU),
            Conv(num_ch, num_ch, 3, padding=1),
            Dropout(p=dropout_rate),
            BatchNorm(num_ch, momentum=BN_MOMENTUM, eps=0.001),
            nn.PReLU(num_ch, init=INIT_PRELU),
            Conv(num_ch, num_ch, 3, padding=1))

    def forward(self, x_in):
        return self.conv_path(x_in) + x_in


class MonoRes(nn.Module):
    def __init__(self, num_ch, ndims=3):
        super(MonoRes, self).__init__()
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if ndims is 2 else nn.BatchNorm3d

        self.conv_path = nn.Sequential(
            BatchNorm(num_ch, momentum=BN_MOMENTUM, eps=0.001),
            nn.PReLU(num_ch, init=INIT_PRELU),
            Conv(num_ch, num_ch, 3, padding=1))

    def forward(self, x_in):
        x_out = self.conv_path(x_in) + x_in
        return x_out


class DownStep(nn.Module):
    def __init__(self, in_ch, ndims=3):
        super(DownStep, self).__init__()
        MaxPool = nn.MaxPool2d if ndims is 2 else nn.MaxPool3d
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if ndims is 2 else nn.BatchNorm3d

        self.pool_path = MaxPool(2)
        self.conv_path = nn.Sequential(
            BatchNorm(in_ch, momentum=BN_MOMENTUM, eps=0.001),
            nn.PReLU(in_ch, init=INIT_PRELU),
            Conv(in_ch, in_ch, 3, padding=1, stride=2))

    def forward(self, x_in):
        x_out = torch.cat((self.conv_path(x_in), self.pool_path(x_in)), dim=1)  # Channel dimension
        return x_out
