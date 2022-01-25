import copy
import shutil
import subprocess

import torch
from torch import nn
import numpy as np
import nibabel as nib
import os
from scipy import ndimage

import niclib as nl

from .model_utils import PatchSet_LATS_test


def postprocess_brain(probs, internal_threshold=0.85, external_threshold=0.15):
    """Post processes a Deep Learning segmentation that can put big background probabilities in middle"""

    ### 1. Internal Brain Mask
    # Get brain mask from probs (wherever the background probability is < TH we will be considered brain)
    internal_brain_mask = (probs[0] < (1.0 - internal_threshold)).astype(float)
    # Get largest connected component as brain
    internal_brain_mask = nl.data.get_largest_connected_component(internal_brain_mask).astype(float)
    # Fill any holes left from binarization
    internal_brain_mask = ndimage.binary_fill_holes(internal_brain_mask).astype(float)

    ### 2. External Brain Mask
    # Get brain mask from probs (wherever the background probability is < TH we will be considered brain)
    external_brain_mask = (probs[0] < (1.0 - external_threshold)).astype(float)
    # Get largest connected component as brain
    external_brain_mask = nl.data.get_largest_connected_component(external_brain_mask).astype(float)
    # Dilate a bit the mask to include extra borders
    external_brain_mask = ndimage.binary_dilation(external_brain_mask, iterations=3).astype(float)
    # Fill any holes left from binarization
    external_brain_mask = ndimage.binary_fill_holes(external_brain_mask).astype(float)

    ### Apply brain_mask to probs and renormalize
    # Overwritte all channels of what we are confident is background
    probs[0][external_brain_mask < 0.5] = 1.0
    probs[1][external_brain_mask < 0.5] = 0.0
    probs[2][external_brain_mask < 0.5] = 0.0
    probs[3][external_brain_mask < 0.5] = 0.0

    # Set to 0 the background probability of what we know is brain tissue
    probs[0][internal_brain_mask > 0.5] = 0.0
    # Renormalize to get internal probabilities to add to 1
    probs = probs / np.sum(probs, axis=0)

    return probs


def segment_tissue_LATS(img,
                        trained_model,
                        in_shape=(2, 32, 32, 32),
                        out_shape=(4, 32, 32, 32),
                        extraction_step=(12, 12, 12),
                        activation=nn.Softmax(dim=1),
                        postprocessing=True,
                        verbose=True,
                        norm_type='minmax'):

    if img.shape[-1] == 1:
        img = np.squeeze(img, axis=-1)
    assert len(img.shape) == 4

    predictor = PatchTester_LATS(
        patch_shape=in_shape,
        patch_out_shape=out_shape,
        extraction_step=extraction_step,
        activation=activation,
        verbose=verbose,
        norm_type=norm_type)

    probs = predictor.predict(trained_model, img)

    if postprocessing:
        probs = postprocess_brain(probs)

    return probs


class PatchTester_LATS:
    """Forward pass a volume through the given network using uniformly sampled patches. After a patch is predicted, it
    is accumulated by averaging back in a common space.

    :param patch_shape: tuple (X, Y, Z) with the input patch shape of the model.
    :param patch_out_shape: (default: None) shape of the network forward passed patch, if None it is assumed to be of
        the same shape as ``patch_shape``.
    :param extraction_step: tuple (X, Y, Z) with the extraction step to uniformly sample patches.
    :param str normalize: either 'none', 'patch' or 'image'.
    :param activation: (default: None) the output activation after the forward pass.
    :param int batch_size: (default: 32) batch size for prediction, bigger batch sizes can speed up prediction if
        gpu utilization (NOT gpu memory) is under 100%.
    """

    def __init__(self, patch_shape, extraction_step, activation=None, batch_size=32, patch_out_shape=None, verbose=True, norm_type='minmax'):
        self.in_shape = patch_shape
        self.out_shape = patch_shape if patch_out_shape is None else patch_out_shape
        self.extraction_step = extraction_step
        self.bs = batch_size
        self.activation=activation
        self.v = verbose
        self.norm_type = norm_type

        assert len(extraction_step) == 3, 'Please give extraction step as (X, Y, Z)'
        assert len(self.in_shape) == len(self.out_shape) ==  4, 'Please give shapes as (CH, X, Y, Z)'

        self.num_ch_out = self.out_shape[0]

    def predict(self, model, x, device='cuda'):
        """ Predict the given volume ``x`` using the provided ``model``.

        :param normROI:
        :param torch.nn.Module model: The trained torch model.
        :param x: the input volume with shape (CH, X, Y, Z) to predict.
        :param mask: (default: None) a binary array of the same shape as x that defines the ROI for patch extraction.
        :param str device: the torch device identifier.
        :return: The accumulated outputs of the network as an array of the same shape as x.
        """
        from niclib.utils import print_progress_bar, RemainingTimeEstimator
        from niclib.generator import make_generator
        from niclib.generator.patch import PatchSet, sample_centers_uniform, _get_patch_slice

        assert len(x.shape) == 4, 'Please give image with shape (CH, X, Y, Z)'

        x_orig = copy.copy(x)

        # First pad image to ensure all voxels in volume are processed independently of extraction_step
        pad_dims = [(0,)] + [(int(np.ceil(in_dim / 2.0)),) for in_dim in self.in_shape[1:]]
        x = np.pad(x, pad_dims, mode='edge')

        # Create patch generator with known patch center locations.
        x_centers = sample_centers_uniform(x[0], self.in_shape[1:], self.extraction_step)
        x_slices = _get_patch_slice(x_centers, self.in_shape[1:])

        patch_gen = make_generator(
            set=PatchSet_LATS_test(x, self.in_shape[1:], x_centers, norm_type=self.norm_type),
            batch_size=self.bs,
            shuffle=False)

        # Put accumulation in torch (GPU accelerated :D)
        voting_img = torch.zeros((self.num_ch_out,) + x[0].shape, device=device).float()
        counting_img = torch.zeros_like(voting_img).float()

        # Perform inference and accumulate results in torch (GPU accelerated :D (if device is cuda))
        model.eval()
        model.to(device)
        with torch.no_grad():
            rta = RemainingTimeEstimator(len(patch_gen)) if self.v else None

            for n, (x_patch, x_slice) in enumerate(zip(patch_gen, x_slices)):
                x_patch = x_patch.to(device)

                y_pred = model(x_patch)
                if self.activation is not None:
                    y_pred = self.activation(y_pred)

                batch_slices = x_slices[self.bs * n:self.bs * (n + 1)]
                for predicted_patch, patch_slice in zip(y_pred, batch_slices):
                    voting_img[patch_slice] += predicted_patch
                    counting_img[patch_slice] += torch.ones_like(predicted_patch)

                if self.v:
                    print_progress_bar(self.bs * n, self.bs * len(patch_gen),
                                       suffix="patches predicted - ETA: {}".format(rta.update(n)))
            if self.v:
                print_progress_bar(self.bs * len(patch_gen), self.bs * len(patch_gen),
                                   suffix="patches predicted - ETA: {}".format(rta.elapsed_time()))

        counting_img[counting_img == 0.0] = 1.0  # Avoid division by 0
        predicted_volume = torch.div(voting_img, counting_img).detach().cpu().numpy()

        # Unpad volume to return to original shape
        unpad_slice = [slice(None)] + [slice(in_dim[0], x_dim - in_dim[0]) for in_dim, x_dim in zip(pad_dims[1:], x.shape[1:])]
        predicted_volume = predicted_volume[tuple(unpad_slice)]

        assert np.array_equal(x_orig.shape[1:], predicted_volume.shape[1:]), '{} != {}'.format(x_orig.shape, predicted_volume.shape)
        return predicted_volume

if __name__ == '__main__':
    print('hi')
    nifti_seg = nib.load('/media/user/dades/DATASETS-WS/miriad-longitudinal-workspace/mcross_miriad_rescan_sim0,0_lr0,1/196/03_2_tissue_probs.nii.gz')

    seg = np.squeeze(nifti_seg.get_fdata())
    seg_post, internal, external = postprocess_brain(seg)

    nib.Nifti1Image(np.transpose(seg_post, axes=(1, 2, 3, 0)), nifti_seg.affine, nifti_seg.header).to_filename('hi.nii.gz')
    nib.Nifti1Image(internal, nifti_seg.affine, nifti_seg.header).to_filename('hi_internal.nii.gz')
    nib.Nifti1Image(external, nifti_seg.affine, nifti_seg.header).to_filename('hi_external.nii.gz')