import copy
import itertools
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as f
import os
import nibabel as nib

from PANTHER.model_utils import SUNETx4
import niclib as nl
import acglib as acg


from lats_load import load_healthy_lesion_dataset

cfg_default = {
    'patch_shape': (32, 32, 32),

    'num_patches': 150000,
    'train_fraction': 0.9,
    'norm_type': 'minmax', # ['zscore', '0to1']

    'do_mask_inpainting': True,
    'detach_inpainter': False,

    'seg_loss': acg.losses.soft_crossentropy_with_logits,
    'rec_loss': nn.MSELoss(),
    'rec_weight': 1.0,

    'optimizer_opts': dict(lr=0.2),
    'max_epochs': 200,
    'batch_size': 32,
    'train_metrics': {},
    'val_metrics': {},

    'early_stopping_metric': 'loss',
    'patience': 8,
    'checkpoint_metric': 'loss',

    'extraction_step': (12, 12, 12)
}


def get_normalize_params_LITS(arr, mask=None, norm_type='minmax'):
    norm_img = copy.deepcopy(arr)
    if mask is not None:
        norm_img[mask == 0] = np.nan

    minmax_low, minmax_high = np.nanpercentile(norm_img, [0.05, 99.95], axis=(-3, -2, -1), keepdims=True)

    arr_mean = np.nanmean(norm_img.astype(float), axis=(-3, -2, -1), keepdims=True)
    arr_std = np.nanstd(norm_img.astype(float), axis=(-3, -2, -1), keepdims=True)

    arr_min = np.nanmin(norm_img, axis=(-3, -2, -1), keepdims=True)
    arr_max = np.nanmax(norm_img, axis=(-3, -2, -1), keepdims=True)

    del norm_img
    return {'type': norm_type,
            '0to1_params': (arr_min, arr_max),
            'zscore_params': (arr_mean, arr_std),
            'minmax_params': (minmax_low, minmax_high)}


def normalize_LITS(arr, norm_params, norm_type=None):
    if norm_type is None:
        norm_type = norm_params['type']
    
    if norm_type is 'minmax':
        new_low, new_high = norm_params['minmax_params']
        norm_arr = (arr - new_low) / (new_high - new_low)  # Put between 0 and 1
        norm_arr = np.clip((2.0 * norm_arr) - 1.0, -1.0, 1.0)  # Put between -1 and 1 and clip extrema
    elif norm_type is 'zscore':
        arr_mean, arr_std = norm_params['zscore_params']
        norm_arr = (arr - arr_mean) / arr_std
    elif norm_type is '0to1':
        new_low, new_high = norm_params['0to1_params']
        norm_arr = (arr - new_low) / (new_high - new_low)  # Put between 0 and 1
    else:
        raise ValueError(f'Normalization "{norm_type}" not recognized')
    
    return norm_arr
    


class WNet_LITS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.impainter = SUNETx4(in_ch=2, out_ch=1, activation=nn.Tanh())
        self.segmenter = SUNETx4(in_ch=1, out_ch=4, activation=None)

    def forward(self, x):
        x_t1_holes, lesion_mask = x[:, 0:1, ...], x[:, 1:2, ...]

        # Impaint the lesion hole
        x_impainted = self.impainter(x)

        # Get original voxels from no-hole and impainted voxels from hole area
        x_reconstructed = torch.where(
            condition=lesion_mask <= 0.0, input=x_t1_holes, other=x_impainted)

        y = self.segmenter(x_reconstructed)
        return y if not self.training else (x_impainted, y)


class WNet_LITS_v2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.impainter = SUNETx4(in_ch=2, out_ch=1, activation=nn.Tanh())
        self.segmenter = SUNETx4(in_ch=1, out_ch=4, activation=None)

    def forward(self, x):
        x_t1_holes, lesion_mask = x[:, 0:1, ...], x[:, 1:2, ...]

        # Impaint the lesion hole if there is a lesion
        if self.training:
            x_impainted = self.impainter(x)
            x_reconstructed = torch.where(
                condition=lesion_mask <= 0.0, input=x_t1_holes, other=x_impainted)
            y = self.segmenter(x_reconstructed)
            return x_impainted, y
        else:
            if torch.any(lesion_mask == 1):
                x_impainted = self.impainter(x)
                x_reconstructed = torch.where(
                    condition=lesion_mask <= 0.0, input=x_t1_holes, other=x_impainted)
            else:
                x_reconstructed = x_t1_holes

            return self.segmenter(x_reconstructed)

class WNet_LITS_v3(torch.nn.Module):
    def __init__(self, detach_inpainter=False, inpainter_activation=nn.Tanh()):
        super().__init__()
        if detach_inpainter:
            print('WNet_LITS_v3 with detached inpainter')
        self.detach_inpainter = detach_inpainter
        self.impainter = SUNETx4(in_ch=2, out_ch=1, activation=inpainter_activation)
        self.segmenter = SUNETx4(in_ch=1, out_ch=4, activation=None)
        
        print(type(inpainter_activation))
        
    def forward(self, x):
        x_t1_holes, lesion_mask = x[:, 0:1, ...], x[:, 1:2, ...]

        # Impaint the lesion hole if there is a lesion
        if self.training:
            x_impainted = self.impainter(x)
            x_reconstructed = torch.where(
                condition=lesion_mask <= 0.0, input=x_t1_holes, other=x_impainted)
            if self.detach_inpainter:
                x_reconstructed = x_reconstructed.detach()
            y = self.segmenter(x_reconstructed)
            return x_impainted, y
        else:
            if torch.any(lesion_mask == 1):
                x_impainted = self.impainter(x)
                x_reconstructed = torch.where(
                    condition=lesion_mask <= 0.0, input=x_t1_holes, other=x_impainted)
                if self.detach_inpainter:
                    x_reconstructed = x_reconstructed.detach()
            else:
                x_reconstructed = x_t1_holes

            return self.segmenter(x_reconstructed)

class WNet_LITS_loss(torch.nn.Module):
    def __init__(self, rec_loss, seg_loss, rec_w=1.0):
        super().__init__()
        self.rec_w = rec_w
        self.rec_loss = rec_loss
        self.seg_loss = seg_loss

    def forward(self, output, target):
        """
        output may be:
            - tuple (t1_reconstructed, t1_segmentation*)
            - tensor t1_segmentation*
        target will always be a tuple (t1_original, t1_segmentation)
        """
        if isinstance(output, tuple):
            # TRAINING
            return self.rec_w * self.rec_loss(output[0], target[0]) + self.seg_loss(output[1], target[1])
        else:
            # VALIDATION
            return self.seg_loss(output, target[1])


def get_instructions_new(case_idx, case, patch_shape, num_patches, norm_type):
    instructions = []

    num_balanced_centers = int(np.floor(num_patches * 0.5))
    num_hole_centers = int(np.floor(num_patches * 0.5))

    ### Get half of the centers from healthy image without mask
    # Get normalization parameters
    healthy_normalize_params = get_normalize_params_LITS(
            case['t1_brain'], mask=case['t1_brain'] > 0, norm_type=norm_type)
    # Get patch centers
    centers_balanced = acg.patch.sample_centers_labels_fraction(
        labels_image=np.argmax(case['probs'], axis=0),
        labels_fractions={0: 0.1, 1: 0.3, 2: 0.3, 3: 0.3},
        patch_shape=patch_shape,
        n=num_balanced_centers,
        add_rand_offset=True)

    # Add healthy sampled instructions
    for center in centers_balanced:
        instructions.append({
            'case_idx': case_idx,
            'lesion_key': None,
            'center': center,
            'patch_shape': patch_shape,
            'normalize_params': healthy_normalize_params})

    ### Get other half from masks sampled on holes with the right normalization parameters
    centers_per_mask = num_hole_centers // len(case['lesion_masks'])

    # Get lesion keys in list
    for lesion_key in case['lesion_masks'].keys():
        # Get binary dense version of lesion_mask
        lesion_mask = case['lesion_masks'][lesion_key].get_dense()
        # Compute new normalization parameters considering the hole
        lesion_norm_params = get_normalize_params_LITS(
            arr=case['t1_brain'],
            mask=np.logical_and(case['t1_brain'] > 0, lesion_mask <= 0.0),
            norm_type=norm_type)
        # Get centers sampled on the hole, with a random offset
        centers_lesion = acg.patch.sample_centers_mask(
            mask_image=lesion_mask[0],
            patch_shape=patch_shape,
            n=centers_per_mask,
            add_rand_offset=True)
        # Add hole sampled instructions
        for center in centers_lesion:
            instructions.append({
                'case_idx': case_idx,
                'lesion_key': lesion_key,
                'center': center,
                'patch_shape': patch_shape,
                'normalize_params': lesion_norm_params})

    return instructions



def extract_train_sample(instr, data):
    case = data[instr['case_idx']]

    ### INPUT PATCHES
    # Extract image patch
    t1_patch = acg.patch.get_patch(case['t1_brain'], instr['center'], instr['patch_shape'])

    # Extract lesion patch
    if instr['lesion_key'] is not None:
        lesion_mask = case['lesion_masks'][instr['lesion_key']]
        lesion_patch = copy.deepcopy(lesion_mask.get_patch(instr['center'], instr['patch_shape'])).astype(t1_patch.dtype)
    else:
        lesion_patch = np.zeros_like(t1_patch)
    # Put lesion patch between -1 and 1
    lesion_patch = (lesion_patch - 0.5) * 2.0

    # Apply lesion mask to image patch
    t1_hole_patch = copy.deepcopy(t1_patch) * (lesion_patch < 0.0).astype(np.float)
    # Normalize image_patch by minmax
    t1_hole_patch = normalize_LITS(t1_hole_patch, instr['normalize_params'])

    # Concatenate image and lesion patches
    input_numpy = np.concatenate([t1_hole_patch, lesion_patch], axis=0)
    input_torch = torch.tensor(np.ascontiguousarray(input_numpy), dtype=torch.float32)

    ### TARGET PATCHES
    # Normalization for reconstruction target
    rec_patch = normalize_LITS(t1_patch, instr['normalize_params'])
    
    seg_patch = acg.patch.get_patch(case['probs'], instr['center'], instr['patch_shape'])
    tgt_torch = (
        torch.tensor(np.ascontiguousarray(rec_patch), dtype=torch.float32),
        torch.tensor(np.ascontiguousarray(seg_patch), dtype=torch.float32))

    return input_torch, tgt_torch


def store_patch_nifti(train_gen):
    patch_store = []

    for n, (x, y) in enumerate(train_gen):
        x = x.numpy()
        y = (y[0].numpy(), y[1].numpy())

        for batch_idx in range(x.shape[0]):
            t1_hole = (x[batch_idx, 0, :, :, :] + 1) * 0.5
            lesion_mask = (x[batch_idx, 1, :, :, :] + 1) * 0.5
            t1_original = (y[0][batch_idx, 0, :, :, :] + 1) * 0.5

            bg = y[1][batch_idx, 0, :, :, :]  # BG
            csf = y[1][batch_idx, 1, :, :, :]  # CSF
            gm = y[1][batch_idx, 2, :, :, :]  # GM
            wm = y[1][batch_idx, 3, :, :, :]  # WM

            if np.sum(lesion_mask) > 0.0 and np.sum(csf) > 0.0 and np.sum(gm) > 0.0 and np.sum(wm) > 0.0:
                patch_store.append(np.concatenate([
                    t1_original,  # T1 original
                    t1_hole, # T1 hole
                    lesion_mask, # Lesion mask
                    t1_original * lesion_mask, # lesion_hole
                    bg,
                    csf,
                    gm,
                    wm
                 ], axis=1))

        if len(patch_store) > 20 or n > 1000:
            break

    print(len(patch_store))
    patch_store = np.stack(patch_store, axis=0)
    patch_store = np.transpose(patch_store, axes=(1, 2, 3, 0)) # Put batch in last position
    nib.Nifti1Image(patch_store, np.eye(4)).to_filename('lits_train_batch_16.nii.gz')



def train_LITS(cfg_run, dataset, fpath_trained_model):
    cfg = copy.deepcopy(cfg_default)
    cfg.update(cfg_run)
    print("\nTraining LITS network")

    def build_generators(dataset_, num_patches, norm_type):
        num_patches_per_img = int(np.round(num_patches / len(dataset_)))

        print('using new sampling')
        cases_instructions = acg.parallel_run(
            func=get_instructions_new,
            args=[[case_idx, case, cfg['patch_shape'], num_patches_per_img, norm_type]
                  for case_idx, case in enumerate(dataset_)],
            num_threads=12)

        all_instructions = list(itertools.chain.from_iterable(cases_instructions))

        gen = acg.generators.construct_dataloader(
            dataset=acg.generators.InstructionDataset(
                instructions=all_instructions, data=dataset_, get_item_func=extract_train_sample),
            batch_size=cfg['batch_size'],
            shuffle=True)
        return gen

    train_dataset, val_dataset = nl.split_list(dataset, fraction=cfg['train_fraction'])

    num_patches_train = int(cfg['num_patches'] * cfg['train_fraction'])
    print("Building training patch generator...")
    train_gen = build_generators(train_dataset, num_patches=num_patches_train, norm_type=cfg['norm_type'])

    num_patches_val = int(cfg['num_patches'] * (1.0 - cfg['train_fraction']))
    print("Building validation patch generator...")
    val_gen = build_generators(val_dataset, num_patches=num_patches_val, norm_type=cfg['norm_type'])

    # store_patch_nifti(train_gen)
    # raise NotImplementedError

    if cfg['norm_type'] == 'minmax':
        inpainter_activation = nn.Tanh()
    elif cfg['norm_type'] == '0to1':
        inpainter_activation = nn.Sigmoid()
    else:
        inpainter_activation = nn.Identity()
    
    model_def = WNet_LITS_v3(detach_inpainter=cfg['detach_inpainter'], inpainter_activation=inpainter_activation)

    loss_func = WNet_LITS_loss(
        seg_loss=cfg['seg_loss'],
        rec_loss=cfg['rec_loss'],
        rec_w=cfg['rec_weight'])

    trainer = nl.net.train.Trainer(
        max_epochs=cfg['max_epochs'],
        loss_func=loss_func,
        optimizer=torch.optim.Adadelta,
        optimizer_opts=cfg['optimizer_opts'],
        train_metrics=cfg['train_metrics'],
        val_metrics=cfg['val_metrics'],
        plugins=[
            nl.net.train.Logger(
                filepath=os.path.join(
                    nl.get_base_path(fpath_trained_model),
                    '{}_log.csv'.format(nl.get_filename(fpath_trained_model, extension=False)))),
            nl.net.train.ProgressBar(print_interval=0.4),
            nl.net.train.EarlyStopping(
                metric_name=cfg['early_stopping_metric'], patience=cfg['patience'], mode='min'),
            nl.net.train.ModelCheckpoint(
                filepath=fpath_trained_model, save='best', metric_name=cfg['checkpoint_metric'], mode='min')],
        device='cuda',
        multigpu=False)

    return trainer.train(model_def, train_gen, val_gen)



def extract_LITS_test_patch(center, data, patch_shape, normalize_params):
    """
    Data is a tuple (t1_hole, lesion_mask)
    """
    t1_hole, lesion_mask = data[0], data[1]

    t1_patch = acg.patch.get_patch(t1_hole, center, patch_shape[1:])
    lesion_patch = acg.patch.get_patch(lesion_mask, center, patch_shape[1:])

    # Normalize image_patch
    t1_patch = normalize_LITS(t1_patch, normalize_params)

    # Put lesion patch between -1 and 1
    lesion_patch = (lesion_patch - 0.5) * 2.0

    # Concatenate image and lesion patches
    input_numpy = np.stack([t1_patch, lesion_patch], axis=0)
    input_torch = torch.tensor(np.ascontiguousarray(input_numpy), dtype=torch.float32)

    return input_torch


def postprocess_LITS_test_patch(patch_out):
    """Remember patch_out has dimension (BS, CH, X, Y, Z)"""
    # ACTIVATE with softmax
    patch_activated = f.softmax(patch_out, dim=1)
    return patch_activated


def segment_image_LITS(t1_image,
                       lesion_mask,
                       trained_model,
                       in_shape,
                       out_shape,
                       extraction_step,
                       norm_type,
                       postprocessing=False,
                       verbose=True,
                       normalize_params=None):

    assert t1_image.ndim == 3 and lesion_mask.ndim == 3

    # Obtain t1 with holes image
    t1_hole = copy.deepcopy(t1_image) *  (lesion_mask <= 0.0).astype(np.float)
    lesion_mask = (lesion_mask > 0.0).astype(np.float)

    # Get normalize params from t1 already with holes
    if normalize_params is None:
        normalize_params = get_normalize_params_LITS(t1_hole, mask=t1_hole > 0.0, norm_type=norm_type)
    else:
        pass
        # print(f'Given norm params    {normalize_params}')
        # print(f'Computed norm params {get_normalize_params_LITS(t1_hole, mask=t1_hole > 0.0)}')

    # Input image is already hole
    inference_segmentation = acg.inference.inference_image_patches(
        image=np.stack([t1_hole, lesion_mask], axis=0),
        model=trained_model,
        patch_shape_in=in_shape,
        patch_shape_out=out_shape,
        step=extraction_step,
        batch_size=16,
        device='cuda',
        extract_patch_func=lambda center, data : extract_LITS_test_patch(center, data, in_shape, normalize_params),
        postprocess_patch_func=postprocess_LITS_test_patch,
        verbose=verbose)

    # Perform brain tissue post processing
    if postprocessing:
        raise NotImplementedError

    return inference_segmentation


def segment_LITS_dataset(cfg_run, test_dataset, trained_model, seg_workspace, decimals=5):
    cfg = copy.deepcopy(cfg_default)
    cfg.update(cfg_run)

    print('\nSegmenting {} test images...'.format(len(test_dataset)))
    print('  patch shape: {}'.format(cfg['patch_shape']))
    print('  norm type: {}'.format(cfg['norm_type']))
    
    if isinstance(trained_model, str):
        trained_model = torch.load(trained_model)

    def segment(t1_brain_fpath, lesion_mask_fpath, probs_fpath_out, rerun_=False):
        if os.path.isfile(probs_fpath_out) and not rerun_:
            return

        t1_brain_nifti = nib.load(t1_brain_fpath)
        t1_brain = t1_brain_nifti.get_fdata()

        if lesion_mask_fpath is None:
            lesion_mask = np.zeros_like(t1_brain)
        else:
            lesion_mask = (nib.load(lesion_mask_fpath).get_fdata() > 0.0).astype(float)

        #start = time.time()
        probs = segment_image_LITS(
            t1_image=t1_brain,
            lesion_mask=lesion_mask,
            trained_model=trained_model,
            in_shape=(2,) + cfg['patch_shape'],
            out_shape=(4,) + cfg['patch_shape'],
            extraction_step=cfg['extraction_step'],
            norm_type=cfg['norm_type'],
            postprocessing=False,
            verbose=False)
        #print(f'Took {time.time() - start:.3f} seconds')

        # Prepare for storage and save
        probs = np.transpose(probs, (1, 2, 3, 0))  # Put channels on last position
        probs = np.round(probs, decimals)  # round a bit to avoid gigantic files
        nib.Nifti1Image(probs, t1_brain_nifti.affine, t1_brain_nifti.header).to_filename(probs_fpath_out)


    rta = acg.time_utils.RemainingTimeEstimator(len(test_dataset))
    for n, case in enumerate(test_dataset):
        # Segment once without any lesion mask
        segment(
            t1_brain_fpath=case['t1_brain_fpath'],
            lesion_mask_fpath=None,
            probs_fpath_out=os.path.join(seg_workspace, '{}__probs.nii.gz'.format(case['id'])))

        # For each lesion mask in dataset
        for lesion_id, lesion_mask_fp in case['lesion_mask_fpaths'].items():
            segment(
                t1_brain_fpath=case['t1_brain_fpath'],
                lesion_mask_fpath=lesion_mask_fp,
                probs_fpath_out=os.path.join(seg_workspace,
                                             f'{case["id"]}__lesion__{lesion_id}__probs.nii.gz'),
                rerun_=False
            )

        acg.print_utils.print_progress_bar(n, len(test_dataset) + 1, suffix=' images segmented - ETA: {} - {}'.format(
            rta.update(n), case['id']))
    acg.print_utils.print_progress_bar(len(test_dataset) - 1, len(test_dataset),
                       suffix=' images segmented - ETA: {}'.format(rta.elapsed_time()))






if __name__ == '__main__':
    import nibabel as nib
    import matplotlib.pyplot as plt
    pass
    # a = nib.load('/home/albert/Desktop/docker_tests/xnat_vicorob_E00365/t1-preprocessed.nii.gz').get_fdata()
    #
    # norm_params_ = get_normalize_params_LITS(a, mask=(a > 0))
    # a1 = normalize_LITS(a, norm_params_)
    #
    # bins = np.arange(-3.0, 3.0, 0.05)
    # plt.hist(a1[np.nonzero(a)], bins=bins)
    # plt.show()