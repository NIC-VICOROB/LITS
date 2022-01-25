import copy
import itertools

import numpy as np
import torch
from torch import nn
import os
import nibabel as nib

import niclib as nl
from niclib.generator import BalancedSampling, make_generator, PatchSet, ZipSet

from .model_utils import SUNETx4, PatchSet_LATS

import acglib as acg

cfg_default = {
    'patch_shape': (32, 32, 32),

    'num_patches': 150000,
    'train_fraction': 0.9,
    'balanced_fraction': 0.5,
    'weighted_fraction': 0.5,
    'norm_type': 'minmax',

    'seg_loss': nl.loss.LossWrapper(nn.CrossEntropyLoss(),
                                     preprocess_fn=lambda out, tgt: (out, torch.argmax(tgt, dim=1))),
    'dropout': 0.0,

    'optimizer_opts': dict(lr=0.2),
    'max_epochs': 200,
    'batch_size': 32,
    'train_metrics': {},
    'val_metrics': {},

    'early_stopping_metric': 'loss',
    'patience': 10,

    'checkpoint_metric': 'loss'
}


def get_instructions(case_idx, case, patch_shape, num_balanced, num_weighted):
    # Get normalization parameters
    normalize_params = \
        acg.generators.get_normalize_params(case['t1_brain'], 'minmax', mask=case['t1_brain'] > 0)
    # Get patch centers
    balanced_centers = acg.patch.sample_centers_balanced(
        np.argmax(case['probs'], axis=0), patch_shape, num_balanced, add_rand_offset=True, exclude=[0])

    if num_weighted > 0:
        weighted_centers = acg.patch.sample_centers_weighted(
            weights_image=nib.load(case['average_lesion_mask_fpath']).get_fdata().astype(np.float16),
            num_centers=num_weighted,
            patch_shape=patch_shape,
            add_rand_offset=True)
    else:
        weighted_centers = []

    all_centers = balanced_centers + weighted_centers
    # Get lesion keys in list
    lesion_keys = list(case['lesion_masks'].keys())
    # Build instructions
    instructions = []
    for n, center in enumerate(all_centers):
        instructions.append({
            'case_idx': case_idx,
            'lesion_key': lesion_keys[n % len(lesion_keys)] if len(lesion_keys) > 0 else None,
            'center': center,
            'patch_shape': patch_shape,
            'normalize_params': normalize_params})
    return instructions


def extract_train_sample(instr, data):
    case = data[instr['case_idx']]

    ### INPUT PATCH
    # Extract image patch
    image_patch = acg.patch.get_patch(case['t1_brain'], instr['center'], instr['patch_shape'])
    # Extract lesion patch
    if instr['lesion_key'] is not None:
        lesion_mask = case['lesion_masks'][instr['lesion_key']]
        lesion_patch = copy.deepcopy(lesion_mask.get_patch(instr['center'], instr['patch_shape'])).astype(image_patch.dtype)
    else:
        lesion_patch = np.zeros_like(image_patch)
    # Apply lesion mask to image patch
    image_patch = image_patch * (lesion_patch == 0).astype(np.float)
    # Normalize image_patch by minmax
    image_patch = acg.generators.normalize(image_patch, *instr['normalize_params'])
    # Put lesion patch between -1 and 1
    lesion_patch = (lesion_patch - 0.5) * 2.0
    # Concatenate image and lesion patches
    input_numpy = np.concatenate([image_patch, lesion_patch], axis=0)

    ### TARGET PATCH
    tgt_numpy = acg.patch.get_patch(case['probs'], instr['center'], instr['patch_shape'])

    # Convert to torch tensors
    input_torch = torch.tensor(np.ascontiguousarray(input_numpy), dtype=torch.float32)
    tgt_torch = torch.tensor(np.ascontiguousarray(tgt_numpy), dtype=torch.float32)
    return input_torch, tgt_torch



def train_calbaq_lats(cfg_run, dataset, fpath_trained_model):
    cfg = copy.deepcopy(cfg_default)
    cfg.update(cfg_run)
    print("\nTraining CALBAQ-DL for LATS")

    def build_generators(dataset_, num_patches):
        num_patches_balanced = num_patches * cfg['balanced_fraction']
        num_patches_weighted = num_patches * cfg['weighted_fraction']

        balanced_per_img = int(np.round(num_patches_balanced / len(dataset_)))
        weighted_per_img = int(np.round(num_patches_weighted / len(dataset_)))

        cases_instructions = acg.parallel_run(
            func=get_instructions,
            args=[[case_idx, case, cfg['patch_shape'], balanced_per_img, weighted_per_img]
                  for case_idx, case in enumerate(dataset_)],
            num_threads=12)

        all_instructions = list(itertools.chain.from_iterable(cases_instructions))

        gen = acg.generators.construct_dataloader(
            dataset=acg.generators.InstructionDataset(
                instructions=all_instructions,data=dataset_, get_item_func=extract_train_sample),
            batch_size=cfg['batch_size'],
            shuffle=True)

        return gen

    train_dataset, val_dataset = nl.split_list(dataset, fraction=cfg['train_fraction'])

    num_patches_train = int(cfg['num_patches'] * cfg['train_fraction'])
    print("Building training patch generator...")
    train_gen = build_generators(train_dataset, num_patches=num_patches_train)

    num_patches_val = int(cfg['num_patches'] * (1.0 - cfg['train_fraction']))
    print("Building validation patch generator...")
    val_gen = build_generators(val_dataset, num_patches=num_patches_val)

    model_def = SUNETx4(in_ch=2, out_ch=4, activation=None, dropout_rate=0.0)

    loss_func = cfg['seg_loss']

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



def train_calbaq_lats_old(cfg_run, dataset, fpath_trained_model):
    cfg = copy.deepcopy(cfg_default)
    cfg.update(cfg_run)
    print("\nTraining CALBAQ-DL for LATS")

    def build_generators(dataset_, num_patches):
        # Get images out
        t1_brains = [case['t1_brain'] for case in dataset_]
        lesion_masks = [case['lesion_masks'] for case in dataset_]
        t1_probs = [case['probs'] for case in dataset_]

        # Load the average lesion masks for each case,
        avg_lesion_masks = [nib.load(case['average_lesion_mask_fpath']).get_fdata().astype(np.float16)
                            for case in dataset_]

        num_patches_balanced = num_patches * cfg['balanced_fraction']
        num_patches_weighted = num_patches * cfg['weighted_fraction']

        ### BALANCED SAMPLING
        num_balanced_patches_per_image = int(np.round(num_patches_balanced / len(dataset_)))
        def sample_centers_balanced(t1, t1_prob):
            argmax_labels = [np.argmax(t1_prob, axis=0).astype(np.uint8)]
            center_sampling = BalancedSampling(argmax_labels, num_balanced_patches_per_image, add_rand_offset=True)
            t1_patch_centers = center_sampling.sample_centers([t1], cfg['patch_shape'])
            del argmax_labels, center_sampling
            return t1_patch_centers[0]
        balanced_centers = nl.parallel_load(
            load_func=sample_centers_balanced,
            arguments=[[t1_, t1_prob_] for t1_, t1_prob_ in zip(t1_brains, t1_probs)],
            num_workers=12)

        ### WEIGHTED SAMPLING
        num_weighted_patches_per_image = int(np.round(num_patches_weighted / len(dataset_)))
        weighted_centers = nl.parallel_load(
            load_func=sample_patch_centers_weighted,
            arguments=[[alm, num_weighted_patches_per_image, cfg['patch_shape']] for alm in avg_lesion_masks],
            num_workers=12)

        # Build and return generator
        patch_centers = [bc + wc for bc, wc in zip(balanced_centers, weighted_centers)]

        gen = make_generator(
            set=ZipSet([
                    PatchSet_LATS(
                        t1_brains, lesion_masks, cfg['patch_shape'], patch_centers, norm_type=cfg['norm_type']),
                    PatchSet(t1_probs, cfg['patch_shape'], None, normalize='none', centers=patch_centers)]),
            batch_size=cfg['batch_size'],
            shuffle=True)

        return gen

    train_dataset, val_dataset = nl.split_list(dataset, fraction=cfg['train_fraction'])

    num_patches_train = int(cfg['num_patches'] * cfg['train_fraction'])
    print("Building training patch generator...")
    train_gen = build_generators(train_dataset, num_patches=num_patches_train)

    num_patches_val = int(cfg['num_patches'] * (1.0 - cfg['train_fraction']))
    print("Building validation patch generator...")
    val_gen = build_generators(val_dataset, num_patches=num_patches_val)

    model_def = SUNETx4(in_ch=2, out_ch=4, activation=None, dropout_rate=0.0)

    loss_func = cfg['seg_loss']

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

