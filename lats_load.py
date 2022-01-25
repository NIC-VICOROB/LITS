import copy
import json
import os
import time

import nibabel as nib
import numpy as np
import random

import niclib as nl

import scipy.sparse as sp
import acglib as acg

class BinaryMaskStorage4D:
    """
    Class to store 4 dimensional sparse matrices and acces slices of them
    """
    def __init__(self, dense):
        assert len(dense.shape) == 4
        self.dense_shape = dense.shape
        self.d = [[sp.dok_matrix(x, dtype=np.uint8) for x in channels] for channels in dense]
        del dense
        
    def get_dense(self, slcs=None):
        slcs = (slice(None),) * 4 if slcs is None else slcs
        assert len(slcs) == 4
        return \
            np.stack([np.stack([s_x[slcs[2:]].toarray() for s_x in s_ch[slcs[1]]], 0) for s_ch in self.d[slcs[0]]], 0)

    def get_patch(self, center, patch_shape):
        slcs = acg.patch.get_patch_slices(len(self.dense_shape), center, patch_shape)
        assert len(slcs) == 4
        return \
            np.stack([np.stack([s_x[slcs[2:]].toarray() for s_x in s_ch[slcs[1]]], 0) for s_ch in self.d[slcs[0]]], 0)


def list_dir_paths(p):
    return sorted([f.path for f in os.scandir(p) if f.is_dir()])

def list_filepaths(p):
    return sorted([f.path for f in os.scandir(p) if f.is_file()])

def list_filenames(p):
    return sorted([f.name for f in os.scandir(p) if f.is_file()])


def remove_ext(fp):
    return fp.split('.', 1)[0]

def as_float16(arr):
    if np.max(arr) >= np.finfo(np.float16).max:
        arr = arr / (np.max(arr) / (np.finfo(np.float16).max + 1))
    return arr.astype(np.float16)

def get_campinas_train_test_ids(campinas_dataset_path):
    campinas_ids = [cp.split(os.sep)[-1] for cp in list_dir_paths(campinas_dataset_path)]

    philips_ids = [cid for cid in campinas_ids if cid.split('_')[1] == 'philips']
    ge_ids = [cid for cid in campinas_ids if cid.split('_')[1] == 'ge']
    siemens_ids = [cid for cid in campinas_ids if cid.split('_')[1] == 'siemens']

    # Get last 10 cases from each scanner for healthy tests
    num_test = 15
    train_ids = philips_ids[:-num_test] + ge_ids[:-num_test] + siemens_ids[:-num_test]
    test_ids = philips_ids[-num_test:] + ge_ids[-num_test:] + siemens_ids[-num_test:]

    return train_ids, test_ids


def get_challenge2016_train_test_ids(challenge2016_dataset_path):
    challenge2016_ids = ['01016SACH', '01038PAGU', '01039VITE', '01040VANE', '01042GULE',
                         '07001MOEL', '07003SATH', '07010NABO', '07040DORE', '07043SEME',
                         '08002CHJE', '08027SYBR', '08029IVDI', '08031SEVE', '08037ROGU']

    return challenge2016_ids[:-3], challenge2016_ids[-3:]


def get_wmh2017_train_test_ids(wmh2017_dataset_path):
    wmh2017_ids = [cp.split(os.sep)[-1] for cp in list_dir_paths(wmh2017_dataset_path)]

    train_fraction = 0.9
    num_test = int(np.ceil(len(wmh2017_ids) * (1.0 - train_fraction)))

    return wmh2017_ids[:-num_test], wmh2017_ids[-num_test:]


def get_isbi2015_train_test_ids(isbi2015_dataset_path):
    isbi2015_train_ids = ['training01_01_mask1', 'training01_02_mask1', 'training01_03_mask1',
                          'training01_04_mask1', 'training02_01_mask1', 'training02_02_mask1',
                          'training02_03_mask1', 'training02_04_mask1', 'training03_01_mask1',
                          'training03_02_mask1', 'training03_03_mask1', 'training03_04_mask1',
                          'training03_05_mask1']

    isbi2015_test_ids = ['training04_01_mask1', 'training04_02_mask1', 'training04_03_mask1',
                         'training04_04_mask1', 'training05_01_mask1', 'training05_02_mask1',
                         'training05_03_mask1', 'training05_04_mask1']

    return isbi2015_train_ids, isbi2015_test_ids



def load_healthy_lesion_dataset(healthy_lesion_path, healthy_ids, lesion_ids, load_images=False, max_lesion_masks_per_healthy=None):
    """
    Loads a healthy lesion dataset.
        It will load the t1 images and segmentation from healthy_ids.
        It will load the binary lesion masks registered to the healthy case from lesion_ids.
    Then it will apply a subset of lesion masks to each healthy case, adding holes with value 0.

    The number of returned cases will be= healthy_ids * (lesion_masks_per_healthy + 1)

    Each case will be a dictionary with the entries:
        't1_fpath', 'lesion_mask_fpath', 'normFOV_fpath', 'probs_fpaths'
    If load_images, then it will additionally have:
        't1', 'lesion_mask', 'normFOV', 'probs'
    """

    print(f'Loading healthy_lesion dataset: {len(healthy_ids)} cases with {max_lesion_masks_per_healthy} masks each...')

    case_paths = list_dir_paths(healthy_lesion_path)
    case_paths = [case_path_ for case_path_ in case_paths if case_path_.split(os.sep)[-1] in healthy_ids]
    arguments = [[n_, case_path_] for n_, case_path_ in enumerate(case_paths)]

    def load_case(n, case_path):
        nl.print_progress_bar(n, len(arguments))

        healthy_id = case_path.split(os.sep)[-1]

        # Get all lesion mask filenames
        mask_filenames = [mfn for mfn in list_filenames(case_path)
                          if mfn.startswith('lesion_mask_') and mfn.endswith('.nii.gz')]

        # Filter filenames by id and generate entire filepath
        mask_filepaths = [os.path.join(case_path, mfn) for mfn in mask_filenames
                          if remove_ext(mfn).split('_', 2)[-1] in lesion_ids]

        #   Randomly get a subset of size max_lesion_masks_per_healthy from mask_filepaths
        if max_lesion_masks_per_healthy is not None:
            random.shuffle(mask_filepaths)
            mask_filepaths = mask_filepaths[:max_lesion_masks_per_healthy]

        mask_ids = [mask_filepath.split(os.sep)[-1].split('.')[0].split('_', 2)[-1] for mask_filepath in mask_filepaths]

        # Load the healthy images ONCE for this healthy case
        t1_fpath = os.path.join(case_path, 't1.nii.gz')
        t1_brain_fpath = os.path.join(case_path, 't1_brain.nii.gz')
        probs_fpaths = [os.path.join(case_path, f't1_brain_fast_{n}.nii.gz') for n in [0, 1, 2]]
        avg_lesion_mask_fpath = os.path.join(case_path, 'average_lesion_mask.nii.gz')

        case = {
            'id': healthy_id,
            't1_fpath': t1_fpath,
            't1_brain_fpath': t1_brain_fpath,
            'lesion_mask_fpaths': {k: v for k, v in zip(mask_ids, mask_filepaths)},
            'probs_fpaths': probs_fpaths,
            'average_lesion_mask_fpath': avg_lesion_mask_fpath
        }

        if load_images:
            t1_brain_nifti = nib.load(t1_brain_fpath)
            t1_brain = as_float16(np.expand_dims(t1_brain_nifti.get_fdata(), axis=0))
            t1_brain_nifti.uncache()
            
            probs_nifti = [nib.load(pfp) for pfp in probs_fpaths]
            probs = [pn.get_fdata() for pn in probs_nifti]
            bg_prob = 1.0 - np.add(probs[0], np.add(probs[1], probs[2]))
            probs.insert(0, bg_prob)
            probs = np.stack(probs, axis=0).astype(np.float16)
            [pn.uncache() for pn in probs_nifti]
            
            lesion_masks = {lid: BinaryMaskStorage4D(np.expand_dims(nib.load(lfp).get_fdata(), axis=0))
                            for lid, lfp in case['lesion_mask_fpaths'].items()}

            case.update({
                't1_brain': t1_brain,
                'lesion_masks': lesion_masks,
                'probs': probs
            })

        return case

    dataset = nl.parallel_load(load_case, arguments, num_workers=10)

    print(f'Loaded dataset with {len(dataset)} cases')
    return dataset



def load_healthy_lesion_dataset_old(healthy_lesion_path, healthy_ids, lesion_ids, max_lesion_masks_per_healthy,
                                add_nomask_case=False, load_images=False):
    """
    Loads a healthy lesion dataset.
        It will load the t1 images and segmentation from healthy_ids.
        It will load the binary lesion masks registered to the healthy case from lesion_ids.
    Then it will apply a subset of lesion masks to each healthy case, adding holes with value 0.

    The number of returned cases will be= healthy_ids * (lesion_masks_per_healthy + 1)

    Each case will be a dictionary with the entries:
        't1_fpath', 'lesion_mask_fpath', 'normFOV_fpath', 'probs_fpaths'
    If load_images, then it will additionally have:
        't1', 'lesion_mask', 'normFOV', 'probs'
    """

    print('Loading healthy_lesion dataset...')

    case_paths = list_dir_paths(healthy_lesion_path)
    case_paths = [case_path_ for case_path_ in case_paths if case_path_.split(os.sep)[-1] in healthy_ids]
    arguments = [[n_, case_path_] for n_, case_path_ in enumerate(case_paths)]

    def load_case(n, case_path):
        nl.print_progress_bar(n, len(arguments))

        healthy_id = case_path.split(os.sep)[-1]

        # Get all lesion mask filenames
        mask_filenames = [mfn for mfn in list_filenames(case_path)
                          if mfn.startswith('lesion_mask_') and mfn.endswith('.nii.gz')]

        # Filter filenames by id and generate entire filepath
        mask_filepaths = [os.path.join(case_path, mfn) for mfn in mask_filenames
                          if remove_ext(mfn).split('_', 2)[-1] in lesion_ids]

        #   Randomly get a subset of size max_lesion_masks_per_healthy from mask_filepaths
        if max_lesion_masks_per_healthy is not None:
            random.shuffle(mask_filepaths)
            mask_filepaths = mask_filepaths[:max_lesion_masks_per_healthy]

        # Load the healthy images ONCE for this healthy case
        t1, normFOV, probs = None, None, None
        t1_fpath = os.path.join(case_path, 't1.nii.gz')
        t1_brain_fpath = os.path.join(case_path, 't1_brain.nii.gz')
        normFOV_fpath = os.path.join(case_path, 't1_brain_mask.nii.gz')
        probs_fpaths = [os.path.join(case_path, f't1_brain_fast_{n}.nii.gz') for n in [0, 1, 2]]

        if load_images:
            t1 = as_float16(np.expand_dims(nib.load(t1_fpath).get_fdata(), axis=0))
            normFOV = np.expand_dims(nib.load(normFOV_fpath).get_fdata(), axis=0).astype(np.float16)

            probs = [nib.load(pfp).get_fdata() for pfp in probs_fpaths]
            bg_prob = 1.0 - np.add(probs[0], np.add(probs[1], probs[2]))
            probs.insert(0, bg_prob)
            probs = np.stack(probs, axis=0).astype(np.float16)

        # Generate a case for no mask, and then one for each mask filepath
        subject_cases = []

        if add_nomask_case: # Add case with no masks
            case = {
                'id': healthy_id,
                't1_fpath': t1_fpath,
                't1_brain_fpath': t1_brain_fpath,
                'lesion_mask_fpath': '',
                'normFOV_fpath': normFOV_fpath,
                'probs_fpaths': probs_fpaths,}
            if load_images:
                case.update({
                    't1': t1,
                    'normFOV': normFOV,
                    'lesion_mask': np.zeros_like(t1),
                    'probs': probs})

            subject_cases.append(case)

        for mask_filepath in mask_filepaths:
            lesion_id = mask_filepath.split(os.sep)[-1].split('.')[0].split('_', 2)[-1]

            case = {
                'id': healthy_id + '__lesion__' + lesion_id,
                't1_fpath': t1_fpath,
                't1_brain_fpath': t1_brain_fpath,
                'lesion_mask_fpath': mask_filepath,
                'normFOV_fpath': normFOV_fpath,
                'probs_fpaths': probs_fpaths}

            if load_images:
                lesion_mask = np.expand_dims(nib.load(case['lesion_mask_fpath']).get_fdata(), axis=0).astype(np.float16)
                case.update({
                    't1': (copy.deepcopy(t1) * (1.0 - lesion_mask)).astype(np.float16),
                    'normFOV': normFOV,
                    'lesion_mask': lesion_mask,
                    'probs': probs})

            subject_cases.append(case)

        return subject_cases

    dataset_subject_cases = nl.parallel_load(load_case, arguments, num_workers=10)

    dataset = []
    for subject_cases_ in dataset_subject_cases:
        dataset += subject_cases_

    print(f'Loaded dataset with {len(dataset)} cases')
    return dataset