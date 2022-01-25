import copy
import os
import shutil
import subprocess
import time

import niclib as nl
import nibabel as nib
import numpy as np
import torch

import traceback

import copy
import nibabel as nib

import numpy as np
from numpy.random import default_rng


#from .use_utils import segment_tissue_LATS

cfg_default = {
    'patch_shape': (32, 32, 32),
    'norm_type': 'minmax'
}

def list_dir_paths(p):
    return sorted([f.path for f in os.scandir(p) if f.is_dir()])


def generate_artificial_lesion(t1, lesion_mask, tissue_labels):
    # Compute the lesion intensities distribution parameters
    gm_mean = np.mean(t1[tissue_labels == 2])
    wm_mean = np.mean(t1[tissue_labels == 3])
    
    gm_wm_mean = np.mean([gm_mean, wm_mean])
    gm_wm_std = (wm_mean - gm_mean) / 4.0
    
    # Generate lesion intensities according to normal distribution
    lesion_intensities = default_rng().normal(loc=gm_wm_mean, scale=gm_wm_std, size=lesion_mask.shape)

    # Overwrite the intensities on the original image
    t1_brain_lesion = copy.deepcopy(t1)
    t1_brain_lesion[lesion_mask == 1] = lesion_intensities[lesion_mask == 1]

    return t1_brain_lesion



def run_fast(t1_nifti, probs_out_filepath, tmp_path, decimals = 4, erase_tmp=True):
    os.makedirs(tmp_path, exist_ok=True)

    filepath_in = os.path.join(tmp_path, 't1.nii.gz')
    t1_nifti.to_filename(filepath_in)

    print('Running FAST: {}'.format(filepath_in))
    subprocess.check_call(['bash', '-c', 'fast {}'.format(filepath_in)])

    fast_pves_fpaths = [nl.remove_extension(filepath_in) + '_pve_{}.nii.gz'.format(i) for i in range(3)]

    fast_pves_niftis = [nib.load(fp) for fp in fast_pves_fpaths]
    fast_pves = [fpn.get_data() for fpn in fast_pves_niftis]
    fast_pves.insert(0, 1.0 - np.sum(np.stack(fast_pves, axis=0), axis=0))  # Background probability

    # Prepare for storage and save
    probs = np.stack(fast_pves, axis=0)
    probs = np.transpose(probs, (1, 2, 3, 0))  # Put channels on last position
    probs = np.round(probs, decimals)  # round a bit to avoid gigantic files
    nib.Nifti1Image(probs, fast_pves_niftis[0].affine, fast_pves_niftis[0].header).to_filename(probs_out_filepath)

    if erase_tmp:
        shutil.rmtree(tmp_path)


def store_non_wm_lesions(lesion_mask_fpath, wm_mask_fpath, filepath_out):
    lesions_nifti = nib.load(lesion_mask_fpath)
    wm_mask_nifti = nib.load(wm_mask_fpath)
    lesion_mask = lesions_nifti.get_fdata()
    wm_mask = wm_mask_nifti.get_fdata()

    from scipy import ndimage
    lesions_comp, num_labels = ndimage.measurements.label(lesion_mask, structure=np.ones((3, 3, 3)))
    dil_lesions_comp = ndimage.grey_dilation(lesions_comp, structure=np.ones((3, 3, 3))).astype(float) - 1.0

    dil_lesions = ndimage.binary_dilation(lesion_mask, structure=np.ones((3, 3, 3)))
    outborder = np.logical_xor(dil_lesions, lesion_mask > 0).astype(float)

    wm_labels = np.unique(dil_lesions_comp * outborder * wm_mask)
    all_labels = np.arange(1, num_labels + 1)
    non_wm_labels = np.setdiff1d(all_labels, wm_labels)

    non_wm_lesions = np.zeros_like(lesion_mask)
    for non_wm_label in non_wm_labels:
        non_wm_lesion_mask = ndimage.binary_erosion(dil_lesions_comp == non_wm_label, structure=np.ones((3, 3, 3)))
        non_wm_lesions[non_wm_lesion_mask] = 1.0

    nl.save_nifti(
        filepath=filepath_out,
        volume=non_wm_lesions,
        reference=lesions_nifti)

    return len(non_wm_labels)



def fill_prados_artificial_lesion_and_segment_FAST(
    t1_filepath, lesion_mask_fpath, healthy_tissue_probs, path_out, probs_out_filepath, decimals = 4):

    t1_nifti = nib.load(t1_filepath)
    lesion_mask = nib.load(lesion_mask_fpath).get_fdata()

    ### Paint artificial lesion with healthy_tissue_labels
    t1_lesion_fpath = os.path.join(path_out, 't1_lesion.nii.gz')
    t1_lesion = generate_artificial_lesion(
        t1=t1_nifti.get_fdata(),
        lesion_mask=lesion_mask,
        tissue_labels=np.argmax(healthy_tissue_probs, axis=0))
    t1_lesion_nifti = nib.Nifti1Image(t1_lesion, t1_nifti.affine, t1_nifti.header)

    os.makedirs(path_out, exist_ok=True)
    t1_lesion_nifti.to_filename(t1_lesion_fpath)

    start = time.time()

    ### Run niftyseg lesion_filling
    t1_filled_fpath = os.path.join(path_out, 't1_lesion_filled.nii.gz')
    lesion_filling_cmd = f'seg_FillLesions -i {t1_lesion_fpath} -l {lesion_mask_fpath} -o {t1_filled_fpath}'
    print(lesion_filling_cmd)
    try:
        subprocess.check_call(['bash', '-c', lesion_filling_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except Exception:
        print('Error with case {}'.format(t1_lesion_fpath))
        return

    ### Run FAST on artifical lesion filled t1
    run_fast(
        t1_nifti=nib.load(t1_filled_fpath),
        probs_out_filepath=probs_out_filepath,
        tmp_path=os.path.join(path_out, 'fast_lesion_filled'),
        decimals=decimals,
        erase_tmp=True)

    print(f'Took {time.time() - start:.3f} seconds')




def fill_valverde_artificial_lesion_and_segment_FAST(
    t1_filepath, lesion_mask_fpath, healthy_tissue_probs, path_out, probs_out_filepath, decimals = 4):

    t1_nifti = nib.load(t1_filepath)
    lesion_mask = nib.load(lesion_mask_fpath).get_fdata()

    ### Paint artificial lesion with healthy_tissue_labels
    t1_lesion_fpath = os.path.join(path_out, 't1_lesion.nii.gz')
    t1_lesion = generate_artificial_lesion(
        t1=t1_nifti.get_fdata(),
        lesion_mask=lesion_mask,
        tissue_labels=np.argmax(healthy_tissue_probs, axis=0))
    t1_lesion_nifti = nib.Nifti1Image(t1_lesion, t1_nifti.affine, t1_nifti.header)
    t1_lesion_nifti.to_filename(t1_lesion_fpath)

    start = time.time()

    ### Segment FAST and fill artificial lesion with artificial_tissue_labels
    t1_lesion_probs_fpath = os.path.join(path_out, 't1_lesion_probs.nii.gz')
    run_fast(
        t1_nifti=t1_lesion_nifti,
        probs_out_filepath=t1_lesion_probs_fpath,
        tmp_path=os.path.join(path_out, 'fast_lesion'),
        decimals=decimals,
        erase_tmp=True)

    ### Get wm mask from probs and store
    t1_lesion_wm_mask = (np.argmax(nib.load(t1_lesion_probs_fpath).get_fdata(), axis=-1) == 3).astype(np.int16)

    t1_lesion_wm_mask_fpath = os.path.join(path_out, 't1_lesion_wm_mask.nii.gz')
    nib.Nifti1Image(t1_lesion_wm_mask, t1_nifti.affine, t1_nifti.header).to_filename(t1_lesion_wm_mask_fpath)

    ### Run valverde lesion_filling
    t1_filled_fpath = os.path.join(path_out, 't1_lesion_filled.nii.gz')
    lesion_filling_cmd = 'lesion_filling -i {} -o {} -l {} -w {}'.format(
        t1_lesion_fpath, t1_filled_fpath, lesion_mask_fpath, t1_lesion_wm_mask_fpath)
    print(lesion_filling_cmd)

    try:
        subprocess.check_call(['bash', '-c', lesion_filling_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except Exception:
        print('Error with case {}'.format(t1_lesion_fpath))

        store_non_wm_lesions(
            lesion_mask_fpath=lesion_mask_fpath,
            wm_mask_fpath=t1_lesion_wm_mask_fpath,
            filepath_out=os.path.join(path_out, 'non_wm_lesions.nii.gz'))

        return

    ### Run FAST on artifical lesion filled t1
    run_fast(
        t1_nifti=nib.load(t1_filled_fpath),
        probs_out_filepath=probs_out_filepath,
        tmp_path=os.path.join(path_out, 'fast_lesion_filled'),
        decimals=decimals,
        erase_tmp=True)

    print(f'Took {time.time() - start:.3f} seconds')


def fill_artificial_lesion_and_segment_FAST(
        t1_filepath, lesion_mask_fpath, healthy_tissue_probs, path_out, probs_out_filepath, decimals = 4):
    os.makedirs(path_out, exist_ok=True)

    t1_nifti = nib.load(t1_filepath)
    lesion_mask = nib.load(lesion_mask_fpath).get_fdata()

    ### Paint artificial lesion with healthy_tissue_labels
    t1_lesion_fpath = os.path.join(path_out, 't1_lesion.nii.gz')
    t1_lesion = generate_artificial_lesion(
        t1=t1_nifti.get_fdata(),
        lesion_mask=lesion_mask,
        tissue_labels=np.argmax(healthy_tissue_probs, axis=0))
    t1_lesion_nifti = nib.Nifti1Image(t1_lesion, t1_nifti.affine, t1_nifti.header)
    t1_lesion_nifti.to_filename(t1_lesion_fpath)

    start = time.time()

    ### Segment FAST and fill artificial lesion with artificial_tissue_labels
    t1_lesion_probs_fpath = os.path.join(path_out, 't1_lesion_probs.nii.gz')
    run_fast(
        t1_nifti=t1_lesion_nifti,
        probs_out_filepath=t1_lesion_probs_fpath,
        tmp_path=os.path.join(path_out, 'fast_lesion'),
        decimals=decimals,
        erase_tmp=True)

    ### Get wm mask from probs and store
    t1_lesion_wm_mask = (np.argmax(nib.load(t1_lesion_probs_fpath).get_fdata(), axis=-1) == 3).astype(np.int16)

    t1_lesion_wm_mask_fpath = os.path.join(path_out, 't1_lesion_wm_mask.nii.gz')
    nib.Nifti1Image(t1_lesion_wm_mask, t1_nifti.affine, t1_nifti.header).to_filename(t1_lesion_wm_mask_fpath)

    ### Run FSL lesion_filling
    t1_filled_fpath = os.path.join(path_out, 't1_lesion_filled.nii.gz')
    lesion_filling_cmd = 'lesion_filling -i {} -o {} -l {} -w {}'.format(
        t1_lesion_fpath, t1_filled_fpath, lesion_mask_fpath, t1_lesion_wm_mask_fpath)
    print(lesion_filling_cmd)

    try:
        subprocess.check_call(['bash', '-c', lesion_filling_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except Exception:
        print('Error with case {}'.format(t1_lesion_fpath))

        store_non_wm_lesions(
            lesion_mask_fpath=lesion_mask_fpath,
            wm_mask_fpath=t1_lesion_wm_mask_fpath,
            filepath_out=os.path.join(path_out, 'non_wm_lesions.nii.gz'))

        return

    ### Run FAST on artifical lesion filled t1
    run_fast(
        t1_nifti=nib.load(t1_filled_fpath),
        probs_out_filepath=probs_out_filepath,
        tmp_path=os.path.join(path_out, 'fast_lesion_filled'),
        decimals=decimals,
        erase_tmp=True)

    print(f'Took {time.time() - start:.3f} seconds')



def segment_LATS_fast_artificial_filled(test_dataset, seg_workspace, decimals=5, rerun=False):
    print('\nFAST artificial filled: segmenting {} test images ...'.format(len(test_dataset)))

    def process_case(n, case):
        nl.print_progress_bar(n, len(test_dataset))

        # Load fast segmentation
        fast_pves_niftis = [nib.load(fp) for fp in case['probs_fpaths']]
        fast_pves = [fpn.get_data() for fpn in fast_pves_niftis]
        fast_pves.insert(0, 1.0 - np.sum(np.stack(fast_pves, axis=0), axis=0))  # Background probability
        probs = np.stack(fast_pves, axis=0)

        # Case with no mask, so we copy the original fast segmentation
        probs_fpath_out = os.path.join(seg_workspace, '{}__probs.nii.gz'.format(case['id']))
        if not os.path.isfile(probs_fpath_out):
            nib.Nifti1Image(
                dataobj=np.round(np.transpose(probs, (1, 2, 3, 0)), decimals),
                affine=fast_pves_niftis[0].affine,
                header=fast_pves_niftis[0].header
            ).to_filename(probs_fpath_out)

        # then, for each lesion mask in dataset
        for lesion_id, lesion_mask_fp in case['lesion_mask_fpaths'].items():
            subcase_id = f'{case["id"]}__lesion__{lesion_id}'

            probs_out_filepath = os.path.join(seg_workspace, subcase_id + '__probs.nii.gz')
            if not os.path.isfile(probs_out_filepath):
                fill_artificial_lesion_and_segment_FAST(
                    t1_filepath=case['t1_brain_fpath'],
                    lesion_mask_fpath=lesion_mask_fp,
                    healthy_tissue_probs=probs,
                    path_out=os.path.join(seg_workspace, 'segmentation_workspace', subcase_id),
                    probs_out_filepath=probs_out_filepath)

    args = [[n, case] for n, case in enumerate(test_dataset)]
    nl.parallel_load(process_case, args, num_workers=12)

def segment_LATS_fast_artificial_prados_filled(test_dataset, seg_workspace, decimals=5, rerun=False):
    print('\nFAST artificial PRADOS filled: segmenting {} test images ...'.format(len(test_dataset)))

    def process_case(n, case):
        nl.print_progress_bar(n, len(test_dataset))

        # Load fast segmentation
        fast_pves_niftis = [nib.load(fp) for fp in case['probs_fpaths']]
        fast_pves = [fpn.get_data() for fpn in fast_pves_niftis]
        fast_pves.insert(0, 1.0 - np.sum(np.stack(fast_pves, axis=0), axis=0))  # Background probability
        probs = np.stack(fast_pves, axis=0)

        # Case with no mask, so we copy the original fast segmentation
        probs_fpath_out = os.path.join(seg_workspace, '{}__probs.nii.gz'.format(case['id']))
        if not os.path.isfile(probs_fpath_out):
            nib.Nifti1Image(
                dataobj=np.round(np.transpose(probs, (1, 2, 3, 0)), decimals),
                affine=fast_pves_niftis[0].affine,
                header=fast_pves_niftis[0].header
            ).to_filename(probs_fpath_out)

        # then, for each lesion mask in dataset
        for lesion_id, lesion_mask_fp in case['lesion_mask_fpaths'].items():
            subcase_id = f'{case["id"]}__lesion__{lesion_id}'

            probs_out_filepath = os.path.join(seg_workspace, subcase_id + '__probs.nii.gz')
            if not os.path.isfile(probs_out_filepath):
                fill_prados_artificial_lesion_and_segment_FAST(
                    t1_filepath=case['t1_brain_fpath'],
                    lesion_mask_fpath=lesion_mask_fp,
                    healthy_tissue_probs=probs,
                    path_out=os.path.join(seg_workspace, 'segmentation_workspace', subcase_id),
                    probs_out_filepath=probs_out_filepath)

    args = [[n, case] for n, case in enumerate(test_dataset)]
    nl.parallel_load(process_case, args, num_workers=12)


def segment_LATS_fast_holes(test_dataset, seg_workspace, decimals=5, rerun=False):
    print('\nFAST HOLES: segmenting {} test images ...'.format(len(test_dataset)))

    def process_case(n, case):
        nl.print_progress_bar(n, len(test_dataset))

        tmp_folder = os.path.join(seg_workspace, f'{case["id"]}_fast_tmp')

        no_lesion_probs_fpath_out = os.path.join(seg_workspace, '{}__probs.nii.gz'.format(case['id']))
        if not os.path.isfile(no_lesion_probs_fpath_out) and not rerun:
            # Load fast segmentation
            fast_pves_niftis = [nib.load(fp) for fp in case['probs_fpaths']]
            fast_pves = [fpn.get_data() for fpn in fast_pves_niftis]
            fast_pves.insert(0, 1.0 - np.sum(np.stack(fast_pves, axis=0), axis=0))  # Background probability
            probs = np.stack(fast_pves, axis=0)

            # Case with no mask, so we copy the original fast segmentation
            nib.Nifti1Image(
                dataobj=np.round(np.transpose(probs, (1, 2, 3, 0)), decimals),
                affine=fast_pves_niftis[0].affine,
                header=fast_pves_niftis[0].header
            ).to_filename(no_lesion_probs_fpath_out)

            # then, for each lesion mask in dataset
            for lesion_id, lesion_mask_fp in case['lesion_mask_fpaths'].items():
                subcase_id = f'{case["id"]}__lesion__{lesion_id}'
                t1_nifti = nib.load(case['t1_brain_fpath'])
                t1_holes = t1_nifti.get_fdata() * (1.0 - nib.load(lesion_mask_fp).get_fdata())

                run_fast(
                    t1_nifti=nib.Nifti1Image(t1_holes, t1_nifti.affine, t1_nifti.header),
                    probs_out_filepath=os.path.join(seg_workspace, subcase_id + '__probs.nii.gz'),
                    tmp_path=os.path.join(seg_workspace, 'segmentation_workspace', subcase_id))

    args = [[n, case] for n, case in enumerate(test_dataset)]

    print('RUNNING JUST ONE CASE BECAUSE THIS IS UNTESTED\n'*10)
    nl.parallel_load(process_case, args[:1], num_workers=12)



def segment_LATS_panther(cfg_run, test_dataset, trained_model, seg_workspace, decimals=5, rerun=False):
    cfg = copy.deepcopy(cfg_default)
    cfg.update(cfg_run)

    from niclib.utils import print_progress_bar, RemainingTimeEstimator

    print('\nSegmenting {} test images...'.format(len(test_dataset)))
    print('  patch shape: {}; norm_type: {}'.format(cfg['patch_shape'], cfg['norm_type']))

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

        t1_brain = t1_brain * (1.0 - lesion_mask)
        lesion_mask = (lesion_mask - 0.5) * 2.0

        probs = segment_tissue_LATS(
            img=np.stack([t1_brain, lesion_mask], axis=0),
            trained_model=trained_model,
            verbose=False,
            in_shape=(2,) + cfg['patch_shape'],
            out_shape=(4,) + cfg['patch_shape'],
            postprocessing=False,
            norm_type=cfg['norm_type'])

        # Prepare for storage and save
        probs = np.transpose(probs, (1, 2, 3, 0))  # Put channels on last position
        probs = np.round(probs, decimals)  # round a bit to avoid gigantic files
        nib.Nifti1Image(probs, t1_brain_nifti.affine, t1_brain_nifti.header).to_filename(probs_fpath_out)


    rta = RemainingTimeEstimator(len(test_dataset))
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
                                             f'{case["id"]}__lesion__{lesion_id}__probs.nii.gz'))

        print_progress_bar(n, len(test_dataset) + 1, suffix=' images segmented - ETA: {} - {}'.format(
            rta.update(n), case['id']))
    print_progress_bar(len(test_dataset) - 1, len(test_dataset),
                       suffix=' images segmented - ETA: {}'.format(rta.elapsed_time()))



if __name__ == '__main__':
    print('Doing timing test...')
    probs_fpaths = [f'/media/user/dades/DATASETS-WS/LATS/campinas-lesion/CC0001_philips_15_55_M/t1_brain_fast_{i}.nii.gz' for i in range(3)]
    probs = [nib.load(pfp).get_fdata() for pfp in probs_fpaths]
    bg_prob = 1.0 - np.add(probs[0], np.add(probs[1], probs[2]))
    probs.insert(0, bg_prob)
    probs = np.stack(probs, axis=0)

    fill_artificial_lesion_and_segment_FAST(
        t1_filepath='/media/user/dades/DATASETS-WS/LATS/campinas-lesion/CC0001_philips_15_55_M/t1_brain.nii.gz',
        lesion_mask_fpath='/media/user/dades/DATASETS-WS/LATS/campinas-lesion/CC0001_philips_15_55_M/lesion_mask_8.nii.gz',
        healthy_tissue_probs=probs,
        path_out='/home/user/Desktop/fast_timing_test',
        probs_out_filepath='/home/user/Desktop/to_erase.nii.gz'
    )
