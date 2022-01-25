import json
import os
import shutil
import subprocess

import numpy as np
import nibabel as nib

import acglib as acg

from prepare_utils import register_to_mni, segment_tissue, merge_binary_mask_niftis

MNI_FILEPATH = '/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz'
MNI_BRAIN_FILEPATH = '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'

def list_dir_paths(p):
    return sorted([f.path for f in os.scandir(p) if f.is_dir()])

def list_filepaths(p):
    return sorted([f.path for f in os.scandir(p) if f.is_file()])

def prepare_challenge2016(dataset_path, output_path):
    """
    MNI transform: T1_preprocessed.nii.gz -> Register to MNI brain
    Lesion mask: Consensus.nii.gz
    """

    for case_path in list_dir_paths(dataset_path):
        case_id = case_path.split(os.sep)[-1]

        case_output_path = os.path.join(output_path, case_id)
        os.makedirs(case_output_path, exist_ok=True)

        t1_filepath = os.path.join(case_path, 'T1_preprocessed.nii.gz')
        lesion_filepath = os.path.join(case_path, 'Consensus.nii.gz')

        t1_filepath_out = os.path.join(case_output_path, 't1_brain.nii.gz')
        t1_mni_filepath_out = None #os.path.join(case_output_path, 't1_brain_mni.nii.gz')
        tx_to_mni_out = os.path.join(case_output_path, 't1_to_mni.mat')
        lesion_filepath_out = os.path.join(case_output_path, 'lesion_mask.nii.gz')

        shutil.copy(t1_filepath, t1_filepath_out)
        shutil.copy(lesion_filepath, lesion_filepath_out)

        register_to_mni(t1_filepath_out, MNI_BRAIN_FILEPATH, tx_to_mni_out, reg_filepath_out=t1_mni_filepath_out)


def prepare_isbi2015(dataset_path, output_path):
    """
    MNI transform: preprocessed/**_mprage_**.nii.gz -> Register to MNI brain
    Lesion mask: masks/*_*_mask1.nii.gz AND masks/*_*_mask2.nii.gz
    """
    print('Preparing ISBI 2015 lesion atlas')

    for subject_path in list_dir_paths(dataset_path):
        subject_id = subject_path.split(os.sep)[-1]
        print(subject_id)

        for tpoint in ['01', '02', '03', '04', '05']:
            mask_num = 'mask1'
            t1_filepath = os.path.join(subject_path, f'preprocessed/{subject_id}_{tpoint}_mprage_pp.nii')
            lesion_filepath = os.path.join(subject_path, f'masks/{subject_id}_{tpoint}_{mask_num}.nii')

            case_id = f'{subject_id}_{tpoint}_{mask_num}'
            if not os.path.isfile(t1_filepath) or not os.path.isfile(lesion_filepath):
                print(f'skipping {case_id}')
                continue

            case_output_path = os.path.join(output_path, case_id)
            os.makedirs(case_output_path, exist_ok=True)

            t1_filepath_out = os.path.join(case_output_path, 't1_brain.nii.gz')
            t1_mni_filepath_out = os.path.join(case_output_path, 't1_brain_mni.nii.gz')
            tx_to_mni_out = os.path.join(case_output_path, 't1_to_mni.mat')
            lesion_filepath_out = os.path.join(case_output_path, 'lesion_mask.nii.gz')

            shutil.copy(t1_filepath, t1_filepath_out)
            shutil.copy(lesion_filepath, lesion_filepath_out)

            register_to_mni(t1_filepath_out, MNI_BRAIN_FILEPATH, tx_to_mni_out, reg_filepath_out=t1_mni_filepath_out)

def prepare_wmh_2017(dataset_path, output_path):
    """
    MNI transform: 3DT1_brain.nii.gz -> Register to MNI brain
    Lesion mask: 3DWMH_c1.nii.gz
    """

    print('Preparing WMH2017 lesion atlas')

    for case_path in list_dir_paths(dataset_path):
        case_id = case_path.split(os.sep)[-1]

        case_output_path = os.path.join(output_path, case_id)
        os.makedirs(case_output_path, exist_ok=True)

        t1_filepath = os.path.join(case_path, '3DT1_brain.nii.gz')
        lesion_filepath = os.path.join(case_path, '3DWMH_c1.nii.gz')

        if not os.path.isfile(t1_filepath) or not os.path.isfile(lesion_filepath):
            print(f'skipping {case_id}')
            continue

        t1_filepath_out = os.path.join(case_output_path, 't1_brain.nii.gz')
        t1_mni_filepath_out = os.path.join(case_output_path, 't1_brain_mni.nii.gz')
        tx_to_mni_out = os.path.join(case_output_path, 't1_to_mni.mat')
        lesion_filepath_out = os.path.join(case_output_path, 'lesion_mask.nii.gz')

        nib.as_closest_canonical(nib.load(t1_filepath)).to_filename(t1_filepath_out)
        nib.as_closest_canonical(nib.load(lesion_filepath)).to_filename(lesion_filepath_out)

        register_to_mni(t1_filepath_out, MNI_BRAIN_FILEPATH, tx_to_mni_out, reg_filepath_out=t1_mni_filepath_out)


def prepare_campinas(dataset_path, output_path):
    """
    MNI transform:

    Tissue segmentation:
    """
    print('Preparing campinas-tissue for LATS')

    def prepare_campinas_case(t1_filepath):
        case_id = t1_filepath.split(os.sep)[-1].split('.')[0]
        print(case_id)

        case_output_path = os.path.join(output_path, case_id)
        os.makedirs(case_output_path, exist_ok=True)

        t1_filepath_out = os.path.join(case_output_path, 't1.nii.gz')
        t1_brain_filepath_out = os.path.join(case_output_path, 't1_brain.nii.gz')
        t1_mni_filepath_out = os.path.join(case_output_path, 't1_brain_mni.nii.gz')
        tx_to_mni_out = os.path.join(case_output_path, 't1_to_mni.mat')
        fast_filepaths = [
            os.path.join(case_output_path, 't1_brain_fast_0.nii.gz'),
            os.path.join(case_output_path, 't1_brain_fast_1.nii.gz'),
            os.path.join(case_output_path, 't1_brain_fast_2.nii.gz')]

        # Copy t1 image
        if not os.path.isfile(t1_filepath_out):
            shutil.copy(t1_filepath, t1_filepath_out)

        if not os.path.isfile(t1_brain_filepath_out):
            t1_brain_mask_filepath = os.path.join(os.path.dirname(t1_filepath), 'SS_STAPLE', f'{case_id}_staple.nii.gz')
            t1_nifti = nib.load(t1_filepath)
            t1 = t1_nifti.get_fdata()
            t1_brain_mask = nib.load(t1_brain_mask_filepath).get_fdata()
            # Save t1_brain
            t1_brain_nifti = nib.Nifti1Image(t1 * t1_brain_mask, t1_nifti.affine, t1_nifti.header)
            t1_brain_nifti.to_filename(t1_brain_filepath_out)

        # Registration to MNI
        if not os.path.isfile(tx_to_mni_out):
            register_to_mni(t1_brain_filepath_out, MNI_BRAIN_FILEPATH, tx_to_mni_out, reg_filepath_out=t1_mni_filepath_out)

        # FAST tissue segmentation
        if not all([os.path.isfile(ffp) for ffp in fast_filepaths]):
            segment_tissue(t1_brain_filepath_out)

    acg.parallel_run(prepare_campinas_case, list_filepaths(dataset_path), num_threads=10)


def generate_lats_dataset(healthy_dataset_paths, lesion_atlas_paths, output_path):
    all_healthy_case_paths = []
    for healthy_dataset_path in healthy_dataset_paths:
        all_healthy_case_paths += list_dir_paths(healthy_dataset_path)

    all_lesion_case_paths = []
    for lesion_atlas_path in lesion_atlas_paths:
        all_lesion_case_paths += list_dir_paths(lesion_atlas_path)

    print('Generating lats dataset')
    for healthy_case_path in all_healthy_case_paths:
        print(healthy_case_path)
        healthy_t1_fp = os.path.join(healthy_case_path, 't1.nii.gz')
        healthy_t1_brain_fp = os.path.join(healthy_case_path, 't1_brain.nii.gz')
        healthy_csf_probs_fp = os.path.join(healthy_case_path, 't1_brain_fast_0.nii.gz')
        healthy_gm_probs_fp = os.path.join(healthy_case_path, 't1_brain_fast_1.nii.gz')
        healthy_wm_probs_fp = os.path.join(healthy_case_path, 't1_brain_fast_2.nii.gz')
        healthy_tx_to_mni_fp = os.path.join(healthy_case_path, 't1_to_mni.mat')
        healthy_tx_from_mni_fp = os.path.join(healthy_case_path, 'mni_to_t1.mat')

        # Invert healthy_tx_to_mni_fp
        if not os.path.isfile(healthy_tx_from_mni_fp):
            subprocess.check_output(
                ['bash', '-c', 'convert_xfm -omat {} -inverse {}'.format(healthy_tx_from_mni_fp, healthy_tx_to_mni_fp)])

        # Copy healthy images to output folder
        healthy_case_id = healthy_case_path.split(os.sep)[-1]
        case_output_path = os.path.join(output_path, healthy_case_id)
        os.makedirs(case_output_path, exist_ok=True)

        healthy_t1_out_fp = os.path.join(case_output_path, 't1.nii.gz')
        healthy_t1_brain_mask_out_fp = os.path.join(case_output_path, 't1_brain_mask.nii.gz')
        healthy_t1_brain_out_fp = os.path.join(case_output_path, 't1_brain.nii.gz')
        healthy_csf_probs_out_fp = os.path.join(case_output_path, 't1_brain_fast_0.nii.gz')
        healthy_gm_probs_out_fp = os.path.join(case_output_path, 't1_brain_fast_1.nii.gz')
        healthy_wm_probs_out_fp = os.path.join(case_output_path, 't1_brain_fast_2.nii.gz')

        if not os.path.isfile(healthy_t1_out_fp):
            shutil.copy(healthy_t1_fp, healthy_t1_out_fp)
        if not os.path.isfile(healthy_t1_brain_out_fp):
            shutil.copy(healthy_t1_brain_fp, healthy_t1_brain_out_fp)
        if not os.path.isfile(healthy_t1_brain_mask_out_fp):
            brain_mask_nifti = nib.load(healthy_t1_brain_fp)
            nib.Nifti1Image(
                (brain_mask_nifti.get_fdata() > 0.0).astype(int),
                brain_mask_nifti.affine,
                brain_mask_nifti.header
            ).to_filename(healthy_t1_brain_mask_out_fp)
        if not os.path.isfile(healthy_csf_probs_out_fp):
            shutil.copy(healthy_csf_probs_fp, healthy_csf_probs_out_fp)
        if not os.path.isfile(healthy_gm_probs_out_fp):
            shutil.copy(healthy_gm_probs_fp, healthy_gm_probs_out_fp)
        if not os.path.isfile(healthy_wm_probs_out_fp):
            shutil.copy(healthy_wm_probs_fp, healthy_wm_probs_out_fp)

        # Load wm mask from healthy patient
        healthy_wm_mask = (nib.load(healthy_wm_probs_fp).get_fdata() > 0.5).astype(np.float)

        # Register lesion masks to healthy patient
        healthy_lesion_mask_dict = {}
        for lesion_case_path in all_lesion_case_paths:
            lesion_case_id = lesion_case_path.split(os.sep)[-1]
            lesion_mask_fp = os.path.join(lesion_case_path, 'lesion_mask.nii.gz')
            lesion_tx_to_mni_fp = os.path.join(lesion_case_path, 't1_to_mni.mat')

            # Concatenate transforms (lesion -> MNI & MNI -> healthy)
            tx_lesion_to_healthy_fp = os.path.join(case_output_path, f'{lesion_case_id}_to_{healthy_case_id}.mat')
            if not os.path.isfile(tx_lesion_to_healthy_fp):
                subprocess.check_output(['bash', '-c',
                    f'convert_xfm -omat {tx_lesion_to_healthy_fp} -concat {healthy_tx_from_mni_fp} {lesion_tx_to_mni_fp}'])

            # Register lesion to healthy t1_brain using both transforms
            healthy_lesion_mask_fp = os.path.join(case_output_path, f'lesion_mask_{lesion_case_id}.nii.gz')
            healthy_lesion_mask_dict.update({lesion_case_id: healthy_lesion_mask_fp})

            if not os.path.isfile(healthy_lesion_mask_fp):
                convert_cmd = \
                    'flirt -in {} -applyxfm -init {} -out {} -interp nearestneighbour -setbackground 0 -ref {}'
                subprocess.check_output(['bash', '-c', convert_cmd.format(
                    lesion_mask_fp, tx_lesion_to_healthy_fp, healthy_lesion_mask_fp, healthy_t1_brain_fp)])

                # Keep only lesion voxels inside WM mask
                healthy_lesion_mask_nifti = nib.load(healthy_lesion_mask_fp)
                healthy_lesion_mask = healthy_lesion_mask_nifti.get_fdata()

                healthy_lesion_wm_mask = (healthy_lesion_mask * healthy_wm_mask).astype(np.uint8)

                # Store as compressed int
                nib.Nifti1Image(
                    healthy_lesion_wm_mask.astype(np.uint8), healthy_lesion_mask_nifti.affine, healthy_lesion_mask_nifti.header
                ).to_filename(healthy_lesion_mask_fp)


        # Generate average lesion mask
        avg_lesion_mask_fp = os.path.join(case_output_path, 'average_lesion_mask.nii.gz')

        if not os.path.isfile(avg_lesion_mask_fp):
            lesion_masks = [nib.load(lesion_mask_fp) for lesion_mask_fp in healthy_lesion_mask_dict.values()]
            avg_lesion_mask = np.mean(np.stack([lm.get_fdata() for lm in lesion_masks], axis=0), axis=0)

            nib.Nifti1Image(
                avg_lesion_mask, lesion_masks[0].affine, lesion_masks[0].header
            ).to_filename(avg_lesion_mask_fp)




if __name__ == '__main__':
    # prepare_challenge2016(
    #     dataset_path='/media/user/dades/DATASETS/Challenge2016',
    #     output_path='/media/user/dades/DATASETS-WS/LATS/Challenge2016-lesion-atlas')

    # prepare_isbi2015(
    #     dataset_path='/media/user/dades/DATASETS/ISBI/Training',
    #     output_path='/media/user/dades/DATASETS-WS/LATS/ISBI-lesion-atlas')
    #
    # prepare_wmh_2017(
    #     dataset_path='/media/user/dades/DATASETS/WMH20173D',
    #     output_path = '/media/user/dades/DATASETS-WS/LATS/WMH20173D-lesion-atlas')

    # prepare_campinas(
    #     dataset_path='/media/user/dades/DATASETS/campinas',
    #     output_path='/media/user/dades/DATASETS-WS/LATS/campinas-tissue'
    # )

    generate_lats_dataset(
        healthy_dataset_paths=['/media/user/dades/DATASETS-WS/LATS/campinas-tissue'],
        lesion_atlas_paths=[
            '/media/user/dades/DATASETS-WS/LATS/Challenge2016-lesion-atlas',
            '/media/user/dades/DATASETS-WS/LATS/WMH20173D-lesion-atlas',
            '/media/user/dades/DATASETS-WS/LATS/ISBI-lesion-atlas'
        ],
        output_path='/media/user/dades/DATASETS-WS/LATS/campinas-lesion'
    )
