import subprocess
import nibabel as nib
import os
import numpy as np

import acglib as acg

def register_to_mni(t1_filepath, reference_filepath, transform_filepath_out, reg_filepath_out=None):
    """Registers the image to MNI space and stores the transform to MNI"""
    register_cmd = 'flirt -in {} -ref {} -omat {} '.format(t1_filepath, reference_filepath, transform_filepath_out)
    if reg_filepath_out is not None:
        register_cmd += '-out {} '.format(reg_filepath_out)
    register_opts = '-bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear'
    subprocess.check_output(['bash', '-c', register_cmd + register_opts])


def segment_tissue(filepath_in):
    """Performs 3 tissue segmentation"""

    print('Running FAST: {}'.format(filepath_in))
    subprocess.check_call(['bash', '-c', 'fast {}'.format(filepath_in)])

    pve_fpaths = [acg.path.remove_ext(filepath_in) + '_pve_{}.nii.gz'.format(i) for i in range(3)]
    out_fpaths = [acg.path.remove_ext(filepath_in) + '_fast_{}.nii.gz'.format(i) for i in range(3)]

    for pve_fpath, out_fpath in zip(pve_fpaths, out_fpaths):
        os.rename(pve_fpath, out_fpath)

    # Remove all other files
    os.remove(os.path.join(acg.path.remove_ext(filepath_in) + '_mixeltype.nii.gz'))
    os.remove(os.path.join(acg.path.remove_ext(filepath_in) + '_pveseg.nii.gz'))
    os.remove(os.path.join(acg.path.remove_ext(filepath_in) + '_seg.nii.gz'))


def merge_binary_mask_niftis(input_filepaths, output_filepath):
    nifti_reference = None
    arrays = []
    for n, input_filepath in enumerate(input_filepaths):
        input_nifti = nib.load(input_filepath)
        arrays.append(input_nifti.get_fdata())

        if n == 0:
            nifti_reference = input_nifti

    nib.Nifti1Image(
        np.stack(arrays, axis=-1).astype(np.uint8),
        nifti_reference.affine,
        nifti_reference.header
    ).to_filename(output_filepath)






