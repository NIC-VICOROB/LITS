import subprocess
import os
import shutil
import nibabel as nib
import numpy as np

from .path import remove_ext, make_dirs, get_path, get_filename


def run_fsl_anat(fpath_in, do_fast, do_first, merge_fast_first, remove_anat_dir=True):
    assert os.path.isfile(fpath_in), fpath_in

    anat_dir = remove_ext(fpath_in) + '.anat'
    fpaths_anat_fast = [os.path.join(anat_dir, 'T1_fast_pve_{}.nii.gz'.format(i)) for i in range(3)]
    fpath_anat_first = os.path.join(anat_dir, 'T1_subcort_seg.nii.gz')

    fpaths_tgt_fast = [remove_ext(fpath_in) + '_fast_{}.nii.gz'.format(i) for i in range(4)]
    fpaths_tgt_fast_first = [remove_ext(fpath_in) + '_fast_first_{}.nii.gz'.format(i) for i in range(4)]

    if not os.path.isfile(fpaths_tgt_fast_first[-1]):
        if not all([os.path.isfile(faf) for faf in fpaths_anat_fast]) or not os.path.isfile(fpath_anat_first):
            #anat_cmd =  f'fsl_anat --clobber --weakbias --nocrop --noreorient -i {fpath_in}'
            anat_cmd =  f'fsl_anat --weakbias --nocrop --noreorient -i {fpath_in}'
            print(anat_cmd)
            subprocess.check_output(['bash', '-c', anat_cmd])

        # Load FAST segmentations, add background prob channel and store in the target folder
        fast_nifti = nib.load(fpaths_anat_fast[0])
        fast_pves = [nib.load(fp).get_data() for fp in fpaths_anat_fast]
        fast_pves.insert(0, 1.0 - np.sum(np.stack(fast_pves, axis=0), axis=0))  # Background probability
        for fast_pve, fpath_tgt_fast in zip(fast_pves, fpaths_tgt_fast):
            fast_pve_arr = np.round(fast_pve, decimals=5)
            nib.Nifti1Image(fast_pve_arr, fast_nifti.affine, fast_nifti.header).to_filename(fpath_tgt_fast)

        # Load first segmentation, overwrite the subcortical structures and store in target folder
        first_seg = nib.load(fpath_anat_first).get_data()
        fast_pves[1][first_seg > 0] = 0.0
        fast_pves[2][first_seg > 0] = 1.0
        fast_pves[3][first_seg > 0] = 0.0
        for fast_pve, fpath_tgt_fast_first in zip(fast_pves, fpaths_tgt_fast_first):
            fast_pve_arr = np.round(fast_pve, decimals=5)
            nib.Nifti1Image(fast_pve_arr, fast_nifti.affine, fast_nifti.header).to_filename(fpath_tgt_fast_first)

        if remove_anat_dir:
            shutil.rmtree(anat_dir)


def run_fast(filepath_in):
    print('Running FAST: {}'.format(filepath_in))
    subprocess.check_call(['bash', '-c', 'fast {}'.format(filepath_in)])

    pve_fpaths = [remove_ext(filepath_in) + '_pve_{}.nii.gz'.format(i) for i in range(3)]
    out_fpaths = [remove_ext(filepath_in) + '_fast_{}.nii.gz'.format(i) for i in range(3)]

    for pve_fpath, out_fpath in zip(pve_fpaths, out_fpaths):
        os.rename(pve_fpath, out_fpath)

    # Remove all other files
    os.remove(os.path.join(remove_ext(filepath_in) + '_mixeltype.nii.gz'))
    os.remove(os.path.join(remove_ext(filepath_in) + '_pveseg.nii.gz'))
    os.remove(os.path.join(remove_ext(filepath_in) + '_seg.nii.gz'))



def run_first(filepath_in, is_skull_stripped=True):
    print('Running FIRST: {}'.format(filepath_in))
    first_fpath_out = remove_ext(filepath_in) + '_first.nii.gz'

    # Create temporary path
    tmp_path = make_dirs(os.path.join(get_path(filepath_in), 'first_tmp'))
    t1_fpath = os.path.join(tmp_path, get_filename(filepath_in, ext=True))
    first_fpath_in = os.path.join(tmp_path, 't1_all_fast_firstseg.nii.gz')

    shutil.copy(filepath_in, t1_fpath)
    if is_skull_stripped:
        first_cmd_template = 'run_first_all -b -i {} -o t1'
    else:
        first_cmd_template = 'run_first_all -i {} -o t1'

    subprocess.check_call(['bash', '-c', first_cmd_template.format(t1_fpath)], cwd=tmp_path)

    first_nifti = nib.load(first_fpath_in)
    nib.Nifti1Image(
        (first_nifti.get_fdata() > 0.5).astype(float), first_nifti.affine, first_nifti.header
    ).to_filename(first_fpath_out)

    # Erase all temporary files
    shutil.rmtree(tmp_path)
    
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

    pve_fpaths = [remove_ext(filepath_in) + '_pve_{}.nii.gz'.format(i) for i in range(3)]
    out_fpaths = [remove_ext(filepath_in) + '_fast_{}.nii.gz'.format(i) for i in range(3)]

    for pve_fpath, out_fpath in zip(pve_fpaths, out_fpaths):
        os.rename(pve_fpath, out_fpath)

    # Remove all other files
    os.remove(os.path.join(remove_ext(filepath_in) + '_mixeltype.nii.gz'))
    os.remove(os.path.join(remove_ext(filepath_in) + '_pveseg.nii.gz'))
    os.remove(os.path.join(remove_ext(filepath_in) + '_seg.nii.gz'))