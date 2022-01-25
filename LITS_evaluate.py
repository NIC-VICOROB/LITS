import json

import numpy as np
import nibabel as nib
import os

import niclib as nl
import acglib as acg


def dsc(output : np.ndarray, target: np.ndarray, background_label=0):
    """Dice Similarity Coefficient. Output and target must contain integer labels."""
    assert output.shape == target.shape, f'{output.shape} != {target.shape}'
    output_mask = (output != background_label) if output.dtype != np.bool else output
    target_mask = (target != background_label) if target.dtype != np.bool else target

    intersection = np.sum(np.logical_and(output == target, np.logical_or(output_mask, target_mask)))
    denominator = np.sum(output_mask) + np.sum(target_mask)
    return 2.0 * intersection / denominator if denominator > 0.0 else 0.0


def list_filenames(p):
    return sorted([f.name for f in os.scandir(p) if f.is_file()])


def load_seg(seg_filepath):
    return np.transpose(nib.load(seg_filepath).get_fdata(), axes=(3, 0, 1, 2))  # Put channels first


def compute_lesion_neighbourhood_mask(lesion_mask, patch_size, exclude_lesion=True):
    from scipy import ndimage
    dist_image = ndimage.distance_transform_cdt(1.0 - lesion_mask, metric='chessboard')
    dist_mask = dist_image < patch_size
    if exclude_lesion:
        dist_mask =  np.logical_xor(dist_mask, lesion_mask > 0.0)
    return dist_mask.astype(np.int)


def compute_segmentation_difference_metrics(healthy_seg, lesioned_seg, roi=None):
    if roi is None:
        roi = np.ones_like(healthy_seg[0])

    roi = roi > 0

    healthy_brain = np.sum(healthy_seg[1:], where=np.expand_dims(roi, axis=0))
    healthy_csf = np.sum(healthy_seg[1], where=roi)
    healthy_gm = np.sum(healthy_seg[2], where=roi)
    healthy_wm = np.sum(healthy_seg[3], where=roi)
    healthy_tissue = healthy_gm + healthy_wm

    lesioned_brain = np.sum(lesioned_seg[1:], where=np.expand_dims(roi, axis=0))
    lesioned_csf = np.sum(lesioned_seg[1] , where=roi)
    lesioned_gm = np.sum(lesioned_seg[2] , where=roi)
    lesioned_wm = np.sum(lesioned_seg[3] , where=roi)
    lesioned_tissue = lesioned_gm + lesioned_wm

    brain_diff = 100.0 * (healthy_brain - lesioned_brain) / healthy_brain
    csf_diff = 100.0 * (healthy_csf - lesioned_csf) / healthy_csf
    gm_diff = 100.0 * (healthy_gm - lesioned_gm) / healthy_gm
    wm_diff = 100.0 * (healthy_wm - lesioned_wm) / healthy_wm
    tissue_diff = 100.0 * (healthy_tissue - lesioned_tissue) / healthy_tissue

    healthy_seg_labels = np.argmax(healthy_seg, axis=0)
    lesioned_seg_labels = np.argmax(lesioned_seg, axis=0)

    brain_dsc = dsc(healthy_seg_labels, lesioned_seg_labels)
    csf_dsc = dsc(healthy_seg_labels == 1, lesioned_seg_labels == 1)
    gm_dsc = dsc(healthy_seg_labels == 2, lesioned_seg_labels == 2)
    wm_dsc = dsc(healthy_seg_labels == 3, lesioned_seg_labels == 3)
    tissue_dsc = dsc(healthy_seg_labels >= 2, lesioned_seg_labels >= 2)

    return {'brain_diff': brain_diff,
            'tissue_diff': tissue_diff,
            'csf_diff': csf_diff,
            'gm_diff': gm_diff,
            'wm_diff': wm_diff,
            'brain_dsc': brain_dsc,
            'tissue_dsc': tissue_dsc,
            'csf_dsc': csf_dsc,
            'gm_dsc': gm_dsc,
            'wm_dsc': wm_dsc}



def compute_multiROI_segmentation_difference_metrics(healthy_seg, lesioned_seg, lesion_mask, brain_mask, patch_size=16):
    ### Generate multi ROIs
    brain_roi = (brain_mask > 0.5).astype(float)
    lesion_roi = (lesion_mask > 0.5).astype(float)
    normal_appearing_roi = np.logical_and(brain_mask, lesion_mask < 0.5).astype(float)
    inside_neighbourhood_roi = np.logical_and(
        brain_mask, compute_lesion_neighbourhood_mask(lesion_mask, patch_size, exclude_lesion=True) > 0.5).astype(float)
    outside_neighbourhood_roi = np.logical_and(
        brain_mask, compute_lesion_neighbourhood_mask(lesion_mask, patch_size, exclude_lesion=False) < 0.5).astype(float)

    ### Compute difference metrics
    all_diffs = {}

    # Evaluate segmentation differences in all volume
    whole_seg_diffs = compute_segmentation_difference_metrics(healthy_seg, lesioned_seg, roi=brain_roi)
    all_diffs.update({'whole_' + k: v for k, v in whole_seg_diffs.items()})

    # Evaluate segmentation differences inside/outside lesion_mask
    lesion_seg_diffs = compute_segmentation_difference_metrics(healthy_seg, lesioned_seg, roi=lesion_roi)
    all_diffs.update({'lesion_' + k: v for k, v in lesion_seg_diffs.items()})

    # Evaluate segmentation differences in normal appearing
    normal_appearing_seg_diffs = \
        compute_segmentation_difference_metrics(healthy_seg, lesioned_seg, roi=normal_appearing_roi)
    all_diffs.update({'normal_' + k: v for k, v in normal_appearing_seg_diffs.items()})

    # Evaluate segmentation differences on a radius inside/outside lesion_mask
    inside_fov_seg_diffs = \
        compute_segmentation_difference_metrics(healthy_seg, lesioned_seg, roi=inside_neighbourhood_roi)
    all_diffs.update({'inside_neigh_' + k: v for k, v in inside_fov_seg_diffs.items()})

    outside_fov_seg_diffs = \
        compute_segmentation_difference_metrics(healthy_seg, lesioned_seg, roi=outside_neighbourhood_roi)
    all_diffs.update({'outside_neigh_' + k: v for k, v in outside_fov_seg_diffs.items()})

    return all_diffs



def compute_lesion_composition_metrics(lesioned_seg, lesion_mask, nifti):
    vox2mm = np.prod(np.abs(nifti.header['pixdim'][1:4]))
    lesion_mask = (lesion_mask > 0.5)

    lesion_total_vol = np.sum(lesion_mask) * vox2mm
    lesion_csf = np.sum(lesioned_seg[1], where=lesion_mask) * vox2mm
    lesion_gm = np.sum(lesioned_seg[2], where=lesion_mask) * vox2mm
    lesion_wm = np.sum(lesioned_seg[3], where=lesion_mask) * vox2mm

    return {'lesioned_lesion_total_vol': lesion_total_vol,
            'lesioned_lesion_csf_vol': lesion_csf,
            'lesioned_lesion_gm_vol': lesion_gm,
            'lesioned_lesion_wm_vol': lesion_wm}



def compute_LITS_lesioned_metrics(healthy_seg_fp_dict, lesioned_seg_fp_dict, healthy_lesion_dataset_path, csv_out_filepath):
    print(csv_out_filepath)

    # For each non-lesion seg id
    all_metrics = []
    for n, (healthy_id, healthy_seg_filepath) in enumerate(healthy_seg_fp_dict.items()):
        acg.print_utils.print_progress_bar(n, len(healthy_seg_fp_dict))

        # Load healthy_seg and brain_mask
        healthy_seg = load_seg(healthy_seg_filepath)

        # Get brain mask from original dataset
        brain_mask_filepath = os.path.join(healthy_lesion_dataset_path, healthy_id, f't1_brain.nii.gz')
        brain_mask = (nib.load(brain_mask_filepath).get_fdata() > 0.0).astype(float)

        # Get list of lesion_id and lesioned segmentations
        healthy_case_lesioned_segs = {k.split('__')[-1] : v for k, v in lesioned_seg_fp_dict.items()
                                      if k.split('__')[0] == healthy_id and os.path.isfile(v)}

        def process_case(lesion_id, lesion_seg_filepath):
            # Load lesioned segmentation
            lesioned_seg = load_seg(lesion_seg_filepath)
            # Load lesion mask from original dataset
            lesion_mask_filepath = os.path.join(
                healthy_lesion_dataset_path, healthy_id, f'lesion_mask_{lesion_id}.nii.gz')
            lesion_mask_nifti = nib.load(lesion_mask_filepath)
            lesion_mask = lesion_mask_nifti.get_fdata()

            case_metrics = {
                    'id': f'{healthy_id}__{lesion_id}',
                    'healthy_id': healthy_id,
                    'lesion_id': lesion_id}
            # Compute difference metrics in several ROIs
            case_metrics.update(compute_multiROI_segmentation_difference_metrics(
                healthy_seg=healthy_seg, lesioned_seg=lesioned_seg, lesion_mask=lesion_mask, brain_mask=brain_mask))
            # Compute lesioned segmentation lesion composition metrics
            case_metrics.update(compute_lesion_composition_metrics(
                lesioned_seg=lesioned_seg, lesion_mask=lesion_mask, nifti=lesion_mask_nifti))
            return case_metrics

        healthy_case_metrics = acg.parallel_run(
            process_case, [[lid, lsfp] for lid, lsfp in healthy_case_lesioned_segs.items()], num_threads=12)

        all_metrics += healthy_case_metrics

    nl.save_to_csv(csv_out_filepath, all_metrics)

def compute_LITS_healthy_metrics(healthy_seg_fp_dict, healthy_seg_fp_dict2, healthy_lesion_dataset_path, csv_out_filepath):
    print(csv_out_filepath)

    # For each non-lesion seg id
    all_metrics = []
    for n, (healthy_id, healthy_seg_filepath) in enumerate(healthy_seg_fp_dict.items()):
        acg.print_utils.print_progress_bar(n, len(healthy_seg_fp_dict))

        # Load healthy_seg and brain_mask
        healthy_seg = load_seg(healthy_seg_filepath)

        # Get brain mask from original dataset
        brain_mask_filepath = os.path.join(healthy_lesion_dataset_path, healthy_id, f't1_brain.nii.gz')
        brain_mask = (nib.load(brain_mask_filepath).get_fdata() > 0.0).astype(float)

        # Get list of lesion_id and lesioned segmentations
        healthy_case_lesioned_segs = {k : v for k, v in healthy_seg_fp_dict2.items() if k == healthy_id}

        def process_case(lesion_id, lesion_seg_filepath):
            # Load lesioned segmentation
            healthy_seg2 = load_seg(lesion_seg_filepath)
            # Load lesion mask from original dataset
            lesion_mask = np.zeros_like(healthy_seg2[0])

            case_metrics = {
                    'id': f'{healthy_id}',
                    'healthy_id': healthy_id,
                    'lesion_id': lesion_id}
            # Compute difference metrics in several ROIs
            case_metrics.update(compute_multiROI_segmentation_difference_metrics(
                healthy_seg=healthy_seg, lesioned_seg=healthy_seg2, lesion_mask=lesion_mask, brain_mask=brain_mask))
            # Compute lesioned segmentation lesion composition metrics
            case_metrics.update(compute_lesion_composition_metrics(
                lesioned_seg=healthy_seg2, lesion_mask=lesion_mask, nifti=nib.load(lesion_seg_filepath)))
            return case_metrics

        healthy_case_metrics = acg.parallel_run(
            process_case, [[lid, lsfp] for lid, lsfp in healthy_case_lesioned_segs.items()], num_threads=12)

        all_metrics += healthy_case_metrics

    nl.save_to_csv(csv_out_filepath, all_metrics)
