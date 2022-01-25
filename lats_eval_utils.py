import json

import numpy as np
import nibabel as nib
import os

import niclib as nl

def list_filenames(p):
    return sorted([f.name for f in os.scandir(p) if f.is_file()])

def load_seg(seg_filepath):
    return np.transpose(nib.load(seg_filepath).get_fdata(), axes=(3, 0, 1, 2))  # Put channels first


def compute_lesion_patch_fov_mask(lesion_mask, patch_size, exclude_lesion=True):
    from scipy import ndimage
    fov_dist = np.sqrt(np.power(patch_size / 2.0, 2) * 2)
    dist_image = ndimage.distance_transform_edt(1.0 - lesion_mask)
    dist_mask = dist_image < fov_dist
    if exclude_lesion:
        dist_mask =  np.logical_xor(dist_mask, lesion_mask > 0.0)
    return dist_mask.astype(np.int)


def compute_seg_diff_metrics(healthy_seg, lesion_seg, roi=None):
    if roi is None:
        roi = np.ones_like(healthy_seg[0])

    healthy_brain = np.sum((1.0 - healthy_seg[0]) * roi)
    healthy_csf = np.sum(healthy_seg[1] * roi)
    healthy_gm = np.sum(healthy_seg[2] * roi)
    healthy_wm = np.sum(healthy_seg[3] * roi)

    lesion_brain = np.sum((1.0 - lesion_seg[0]) * roi)
    lesion_csf = np.sum(lesion_seg[1] * roi)
    lesion_gm = np.sum(lesion_seg[2] * roi)
    lesion_wm = np.sum(lesion_seg[3] * roi)

    brain_diff = 100.0 * np.abs(healthy_brain - lesion_brain) / healthy_brain
    csf_diff = 100.0 * np.abs(healthy_csf - lesion_csf) / healthy_csf
    gm_diff = 100.0 * np.abs(healthy_gm - lesion_gm) / healthy_gm
    wm_diff = 100.0 * np.abs(healthy_wm - lesion_wm) / healthy_wm

    return {'brain_diff': brain_diff,
            'csf_diff': csf_diff,
            'gm_diff': gm_diff,
            'wm_diff': wm_diff}


def compute_seg_metrics(output_seg, target_seg):
    def dsc(output : np.ndarray, target: np.ndarray, background_label=0):
        """Dice Similarity Coefficient. Output and target must contain integer labels."""
        assert output.shape == target.shape, f'{output.shape} != {target.shape}'
        output_mask = (output != background_label) if output.dtype != np.bool else output
        target_mask = (target != background_label) if target.dtype != np.bool else target

        intersection = np.sum(np.logical_and(output == target, np.logical_or(output_mask, target_mask)))
        denominator = np.sum(output_mask) + np.sum(target_mask)
        return 2.0 * intersection / denominator if denominator > 0.0 else 0.0

    return {'lesioned_dsc': dsc(output=np.argmax(output_seg, axis=0),
                                target=np.argmax(target_seg, axis=0))}

def get_difference_metrics(healthy_seg, lesion_seg, lesion_mask):
    all_diffs = {}

    # Evaluate segmentation differences in all volume
    whole_seg_diffs = compute_seg_diff_metrics(healthy_seg, lesion_seg, roi=np.ones_like(lesion_mask))
    all_diffs.update({'whole_' + k: v for k, v in whole_seg_diffs.items()})

    # Evaluate segmentation differences inside/outside lesion_mask
    lesion_seg_diffs = compute_seg_diff_metrics(healthy_seg, lesion_seg, roi=lesion_mask)
    all_diffs.update({'lesion_' + k: v for k, v in lesion_seg_diffs.items()})

    normal_appearing_seg_diffs = compute_seg_diff_metrics(healthy_seg, lesion_seg, roi=1.0 - lesion_mask)
    all_diffs.update({'normal_' + k: v for k, v in normal_appearing_seg_diffs.items()})

    # Evaluate segmentation differences on a radius inside/outside lesion_mask
    inside_fov_seg_diffs = compute_seg_diff_metrics(
        healthy_seg, lesion_seg, roi=compute_lesion_patch_fov_mask(lesion_mask, 32, exclude_lesion=True))
    all_diffs.update({'inside_fov_' + k: v for k, v in inside_fov_seg_diffs.items()})

    outside_fov_seg_diffs = compute_seg_diff_metrics(
        healthy_seg, lesion_seg, roi=1.0 - compute_lesion_patch_fov_mask(lesion_mask, 32, exclude_lesion=False))
    all_diffs.update({'outside_fov_' + k: v for k, v in outside_fov_seg_diffs.items()})

    return all_diffs



def evaluate_lats_differences(seg_workspace, healthy_lesion_dataset_path, csv_out_filepath):
    # Get list with all non-lesion segmentation ids
    all_prob_filenames = list_filenames(seg_workspace)

    healthy_prob_filenames = [pf for pf in all_prob_filenames if len(pf.split('__')) == 2]
    healthy_ids = [hf.split('__')[0] for hf in healthy_prob_filenames]

    # For each non-lesion seg id
    all_metrics = []
    for healthy_id, healthy_prob_filename in zip(healthy_ids, healthy_prob_filenames):
        print(healthy_prob_filename)
        healthy_seg_filepath = os.path.join(seg_workspace, healthy_prob_filename)

        try:
            healthy_seg = load_seg(healthy_seg_filepath)
        except Exception:
            print(healthy_seg_filepath)

        # Get list of matching lesion_mask segmentations
        lesion_prob_filenames = [pf for pf in all_prob_filenames
                                 if len(pf.split('__')) == 4 and pf.split('__')[0] == healthy_id]

        lesion_ids = [lpf.split('__')[2] for lpf in lesion_prob_filenames]

        def process_case(lesion_id, lesion_filename):
            # Load segmentation with holes from lesion mask
            lesion_seg_filepath = os.path.join(seg_workspace, lesion_filename)
            lesion_seg = load_seg(lesion_seg_filepath)

            # Get lesion mask from original dataset
            lesion_mask_filepath = os.path.join(
                healthy_lesion_dataset_path, healthy_id, f'lesion_mask_{lesion_id}.nii.gz')
            lesion_mask = nib.load(lesion_mask_filepath).get_fdata()

            case_metrics = get_difference_metrics(healthy_seg, lesion_seg, lesion_mask)
            case_metrics.update({
                    'id': f'{healthy_id}__{lesion_id}',
                    'healthy_id': healthy_id,
                    'lesion_id': lesion_id})

            # Get original segmentation from dataset
            fast_fps = [os.path.join(healthy_lesion_dataset_path, healthy_id, f't1_brain_fast_{i}.nii.gz')
                        for i in range(3)]
            fast_pves = [nib.load(fp).get_data() for fp in fast_fps]
            fast_pves.insert(0, 1.0 - np.sum(np.stack(fast_pves, axis=0), axis=0))  # Background probability
            fast_seg = np.stack(fast_pves, axis=0)

            case_metrics.update(compute_seg_metrics(lesion_seg, fast_seg))

            return case_metrics

        healthy_case_metrics = nl.parallel_load(
            process_case, [[lid, lfn] for lid, lfn in zip(lesion_ids, lesion_prob_filenames)], num_workers=12)

        all_metrics += healthy_case_metrics

    nl.save_to_csv(csv_out_filepath, all_metrics)