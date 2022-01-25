import numpy as np
import niclib as nl

def compute_longitudinal_measures(
        baseline_probs, followup_probs, baseline_nifti=None, followup_nifti=None,
        baseline_gt=None, followup_gt=None, baseline_scale=None, followup_scale=None):

    baseline_probs = baseline_probs.astype('float64')
    followup_probs = followup_probs.astype('float64')

    # First transformation from voxels to mm3
    if baseline_nifti is not None and followup_nifti is not None:
        baseline_VOX2MM = np.prod(np.abs(baseline_nifti.header['pixdim'][1:4]))
        followup_VOX2MM = np.prod(np.abs(followup_nifti.header['pixdim'][1:4]))
    else:
        baseline_VOX2MM, followup_VOX2MM = 1.0, 1.0

    # Tissue metrics
    baseline_volume, followup_volume = np.sum(baseline_probs[1:]) * baseline_VOX2MM, np.sum(followup_probs[1:]) * followup_VOX2MM

    baseline_csf, followup_csf = np.sum(baseline_probs[1]) * baseline_VOX2MM, np.sum(followup_probs[1]) * followup_VOX2MM
    baseline_gray, followup_gray = np.sum(baseline_probs[2]) * baseline_VOX2MM, np.sum(followup_probs[2]) * followup_VOX2MM
    baseline_white, followup_white = np.sum(baseline_probs[3]) * baseline_VOX2MM, np.sum(followup_probs[3]) * followup_VOX2MM
    baseline_tissue, followup_tissue = baseline_gray + baseline_white, followup_gray + followup_white

    change_tissue = 200.0 * (followup_tissue - baseline_tissue) / (followup_tissue + baseline_tissue)
    change_gray = 200.0 * (followup_gray - baseline_gray) / (followup_tissue + baseline_tissue)
    change_white = 200.0 * (followup_white - baseline_white) / (followup_tissue + baseline_tissue)

    longitudinal_metrics = {
        # Volume
        'baseline_brain_vol': baseline_volume,
        'followup_brain_vol': followup_volume,
        # Tissue volume
        'baseline_tissue_vol': baseline_tissue,
        'followup_tissue_vol': followup_tissue,
        # WM volume
        'baseline_wm_vol': baseline_white,
        'followup_wm_vol': followup_white,
        # GM volume
        'baseline_gm_vol': baseline_gray,
        'followup_gm_vol': followup_gray,
        # CSF Volume
        'baseline_csf_vol': baseline_csf,
        'followup_csf_vol': followup_csf,
        # PVBC
        'pvbc': change_tissue,
        'pvbc_white': change_white,
        'pvbc_gray': change_gray,
    }

    # Segmentation metrics
    if baseline_gt is not None and followup_gt is not None:
        baseline_dsc = nl.metrics.dsc(np.argmax(baseline_probs, axis=0), np.argmax(baseline_gt, axis=0))
        followup_dsc = nl.metrics.dsc(np.argmax(followup_probs, axis=0), np.argmax(followup_gt, axis=0))
        longitudinal_metrics.update({
            'baseline_dsc': baseline_dsc,
            'followup_dsc': followup_dsc})


    # Add normalized measures (volume wrt MNI skull size)
    if baseline_scale is not None and followup_scale is not None:
        longitudinal_metrics.update({
            'baseline_brain_vol_norm': longitudinal_metrics['baseline_brain_vol'] * baseline_scale,
            'followup_brain_vol_norm': longitudinal_metrics['followup_brain_vol'] * followup_scale,

            'baseline_tissue_vol_norm': longitudinal_metrics['baseline_tissue_vol'] * baseline_scale,
            'followup_tissue_vol_norm': longitudinal_metrics['followup_tissue_vol'] * followup_scale,

            'baseline_wm_vol_norm': longitudinal_metrics['baseline_wm_vol'] * baseline_scale,
            'followup_wm_vol_norm': longitudinal_metrics['followup_wm_vol'] * followup_scale,

            'baseline_gm_vol_norm': longitudinal_metrics['baseline_gm_vol'] * baseline_scale,
            'followup_gm_vol_norm': longitudinal_metrics['followup_gm_vol'] * followup_scale,

            'baseline_csf_vol_norm': longitudinal_metrics['baseline_csf_vol'] * baseline_scale,
            'followup_csf_vol_norm': longitudinal_metrics['followup_csf_vol'] * followup_scale,
        })

    return longitudinal_metrics


def compute_crossectional_measures(
        baseline_probs, baseline_nifti=None, baseline_gt=None, baseline_scale=None):

    baseline_probs = baseline_probs.astype('float64')

    # First transformation from voxels to mm3
    if baseline_nifti is not None:
        baseline_VOX2MM = np.prod(np.abs(baseline_nifti.header['pixdim'][1:4]))
    else:
        baseline_VOX2MM = 1.0

    # Tissue metrics
    baseline_volume = np.sum(baseline_probs[1:]) * baseline_VOX2MM

    baseline_csf = np.sum(baseline_probs[1]) * baseline_VOX2MM
    baseline_gray = np.sum(baseline_probs[2]) * baseline_VOX2MM
    baseline_white = np.sum(baseline_probs[3]) * baseline_VOX2MM
    baseline_tissue = baseline_gray + baseline_white

    crossectional_metrics = {
        # Volume
        'baseline_brain_vol': baseline_volume,
        # Tissue volume
        'baseline_tissue_vol': baseline_tissue,
        # WM volume
        'baseline_wm_vol': baseline_white,
        # GM volume
        'baseline_gm_vol': baseline_gray,
        # CSF Volume
        'baseline_csf_vol': baseline_csf,
    }

    # Segmentation metrics
    if baseline_gt is not None:
        baseline_dsc = nl.metrics.dsc(np.argmax(baseline_probs, axis=0), np.argmax(baseline_gt, axis=0))
        crossectional_metrics.update({
            'baseline_dsc': baseline_dsc})

    # Add normalized measures (volume wrt MNI skull size)
    if baseline_scale is not None:
        crossectional_metrics.update({
            'baseline_brain_vol_norm': crossectional_metrics['baseline_brain_vol'] * baseline_scale,
            'baseline_tissue_vol_norm': crossectional_metrics['baseline_tissue_vol'] * baseline_scale,
            'baseline_wm_vol_norm': crossectional_metrics['baseline_wm_vol'] * baseline_scale,
            'baseline_gm_vol_norm': crossectional_metrics['baseline_gm_vol'] * baseline_scale,
            'baseline_csf_vol_norm': crossectional_metrics['baseline_csf_vol'] * baseline_scale,
        })

    return crossectional_metrics

