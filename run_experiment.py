import json
import os.path
import random
import socket
import time
import torch
from torch import nn

from lats_load import *

from lats_eval_utils import *

from LITS_utils import train_LITS, segment_LITS_dataset
from LITS_evaluate import compute_LITS_lesioned_metrics

if socket.gethostname() == 'labvisio01':
    DATASET_PATH = '/media/user/dades/DATASETS-WS/LATS/'
    EXPERIMENTS_PATH = '/home/user/mic_home/experiments/LATS/runs'
    WORKSPACES_PATH = '/media/user/dades/DATASETS-WS/LATS-workspaces/'
elif socket.gethostname() == 'mic2' or socket.gethostname() == 'mic3':
    DATASET_PATH = '/home/albert/datasets/LATS'
    WORKSPACES_PATH = '/home/albert/datasets/LATS-workspaces/'
    EXPERIMENTS_PATH = '/home/albert/mic_home/experiments/LATS/runs'
else:
    raise NameError(socket.gethostname())

campinas_train, campinas_test = get_campinas_train_test_ids(os.path.join(DATASET_PATH, 'campinas-tissue'))

ch2016_train, ch2016_test = get_challenge2016_train_test_ids(os.path.join(DATASET_PATH, 'Challenge2016-lesion-atlas'))
isbi_train, isbi_test = get_isbi2015_train_test_ids(os.path.join(DATASET_PATH, 'ISBI-lesion-atlas'))
wmh_train, wmh_test = get_wmh2017_train_test_ids(os.path.join(DATASET_PATH, 'WMH20173D-lesion-atlas'))

print('ch2016_train', len(ch2016_train))
print('isbi_train', len(isbi_train))
print('wmh_train', len(wmh_train))
print('ch2016_train + isbi_train + wmh_train', len(ch2016_train + isbi_train + wmh_train))

print('ch2016_test', len(ch2016_test))
print('isbi_test', len(isbi_test))
print('wmh_test', len(wmh_test))
print('ch2016_test + isbi_test + wmh_test', len(ch2016_test + isbi_test + wmh_test))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

experiment_queue = {
    'LITSv2_p40': {'patch_shape': (40,) * 3, 'num_patches': 75000},
    'LITSv2_p32': {'patch_shape': (32,) * 3, 'num_patches': 150000},
    'LITSv2_p24': {'patch_shape': (24,) * 3, 'num_patches': 350000,},
    'LITSv2_p16': {'patch_shape': (16,) * 3, 'num_patches': 800000},
    'LITSv2_p8': {'patch_shape': (8,) * 3, 'num_patches': 2000000, 'extraction_step': (6, 6, 6)},
    
    'LITSv2_p16_BIG':
        {'patch_shape': (16,) * 3, 'num_patches': 1000000, 'extraction_step': (5, 5, 5)},
}

# Train network with lesion masks and segment test ids
for exp_name, exp_cfg in experiment_queue.items():
    line = '\n' + '-' * 50 + '\n'
    print(line + f'EXPERIMENT: {exp_name}' + line)

    experiment_path = os.path.join(EXPERIMENTS_PATH, exp_name)
    os.makedirs(experiment_path, exist_ok=True)

    segmentation_workspace_path = os.path.join(WORKSPACES_PATH, exp_name)
    os.makedirs(segmentation_workspace_path, exist_ok=True)
    
    if 'model_filepath' not in exp_cfg:
        model_filepath = os.path.join(experiment_path, f'wnet_lits_{exp_name}.pt')
    else:
        print(f'Using existing model @{exp_cfg["model_filepath"]}')
        print(f'TRAINING DISABLED')
        model_filepath = exp_cfg['model_filepath']
        exp_cfg['do_train'] = False
    
    metircs_csv_filepath = os.path.join(experiment_path, f'{exp_name}_differences.csv')


    if 'num_lesion_masks' in exp_cfg:
        MAX_LESION_MASKS = exp_cfg['num_lesion_masks']
    else:
        MAX_LESION_MASKS = None

    if 'dont_train' not in exp_cfg:
        print('=' * 30 + '\nTRAINING LITS\n' + '=' * 30)

        start = time.time()
        dataset_with_train = load_healthy_lesion_dataset(
            healthy_lesion_path=os.path.join(DATASET_PATH, 'campinas-lesion'),
            healthy_ids=campinas_train,
            lesion_ids=ch2016_train + isbi_train + wmh_train,
            max_lesion_masks_per_healthy=MAX_LESION_MASKS, # MAX is 30, but 25 is better
            load_images=True)
        print(f'Loading took {time.time() - start:.2f}s')

        train_LITS(
            cfg_run=exp_cfg,
            dataset=dataset_with_train,
            fpath_trained_model=model_filepath)
        del dataset_with_train


    print('=' * 30 + '\nTESTING LITS\n' + '=' * 30)
    dataset_with_test = load_healthy_lesion_dataset(
        healthy_lesion_path=os.path.join(DATASET_PATH, 'campinas-lesion'),
        healthy_ids=campinas_test,
        lesion_ids=ch2016_test + isbi_test + wmh_test,
        max_lesion_masks_per_healthy=None,
        load_images=False)

    segment_LITS_dataset(
        cfg_run=exp_cfg,
        test_dataset=dataset_with_test,
        trained_model=model_filepath,
        seg_workspace=segmentation_workspace_path,
        decimals=5)

    evaluate_lats_differences(
        seg_workspace=segmentation_workspace_path,
        healthy_lesion_dataset_path=os.path.join(DATASET_PATH, 'campinas-lesion'),
        csv_out_filepath=metircs_csv_filepath)
    del dataset_with_test

    ###################### PAPER METRICS
    print('Computing paper metrics...')
    healthy_seg_dict = {}  # os.path.join(seg_workspace, f'{case["id"]}__probs.nii.gz'))
    lesioned_seg_dict = {}  # os.path.join(seg_workspace, f'{case["id"]}__lesion__{lesion_id}__probs.nii.gz')

    # Get list with all non-lesion segmentation ids
    all_prob_filenames = list_filenames(segmentation_workspace_path)

    ### HEALTHY SEG
    healthy_prob_filenames = [pf for pf in all_prob_filenames if len(pf.split('__')) == 2]
    healthy_ids = [hf.split('__')[0] for hf in healthy_prob_filenames]
    for healthy_id, healthy_prob_filename in zip(healthy_ids, healthy_prob_filenames):
        healthy_seg_filepath = os.path.join(segmentation_workspace_path, healthy_prob_filename)
        if os.path.isfile(healthy_seg_filepath):
            healthy_seg_dict[healthy_id] = healthy_seg_filepath
        
        ## LESIONED SEG
        lesion_prob_filenames = [pf for pf in all_prob_filenames
                                 if len(pf.split('__')) == 4 and pf.split('__')[0] == healthy_id]
        lesioned_case_ids = [healthy_id + '__' + lpf.split('__')[2] for lpf in lesion_prob_filenames]

        for lesioned_id, lesion_prob_filename in zip(lesioned_case_ids, lesion_prob_filenames):
            lesion_prob_filepath = os.path.join(segmentation_workspace_path, lesion_prob_filename)
            if os.path.isfile(lesion_prob_filepath):
                lesioned_seg_dict[lesioned_id] = lesion_prob_filepath
    
    compute_LITS_lesioned_metrics(
            healthy_seg_fp_dict=healthy_seg_dict,
            lesioned_seg_fp_dict=lesioned_seg_dict,
            healthy_lesion_dataset_path=os.path.join(DATASET_PATH, 'campinas-lesion'),
            csv_out_filepath=os.path.join(experiment_path, f'{exp_name}_PAPER_metrics.csv'))

    




    