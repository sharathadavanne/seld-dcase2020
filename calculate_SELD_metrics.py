import os
from metrics import SELD_evaluation_metrics
import cls_feature_class
import parameter
import numpy as np


def get_nb_files(_pred_file_list, _group='split'):
    _group_ind = {'ir': 4, 'ov': 21}
    _cnt_dict = {}
    for _filename in _pred_file_list:

        if _group == 'all':
            _ind = 0
        else:
            _ind = int(_filename[_group_ind[_group]])

        if _ind not in _cnt_dict:
            _cnt_dict[_ind] = []
        _cnt_dict[_ind].append(_filename)

    return _cnt_dict


# --------------------------- MAIN SCRIPT STARTS HERE -------------------------------------------

# Metric evaluation at a fixed hop length of 20 ms (=0.02 seconds) and 3000 frames for a 60s audio

# INPUT DIRECTORY
ref_desc_files = '/home/sharath/Downloads/SELD_2020_dataset/metadata_eval' # reference description directory location
pred_output_format_files = '/home/sharath/Downloads/SELD_2020_dataset/5_foa_eval' # predicted output format directory location

# Load feature class
params = parameter.get_params()
feat_cls = cls_feature_class.FeatureClass(params)

# collect reference files info
ref_files = os.listdir(ref_desc_files)
nb_ref_files = len(ref_files)

# collect predicted files info
pred_files = os.listdir(pred_output_format_files)
nb_pred_files = len(pred_files)

if nb_ref_files != nb_pred_files:
    print('ERROR: Mismatch. Reference has {} and prediction has {} files'.format(nb_ref_files, nb_pred_files))
    exit()

# Calculate scores for different splits, overlapping sound events, and impulse responses (reverberant scenes)
score_type_list = ['all', 'ov', 'ir']
print('Number of predicted files: {}\nNumber of reference files: {}'.format(nb_pred_files, nb_ref_files))
print('\nCalculating {} scores for {}'.format(score_type_list, os.path.basename(pred_output_format_files)))

for score_type in score_type_list:
    print('\n\n---------------------------------------------------------------------------------------------------')
    print('------------------------------------  {}   ---------------------------------------------'.format('Total score' if score_type=='all' else 'score per {}'.format(score_type)))
    print('---------------------------------------------------------------------------------------------------')

    split_cnt_dict = get_nb_files(pred_files, _group=score_type) # collect files corresponding to score_type
    # Calculate scores across files for a given score_type
    for split_key in np.sort(list(split_cnt_dict)):
        # Load evaluation metric class
        eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=feat_cls.get_nb_classes(), doa_threshold=20)
        for pred_cnt, pred_file in enumerate(split_cnt_dict[split_key]):
            # Load predicted output format file
            pred_dict = feat_cls.load_output_format_file(os.path.join(pred_output_format_files, pred_file))
            pred_labels = feat_cls.segment_labels(pred_dict, feat_cls.get_nb_frames())

            # Load reference description file
            gt_dict_polar = feat_cls.load_output_format_file(os.path.join(ref_desc_files, pred_file.replace('.npy', '.csv')))
            gt_dict = feat_cls.convert_output_format_polar_to_cartesian_(gt_dict_polar)
            gt_labels = feat_cls.segment_labels(gt_dict, feat_cls.get_nb_frames())

            # Calculated scores
            eval.update_seld_scores_xyz(pred_labels, gt_labels)

        # Overall SED and DOA scores
        er, f, de, de_f = eval.compute_seld_scores()
        seld_scr = SELD_evaluation_metrics.early_stopping_metric([er, f], [de, de_f])

        print('\nAverage score for {} {} data'.format(score_type, 'fold' if score_type=='all' else split_key))
        print('SELD score (early stopping metric): {:0.2f}'.format(seld_scr))
        print('SED metrics: Error rate: {:0.2f}, F-score:{:0.1f}'.format(er, 100*f))
        print('DOA metrics: DOA error: {:0.1f}, F-score:{:0.1f}'.format(de, 100*de_f))

