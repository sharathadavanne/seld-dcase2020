#
# Implements the localization and detection metrics proposed in the paper
#
# Joint Measurement of Localization and Detection of Sound Events
# Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, Tuomas Virtanen
# WASPAA 2019
#
#
# This script has MIT license
#

import numpy as np
from IPython import  embed
eps = np.finfo(np.float).eps


class SELDMetrics(object):
    def __init__(self, doa_threshold=10, nb_classes=11):
        '''
            This class implements both the class-sensitive localization and location-sensitive detection metrics.
            Additionally, based on the user input, the corresponding averaging is performed within the segment.

        :param nb_classes: Number of sound classes. In the paper, nb_classes = 11
        :param doa_thresh: DOA threshold for location sensitive detection.
        '''

        self._TP = 0
        self._FP = 0
        self._TN = 0
        self._FN = 0

        self._S = 0
        self._D = 0
        self._I = 0

        self._Nref = 0
        self._Nsys = 0

        self._total_DE = 0
        self._DE_TP = 0

        self._spatial_T = doa_threshold
        self._nb_classes = nb_classes

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''

        # Location-senstive detection performance
        ER = (self._S + self._D + self._I) / float(self._Nref + eps)

        prec = float(self._TP) / float(self._Nsys + eps)
        recall = float(self._TP) / float(self._Nref + eps)
        F = 2 * prec * recall / (prec + recall + eps)

        # Class-sensitive localization performance
        DE = self._total_DE / float(self._DE_TP + eps)

        DE_prec = float(self._DE_TP) / float(self._Nsys + eps)
        DE_recall = float(self._DE_TP) / float(self._Nref + eps)
        DE_F = 2 * DE_prec * DE_recall / (DE_prec + DE_recall + eps)

        return ER, F, DE, DE_F

    def update_seld_scores(self, pred, gt):
        '''
        Implements the spatial error averaging according to equation [5] in the paper

        :param pred: dictionary containing class-wise prediction results for each N-seconds segment block
        :param gt: dictionary containing class-wise groundtruth for each N-seconds segment block
        '''
        for block_cnt in range(len(gt.keys())):
            # print('\nblock_cnt', block_cnt, end='')
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # print('\tclass:', class_cnt, end='')
                # Counting the number of ref and sys outputs should include the number of tracks for each class in the segment
                if class_cnt in gt[block_cnt]:
                    self._Nref += 1
                if class_cnt in pred[block_cnt]:
                    self._Nsys += 1

                if class_cnt in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # True positives or False negative case

                    # NOTE: For multiple tracks per class, identify multiple tracks using hungarian algorithm and then
                    # calculate the spatial distance using the following code. In the current code, we are assuming only
                    # one track per class.
                    # embed()
                    gt_list = np.squeeze(np.array(gt[block_cnt][class_cnt][0][1]), 1) * np.pi / 180
                    gt_azi_list, gt_ele_list = gt_list[:, 0], gt_list[:, 1]

                    # gt_azi_list = np.array(gt[block_cnt][class_cnt][0][1]) * np.pi / 180
                    # gt_ele_list = np.array(gt[block_cnt][class_cnt][0][2]) * np.pi / 180

                    pred_list = np.squeeze(np.array(pred[block_cnt][class_cnt][0][1]), 1) * np.pi / 180
                    pred_azi_list, pred_ele_list = pred_list[:, 0], pred_list[:, 1]

                    total_spatial_dist = 0
                    total_framewise_matching_doa = 0
                    gt_ind_list = gt[block_cnt][class_cnt][0][0]
                    pred_ind_list = pred[block_cnt][class_cnt][0][0]
                    for gt_ind, gt_val in enumerate(gt_ind_list):
                        if gt_val in pred_ind_list:
                            total_framewise_matching_doa += 1
                            pred_ind = pred_ind_list.index(gt_val)
                            total_spatial_dist += distance_between_spherical_coordinates_rad(gt_azi_list[gt_ind], gt_ele_list[gt_ind], pred_azi_list[pred_ind], pred_ele_list[pred_ind])

                    if total_spatial_dist == 0 and total_framewise_matching_doa == 0:
                        loc_FN += 1
                        self._FN += 1
                    else:
                        avg_spatial_dist = (total_spatial_dist / total_framewise_matching_doa)

                        self._total_DE += avg_spatial_dist
                        self._DE_TP += 1

                        if avg_spatial_dist <= self._spatial_T:
                            self._TP += 1
                        else:
                            loc_FN += 1
                            self._FN += 1
                elif class_cnt in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # False negative
                    loc_FN += 1
                    self._FN += 1
                elif class_cnt not in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # False positive
                    loc_FP += 1
                    self._FP += 1
                elif class_cnt not in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # True negative
                    self._TN += 1

            self._S += np.minimum(loc_FP, loc_FN)
            self._D += np.maximum(0, loc_FN - loc_FP)
            self._I += np.maximum(0, loc_FP - loc_FN)
        return


def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    Angular distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance

    :return: angular distance in degrees
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist
