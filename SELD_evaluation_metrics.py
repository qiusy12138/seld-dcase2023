# Implements the localization and detection metrics proposed in [1] with extensions to support multi-instance of the same class from [2].
# 实现[1]中提出的定位和检测度量，并进行扩展以支持[2]中同类的多实例。
#
# [1] Joint Measurement of Localization and Detection of Sound Events、
# [1] 声音事件定位和检测的联合测量
# Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, Tuomas Virtanen 人名
# WASPAA 2019
#
# [2] Overview and Evaluation of Sound Event Localization and Detection in DCASE 2019
# [2] DCASE 2019声音事件定位与检测综述与评价
# Politis, Archontis, Annamaria Mesaros, Sharath Adavanne, Toni Heittola, and Tuomas Virtanen. 人名
# IEEE/ACM Transactions on Audio, Speech, and Language Processing (2020). 期刊名
#
# This script has MIT license
# 此脚本具有MIT许可证
#

import numpy as np

eps = np.finfo(np.float).eps
from scipy.optimize import linear_sum_assignment
from IPython import embed


class SELDMetrics(object):
    def __init__(self, doa_threshold=20, nb_classes=11, average='macro'):
        '''
            This class implements both the class-sensitive localization and location-sensitive detection metrics.
            Additionally, based on the user input, the corresponding averaging is performed within the segment.

        :param nb_classes: Number of sound classes. In the paper, nb_classes = 11
        :param doa_thresh: DOA threshold for location sensitive detection.

            这个类实现了类敏感的定位和位置敏感的检测度量。
            此外，基于用户输入，在分段内执行相应的平均。
        ：param nb_classes：声音类的数量。在本文中，nb_classe=11
        ：param doa_thresh：位置敏感检测的doa阈值。
        '''
        self._nb_classes = nb_classes

        # Variables for Location-senstive detection performance
        # 位置感知检测性能的变量
        self._TP = np.zeros(self._nb_classes)
        self._FP = np.zeros(self._nb_classes)
        self._FP_spatial = np.zeros(self._nb_classes)
        self._FN = np.zeros(self._nb_classes)

        self._Nref = np.zeros(self._nb_classes)

        self._spatial_T = doa_threshold

        self._S = 0
        self._D = 0
        self._I = 0

        # Variables for Class-sensitive localization performance
        # 类敏感本地化性能的变量
        self._total_DE = np.zeros(self._nb_classes)

        self._DE_TP = np.zeros(self._nb_classes)
        self._DE_FP = np.zeros(self._nb_classes)
        self._DE_FN = np.zeros(self._nb_classes)

        self._average = average

    def early_stopping_metric(self, _er, _f, _le, _lr):
        """
        Compute early stopping metric from sed and doa errors.

        :param sed_error: [error rate (0 to 1 range), f score (0 to 1 range)]
        :param doa_error: [doa error (in degrees), frame recall (0 to 1 range)]
        :return: early stopping metric result

        根据sed和doa误差计算早期停止度量。
        ：param sed_error：[错误率（0到1范围），f分数（0到一范围）]
        ：param doa_error：[doa错误（以度为单位），帧调用（0到1范围）]
        ：return：提前停止度量结果
        """
        seld_metric = np.mean([
            _er,
            1 - _f,
            _le / 180,
            1 - _lr
        ], 0)
        return seld_metric

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores

        收集最终SELD分数
        ：return：返回位置敏感检测分数和类敏感定位分数
        '''

        
        ER = (self._S + self._D + self._I) / (self._Nref.sum() + eps)
        classwise_results = []
        if self._average == 'micro':
            # Location-sensitive detection performance
            # 位置敏感检测性能
            F = self._TP.sum() / (eps + self._TP.sum() + self._FP_spatial.sum() + 0.5 * (self._FP.sum() + self._FN.sum()))

            # Class-sensitive localization performance
            LE = self._total_DE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else 180
            LR = self._DE_TP.sum() / (eps + self._DE_TP.sum() + self._DE_FN.sum())

            SELD_scr = self.early_stopping_metric(ER, F, LE, LR)

        elif self._average == 'macro':
            # Location-sensitive detection performance
            # 位置敏感检测性能
            F = self._TP / (eps + self._TP + self._FP_spatial + 0.5 * (self._FP + self._FN))

            # Class-sensitive localization performance
            # 类敏感的定位性能
            LE = self._total_DE / (self._DE_TP + eps)
            LE[self._DE_TP==0] = 180.0
            LR = self._DE_TP / (eps + self._DE_TP + self._DE_FN)

            SELD_scr = self.early_stopping_metric(np.repeat(ER, self._nb_classes), F, LE, LR)
            classwise_results = np.array([np.repeat(ER, self._nb_classes), F, LE, LR, SELD_scr])
            F, LE, LR, SELD_scr = F.mean(), LE.mean(), LR.mean(), SELD_scr.mean()
        return ER, F, LE, LR, SELD_scr, classwise_results

    def update_seld_scores(self, pred, gt):
        '''
        Implements the spatial error averaging according to equation 5 in the paper [1] (see papers in the title of the code).
        Adds the multitrack extensions proposed in paper [2]

        The input pred/gt can either both be Cartesian or Degrees

        :param pred: dictionary containing class-wise prediction results for each N-seconds segment block
        :param gt: dictionary containing class-wise groundtruth for each N-seconds segment block

        根据论文[1]中的公式5实现空间误差平均（见代码标题中的论文）。
        增加了论文[2]中提出的多轨道扩展
        输入pred/gt既可以是笛卡尔坐标，也可以是度
        ：param pred：包含每个N秒分段块的类预测结果的字典
        ：param gt：字典，包含每个N秒段块的类基本事实
        '''
        for block_cnt in range(len(gt.keys())):
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # Counting the number of referece tracks for each class in the segment
                # 计算段中每个类的引用轨道数
                nb_gt_doas = max([len(val) for val in gt[block_cnt][class_cnt][0][1]]) if class_cnt in gt[block_cnt] else None
                nb_pred_doas = max([len(val) for val in pred[block_cnt][class_cnt][0][1]]) if class_cnt in pred[block_cnt] else None
                if nb_gt_doas is not None:
                    self._Nref[class_cnt] += nb_gt_doas
                if class_cnt in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # True positives or False positive case
                    # 真阳性TP或假阳性例子FP

                    # NOTE: For multiple tracks per class, associate the predicted DOAs to corresponding reference
                    # 注：对于每个类别的多个轨道，将预测的DOA与相应的参考相关联
                    # DOA-tracks using hungarian algorithm and then compute the average spatial distance between
                    # the associated reference-predicted tracks.
                    # #使用匈牙利算法跟踪DOA，然后计算相关参考预测轨道之间的平均空间距离。

                    # Reference and predicted track matching
                    # 参考和预测轨道匹配
                    matched_track_dist = {}
                    matched_track_cnt = {}
                    gt_ind_list = gt[block_cnt][class_cnt][0][0]
                    pred_ind_list = pred[block_cnt][class_cnt][0][0]
                    for gt_ind, gt_val in enumerate(gt_ind_list):
                        if gt_val in pred_ind_list:
                            gt_arr = np.array(gt[block_cnt][class_cnt][0][1][gt_ind])
                            gt_ids = np.arange(len(gt_arr[:, -1])) #TODO if the reference has track IDS use here - gt_arr[:, -1]
                            gt_doas = gt_arr[:, 1:]

                            pred_ind = pred_ind_list.index(gt_val)
                            pred_arr = np.array(pred[block_cnt][class_cnt][0][1][pred_ind])
                            pred_doas = pred_arr[:, 1:]

                            if gt_doas.shape[-1] == 2: # convert DOAs to radians, if the input is in degrees
                                gt_doas = gt_doas * np.pi / 180.
                                pred_doas = pred_doas * np.pi / 180.

                            dist_list, row_inds, col_inds = least_distance_between_gt_pred(gt_doas, pred_doas)

                            # Collect the frame-wise distance between matched ref-pred DOA pairs
                            # 收集匹配的ref-pred DOA对之间的逐帧距离
                            for dist_cnt, dist_val in enumerate(dist_list):
                                matched_gt_track = gt_ids[row_inds[dist_cnt]]
                                if matched_gt_track not in matched_track_dist:
                                    matched_track_dist[matched_gt_track], matched_track_cnt[matched_gt_track] = [], []
                                matched_track_dist[matched_gt_track].append(dist_val)
                                matched_track_cnt[matched_gt_track].append(pred_ind)

                    # Update evaluation metrics based on the distance between ref-pred tracks
                    # 根据ref pred轨迹之间的距离更新评估指标
                    if len(matched_track_dist) == 0:
                        # if no tracks are found. This occurs when the predicted DOAs are not aligned frame-wise to the reference DOAs
                        loc_FN += nb_pred_doas
                        self._FN[class_cnt] += nb_pred_doas
                        self._DE_FN[class_cnt] += nb_pred_doas
                    else:
                        # for the associated ref-pred tracks compute the metrics
                        # 对于相关的ref pred轨迹，计算度量
                        for track_id in matched_track_dist:
                            total_spatial_dist = sum(matched_track_dist[track_id])
                            total_framewise_matching_doa = len(matched_track_cnt[track_id])
                            avg_spatial_dist = total_spatial_dist / total_framewise_matching_doa

                            # Class-sensitive localization performance
                            # 类敏感的定位性能
                            self._total_DE[class_cnt] += avg_spatial_dist
                            self._DE_TP[class_cnt] += 1

                            # Location-sensitive detection performance
                            # 位置敏感检测性能
                            if avg_spatial_dist <= self._spatial_T:
                                self._TP[class_cnt] += 1
                            else:
                                loc_FP += 1
                                self._FP_spatial[class_cnt] += 1
                        # in the multi-instance of same class scenario, if the number of predicted tracks are greater
                        # than reference tracks count as FP, if it less than reference count as FN
                        # 在同一类场景的多实例中，如果预测轨道的数量大于参考轨道的数量，则将其计数为FP，如果小于参考轨道的计数为FN
                        if nb_pred_doas > nb_gt_doas:
                            # False positive，FP，错分的真实例
                            loc_FP += (nb_pred_doas-nb_gt_doas)
                            self._FP[class_cnt] += (nb_pred_doas-nb_gt_doas)
                            self._DE_FP[class_cnt] += (nb_pred_doas-nb_gt_doas)
                        elif nb_pred_doas < nb_gt_doas:
                            # False negative，FN，错分的假实例
                            loc_FN += (nb_gt_doas-nb_pred_doas)
                            self._FN[class_cnt] += (nb_gt_doas-nb_pred_doas)
                            self._DE_FN[class_cnt] += (nb_gt_doas-nb_pred_doas)
                elif class_cnt in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # False negative，FN，错分的假实例
                    loc_FN += nb_gt_doas
                    self._FN[class_cnt] += nb_gt_doas
                    self._DE_FN[class_cnt] += nb_gt_doas
                elif class_cnt not in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # False positive，FP，错分的真实例
                    loc_FP += nb_pred_doas
                    self._FP[class_cnt] += nb_pred_doas
                    self._DE_FP[class_cnt] += nb_pred_doas

            self._S += np.minimum(loc_FP, loc_FN)
            self._D += np.maximum(0, loc_FN - loc_FP)
            self._I += np.maximum(0, loc_FP - loc_FN)
        return


def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    Angular distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance

    :return: angular distance in degrees

    两个球面坐标之间的角距离
    更多详情：https://en.wikipedia.org/wiki/Great-circle_distance
    ：return：角度距离，单位为度
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    # 确保dist值在-1到1的范围内，否则np.arccos将终止进程
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees

    两个笛卡尔坐标之间的角距离
    更多详情：https://en.wikipedia.org/wiki/Great-circle_distance
    检查“从弦长”部分
    ：return：角度距离，单位为度
    """
    # Normalize the Cartesian vectors
    # 规范化笛卡尔矢量
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

    # Compute the distance
    # 计算距离
    dist = x1*x2 + y1*y2 + z1*z2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def least_distance_between_gt_pred(gt_list, pred_list):
    """
        Shortest distance between two sets of DOA coordinates. Given a set of groundtruth coordinates,
        and its respective predicted coordinates, we calculate the distance between each of the
        coordinate pairs resulting in a matrix of distances, where one axis represents the number of groundtruth
        coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
        groundtruth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
        least cost in this distance matrix.
        :param gt_list_xyz: list of ground-truth Cartesian or Polar coordinates in Radians
        :param pred_list_xyz: list of predicted Carteisan or Polar coordinates in Radians
        :return: cost - distance
        :return: less - number of DOA's missed
        :return: extra - number of DOA's over-estimated

        两组DOA坐标之间的最短距离。给定一组地面实况坐标及其各自的预测坐标，我们计算每个坐标对之间的距离，得出距离矩阵，其中一个轴表示地面实况坐标的数量，
        另一个轴代表预测坐标。估计的峰值数量不必与groundtruth中的峰值数量相同，因此距离矩阵并不总是一个平方矩阵。我们使用匈牙利算法来找到该距离矩阵中的最小代价。
        ：param gt_list_xyz：以弧度为单位的笛卡尔坐标或极坐标的地面实况列表
        ：param pred_list_xyz：以弧度为单位的预测Carteisan或Polar坐标列表
        ：return：成本-距离
        ：return：less-错过的DOA数量
        ：return：额外-DOA过估计数
    """

    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))

    if gt_len and pred_len:
        if len(gt_list[0]) == 3: #Cartesian
            x1, y1, z1, x2, y2, z2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], gt_list[ind_pairs[:, 0], 2], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1], pred_list[ind_pairs[:, 1], 2]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2)
        else:
            az1, ele1, az2, ele2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2)

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind]
    return cost, row_ind, col_ind
