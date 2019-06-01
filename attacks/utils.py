import numpy as np
import sys
sys.path.append('../../')
from py_utils.box_utils.proc_box import jaccard_np


def select_detections(detection_labels, detection_bboxes, target_bboxes, target_labels, threshold=0.3):
    """
    Select detections which have overlap larger than threshold
    :param detection_scores:
    :param detection_bboxes: xyxy
    :param threshold:
    :return:
    """
    inds_list = []
    for i, target_bbox in enumerate(target_bboxes):
        target_label = target_labels[i]
        inds = np.where(detection_labels == target_label)[0]
        IoU = jaccard_np([target_bbox], detection_bboxes[inds, :])[0]
        inds_IoU = inds[np.where(IoU > threshold)[0]]
        if len(inds_IoU):
          inds_list.append(inds_IoU)

    if inds_list == []:
        keep_inds = []
    else:
        keep_inds = np.unique(np.concatenate(inds_list))

    return keep_inds
