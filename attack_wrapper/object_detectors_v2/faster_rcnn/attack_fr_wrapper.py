from attack_wrapper.attack_wrapper import AttackWrapper
import numpy as np
from attacks.utils import select_detections
from attacks.pytorch import attacks
import torch
from torch.autograd import Variable


class FRAttackWrapper(AttackWrapper):
    def search_candidates(self, target_bbox, target_bbox_label, rpn_cls_prob, rpn_bboxes):
        """
        Select valid proposal candidates we focus on: reduce TP (class and offset)
        """
        # Select predictions which have large overlap with targets set
        rpn_labels_break = (rpn_cls_prob[:, 1] > 0.01).astype(np.int32)
        target_bbox_label = (target_bbox_label > 0).astype(np.int32)  # Binary the label
        # Select positive rpns that have large overlap with objects, we aim to reduce TP
        inds_break = select_detections(rpn_labels_break, rpn_bboxes, target_bbox,
                                       target_bbox_label,
                                       threshold=0.3)
        return inds_break

    def gen_adv(self, x, detection_scores, detection_bboxes_offset, real_bboxes_offset, inds_break):
        tp_target = detection_scores[:, 0]
        tp_target = tp_target[inds_break] if len(inds_break) else None

        real_bboxes_offset = Variable(torch.cuda.FloatTensor(real_bboxes_offset)) + 1e-5
        real_offset = real_bboxes_offset[inds_break, :] if len(inds_break) else None
        tp_offset = detection_bboxes_offset[inds_break, :] if len(inds_break) else None

        config = {'box_offset_regression_weight': self.offset_w,
                  'box_conf_weight': self.tp_w,
                  'box_FP_weight': 0,
                  'norm_ord': self.norm_ord,
                  'eps': self.eps,
                  'norm_ball': self.norm_ball}

        return attacks.gen_adv(input=x,
                               config=config,
                               tp_target=tp_target,
                               tp_offset=tp_offset,
                               real_offset=real_offset)
