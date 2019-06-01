"""
This is the wrapper interface for white-box attack on object detectors
"""
import numpy as np
import json, cv2
import os


class AttackWrapper(object):
    def __init__(self,
                 num_iter=200,
                 max_distortion=0.0001,
                 eps=20,
                 norm_ball=0,
                 tp_w=0.5,
                 offset_w=0.5,
                 norm_ord='L2'):
        self.num_iter = num_iter
        self.max_distortion = max_distortion
        self.eps = eps
        self.norm_ball = norm_ball
        self.tp_w = tp_w
        self.offset_w = offset_w
        self.norm_ord = norm_ord

    def gen_adv(self, *args):
        raise NotImplementedError('function gen_adv is not implemented.')

    @staticmethod
    def coco_label():
        with open(os.path.dirname(__file__) + '/coco_label.json', 'r') as f:
            coco_label = json.load(f)
        return coco_label

    def vis(self, im, bboxes, labels, label_dict):
        bboxes = np.int32(np.round(bboxes))
        im = np.clip(im.copy(), 0, 255).astype(np.uint8)
        for i in range(len(labels)):
            label = labels[i]
            class_name = label_dict[str(label)][0]
            bbox = bboxes[i]
            # Draw
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im, '{:s}'.format(class_name), (bbox[0], bbox[1] + 2), font, 0.5, (255, 0, 0), 2,
                        cv2.LINE_AA)
        return im

    def check_distortion(self, perts_np, max_v=255., min_v=0.):
        # Mean square error
        distortion = np.square(perts_np / (max_v - min_v))
        mean_distort = np.mean(distortion)
        return mean_distort
