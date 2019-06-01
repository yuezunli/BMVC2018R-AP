"""
faster-rcnn wrapper
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
from .init_env import DETECTOR_DIR, ROOT_DIR
sys.path.append(DETECTOR_DIR + '/tools')
import _init_paths
from model.config import cfg, cfg_from_file, cfg_from_list
from model.test import im_detect
from model.nms_wrapper import nms

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
from model.config import cfg
from model.bbox_transform import clip_boxes, bbox_transform_inv
from model.test import _get_blobs
import torch
import numpy as np
import pprint


class ObjDet(object):
    """
    face detector wrapper
    """
    def __init__(self, net_name, class_num):
        print('Called with args:')
        print('Using config:')
        pprint.pprint(cfg)

        # load network
        if net_name == 'vgg16':
            net = vgg16()
        elif net_name == 'res50':
            net = resnetv1(num_layers=50)
        elif net_name == 'res101':
            net = resnetv1(num_layers=101)
        elif net_name == 'res152':
            net = resnetv1(num_layers=152)
        elif net_name == 'mobile':
            net = mobilenetv1()
        else:
            raise NotImplementedError

        # load model
        net.create_architecture(81, tag='default', anchor_scales=[4, 8, 16, 32])

        if net_name != 'vgg16':
            saved_model = os.path.join(DETECTOR_DIR, 'output', net_name, 'converted_from_tf',
                                   'coco_900k-1190k', net_name + '_faster_rcnn_iter_1190000.pth')
        else:
            saved_model = os.path.join(DETECTOR_DIR, 'output', net_name, 'finetune_from_scratch',
                                       'coco_350k-490k', net_name + '_faster_rcnn_iter_490000.pth')

        if not os.path.isfile(saved_model):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                           'our server and place them properly?').format(saved_model))

        print(('Loading model check point from {:s}').format(saved_model))
        net.load_state_dict(torch.load(saved_model, map_location=lambda storage, loc: storage))
        print('Loaded.')

        net.eval()
        if not torch.cuda.is_available():
            net._device = 'cpu'
        net.to(net._device)

        self.net = net
        self.class_num = class_num

    def preproc_im(self, im):
        # Process image
        blobs, im_scales = _get_blobs(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"

        im_blob = blobs['data']
        blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
        return blobs

    def predict_all(self, blobs):
        img_h, img_w, im_scale = blobs['im_info']
        _, scores, bbox_pred, rois, rpn_cls_prob, rpn_bbox_pred, rpn_bbox_pred_rec = self.net.test_image_adv(blobs['data'], blobs['im_info'])

        rpn_bbox_pred_rec = rpn_bbox_pred_rec / im_scale
        boxes = rois[:, 1:5] / im_scale
        scores = np.reshape(scores, [scores.shape[0], -1])
        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(torch.from_numpy(boxes), torch.from_numpy(box_deltas))
        pred_boxes = clip_boxes(pred_boxes, [img_h, img_w]).numpy()

        # Recover rpn bboxes
        rpn_bbox_pred = rpn_bbox_pred.view([-1, 4])
        rpn_cls_prob = torch.stack([rpn_cls_prob[:, :, :, 0:self.net._num_anchors],
                                    rpn_cls_prob[:, :, :, self.net._num_anchors:]], dim=-1)
        rpn_cls_prob = rpn_cls_prob.contiguous().view((-1, 2))
        return scores, pred_boxes, rpn_cls_prob, rpn_bbox_pred_rec, rpn_bbox_pred

    def predict(self, blobs):
        """
        Test image
        :param im: bgr
        :return:
        """
        # Test
        img_h, img_w, im_scale = blobs['im_info']
        _, scores, bbox_pred, rois = self.net.test_image(blobs['data'], blobs['im_info'])

        boxes = rois[:, 1:5] / im_scale
        scores = np.reshape(scores, [scores.shape[0], -1])
        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(torch.from_numpy(boxes), torch.from_numpy(box_deltas))
        pred_boxes = clip_boxes(pred_boxes, [img_h, img_w]).numpy()

        return scores, pred_boxes

    def postproc_dets(self, scores, boxes):
        thresh = 0.1
        max_per_image = 100
        # Visualize detections for each class
        dets_all_cls = [[] for _ in range(self.class_num)]
        # skip j = 0, because it's the background class
        for j in range(1, self.class_num):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(torch.from_numpy(cls_dets), cfg.TEST.NMS).numpy() if cls_dets.size > 0 else []
            cls_dets = cls_dets[keep, :]
            dets_all_cls[j] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([dets_all_cls[j][:, -1]
                                      for j in range(1, self.class_num)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, self.class_num):
                    keep = np.where(dets_all_cls[j][:, -1] >= image_thresh)[0]
                    dets_all_cls[j] = dets_all_cls[j][keep, :]
        return dets_all_cls

    def convert_dets(self, dets_all_cls, thresh=0.5):
        dets_label = []
        dets_box = []
        for label in range(len(dets_all_cls)):
            dets = dets_all_cls[label]
            if len(dets) == 0:
                continue
            inds = np.where(dets[:, -1] > thresh)[0]
            if len(inds) == 0:
                continue
            dets = dets[inds, :]
            dets_label.append([label] * dets.shape[0])
            dets_box.append(dets[:, :4])
        if len(dets_label):
            dets_label = np.concatenate(dets_label)
            dets_box = np.concatenate(dets_box, 0)
        return np.array(dets_box), np.array(dets_label)