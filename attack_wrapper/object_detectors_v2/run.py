
import numpy as np
import os, sys
sys.path.append('/../../')
import cv2
import importlib


def main(args):
    detector_module = importlib.import_module('{}.detector'.format(args.net))
    ObjDet = detector_module.ObjDet

    attack_module = importlib.import_module('{}.attack_fr_wrapper'.format(args.net))
    AttackWrapper = attack_module.FRAttackWrapper
    # Set up detector
    detector = ObjDet(net_name=args.base, class_num=81)
    detector.net.zero_grad()

    aw = AttackWrapper()
    coco_labels = AttackWrapper.coco_label()

    imgs_list = os.listdir(args.data_dir)
    for i in range(len(imgs_list)):
        image_name = os.path.basename(imgs_list[i]).split('.')[0]
        print(image_name)
        img = cv2.imread(os.path.join(args.data_dir, imgs_list[i])).astype(np.float32)
        origin_scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
        # Original detection results as gt
        blobs = detector.preproc_im(img)
        scores, boxes, rpn_cls_prob, rpn_bbox_pred_rec, rpn_bbox_pred = detector.predict_all(blobs)
        real_bboxes_offset = rpn_bbox_pred.data.cpu().numpy()
        dets_all_cls = detector.postproc_dets(scores, boxes)
        bboxes, labels = detector.convert_dets(dets_all_cls)
        # Vis img
        vis_im = aw.vis(img, bboxes, labels, coco_labels)
        cv2.imwrite(os.path.join(args.res_dir, image_name + '_orig.jpg'), vis_im)

        noise = np.zeros(list(detector.net._image.size()))
        for iter_n in range(aw.num_iter):
            print('iter: ' + str(iter_n))
            _, _, rpn_cls_prob, rpn_bbox_pred_rec, rpn_bbox_offset = detector.predict_all(blobs)

            inds_break = aw.search_candidates(bboxes, labels, rpn_cls_prob.data.cpu().numpy(), rpn_bbox_pred_rec)

            if len(inds_break) == 0 or iter_n == aw.num_iter - 1:
                break

            np_perturbed = aw.gen_adv(detector.net._image, rpn_cls_prob, rpn_bbox_offset, real_bboxes_offset, inds_break)
            im_np = detector.net._image.data.cpu().numpy() - np_perturbed
            blobs['data'] = np.transpose(im_np, (0, 2, 3, 1))
            noise += np_perturbed

            distort = aw.check_distortion(noise[0])
            print('Distortion: ' + str(distort))
            if distort > aw.max_distortion:  # Jump if distortion is larger than threshold
                break

        # Vis
        noise = np.transpose(noise[0], (1, 2, 0))
        vis_im = noise * 20 + 127
        cv2.imwrite(os.path.join(args.res_dir, image_name + '_noise.jpg'), vis_im)

        scores, boxes = detector.predict(blobs)
        dets_all_cls = detector.postproc_dets(scores, boxes)
        bboxes, labels = detector.convert_dets(dets_all_cls)
        img_resized = cv2.resize(img, None, None, fx=blobs['im_info'][-1], fy=blobs['im_info'][-1])
        vis_im = aw.vis(img_resized + noise, bboxes * blobs['im_info'][-1], labels, coco_labels)
        cv2.imwrite(os.path.join(args.res_dir, image_name + '_perturbed.jpg'), vis_im)

    print('Done...')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='faster_rcnn', help='faster_rcnn or ssd')
    parser.add_argument('--base', type=str, default='vgg16')
    parser.add_argument('--data_dir', type=str, default='demo/')
    parser.add_argument('--res_dir', type=str, default='res/')
    args = parser.parse_args()
    main(args)
