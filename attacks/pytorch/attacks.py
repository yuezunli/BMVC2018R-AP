"""
Adversarial attacks for dense models: object detection, instance segmentation.
pytorch version
"""
import torch
import numpy as np


def get_2nd_probs(probs, labels=-1):
    """
    Get largest prob except labels
    :param probs: N*label_num
    :param labels: N; if labels == -1, the second largest prob will be extracted
    :return:
    """
    probs_np = probs.data.cpu().numpy()
    idxes = np.arange(0, probs_np.shape[0])
    if labels is -1:
        labels = np.argmax(probs_np, -1)
    # Get second large prob
    probs_np[idxes, labels] = 0
    labels_sec = np.argmax(probs_np, -1)
    probs = probs[idxes, labels_sec]
    return probs


def gen_adv(input, config, tp_target=None, tp_offset=None, real_offset=None, fp_target=None):
    """
    Generating adversarial perturbations
    :param input: KxCxHxW
    :param config: configurations
        - box_offset_regression_weight: true positives offset weight
        - box_conf_weight: true positives confidence weight
        - box_FP_weight: false positives weight
        - norm_ord: normalization method
        - eps: scaling parameter
        - norm_ball: clipping oeration
    :param tp_target: KxN (N is the length of true positives)
    :param tp_offset: KxNx4
    :param real_offset: KxNx4, the gt of offset
    :param fp_target: KxM (M is the length of false positives)
    :param mask: where to put noise on
    :return: perturbations KxCxHxW
    """
    loss_TP_loc, loss_TP_class, loss_FP_class = 0, 0, 0

    if tp_offset is not None and real_offset is not None:
        # tmp1 = (tp_offset - real_offset + 1e-5) * (tp_offset - real_offset + 1e-5)
        # loss_TP_loc = torch.exp(-torch.mean(tmp1)) * config.box_offset_regression_weight
        # Offset in [0., 1.]
        tmp1 = torch.abs(tp_offset - real_offset + 1e-5)
        # loss_TP_loc = (1 - torch.mean(tmp1))
        loss_TP_loc = torch.exp(-torch.mean(tmp1)) * config['box_offset_regression_weight']

    if tp_target is not None:
        loss_TP_class = torch.mean(-torch.log(tp_target)) * config['box_conf_weight']

    if fp_target is not None:
        loss_FP_class = torch.mean(-torch.log(fp_target)) * config['box_FP_weight']

    # Define gradient of loss wrt input
    total_loss = loss_TP_loc \
                 + loss_TP_class\
                 + loss_FP_class
    print('ADV Loss:' + str(total_loss.data.cpu().numpy()))
    total_loss.backward()
    grad = input.grad

    grad_np = grad.data.cpu().numpy()
    # Fast gradient method
    if config['norm_ord'] == 'sign':
        normalized_grad = np.sign(grad_np)
    elif config['norm_ord'] == 'L1':
        normalized_grad = grad_np / np.sum(np.abs(grad_np), axis=(2, 3), keepdims=True)
    elif config['norm_ord'] == 'L2':
        normalized_grad = grad_np / np.sqrt(np.sum(grad_np * grad_np, axis=(2,3), keepdims=True))
    else:
        raise ValueError(config['norm_ord'] + 'does not support...')

    # Multiply by constant epsilon
    eta = config['eps'] * normalized_grad
    # Clipping perturbation eta to self.ord norm ball
    if config['norm_ball']:
        eta = np.clip(eta, -config['norm_ball'], config['norm_ball'])
    return eta

