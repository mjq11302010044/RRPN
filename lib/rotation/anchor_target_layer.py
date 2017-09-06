# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import caffe
import yaml

from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from rotation.rbbox_transform import rbbox_transform
from rotation.rbbox import angle_diff
from rotation.rbbox_overlaps import rbbx_overlaps
from inside_judge import ind_inside, condinate_rotate


DEBUG = False



class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        anchor_scales = layer_params.get('scales', (8,16,32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']

	# [x_ctr, y_ctr, height, width, theta] anti-clock-wise angle
	self.bbox_para_num = 5

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::self.bbox_para_num] - self._anchors[:, 0::self.bbox_para_num],
                self._anchors[:, 3::self.bbox_para_num] - self._anchors[:, 1::self.bbox_para_num],
            ))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, self.bbox_para_num))
            self._squared_sums = np.zeros((1, self.bbox_para_num))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)

        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        A = self._num_anchors
        # labels
        top[0].reshape(1, 1, A * height, width)
        # bbox_targets
        top[1].reshape(1, A * self.bbox_para_num, height, width)
        # bbox_inside_weights
        top[2].reshape(1, A * self.bbox_para_num, height, width)
        # bbox_outside_weights
        top[3].reshape(1, A * self.bbox_para_num, height, width)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]
        # GT boxes (x_ctr, y_ctr, height, width, theta, label)
        gt_boxes = bottom[1].data
        # im_info
        im_info = bottom[2].data[0, :]

        if DEBUG:
            print ''
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', gt_boxes.shape
            print 'rpn: gt_boxes', gt_boxes

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),np.zeros((3, width*height)))).transpose()
        # add A anchors (1, A, 5) to
        # cell K shifts (K, 1, 5) to get
        # shift anchors (K, A, 5)
        # reshape to (K*A, 5) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, self.bbox_para_num)) +
                       shifts.reshape((1, K, self.bbox_para_num)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, self.bbox_para_num))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        # inds_inside = np.where(
            # (all_anchors[:, 0] >= -self._allowed_border) &
            # (all_anchors[:, 1] >= -self._allowed_border) &
            # (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            # (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        # )[0]

	import time
	#tic = time.time()
	pt1, pt2, pt3, pt4 = condinate_rotate(all_anchors) # coodinate project
	inds_inside = np.array(ind_inside(pt1, pt2, pt3, pt4, im_info[0], im_info[1])) # inside index
	#print time.time()-tic

        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print 'anchors.shape', anchors.shape

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
	
	
	#print np.array(anchors,dtype=np.float32).shape
	#print gt_boxes[:,0:5].astype(np.float32)
	#print 'aasdf'
        overlaps = rbbx_overlaps(np.ascontiguousarray(anchors, dtype=np.float32), np.ascontiguousarray(gt_boxes[:,0:5], dtype=np.float32),cfg.GPU_ID)
	#print 'sdfwef'

	an_gt_diffs = angle_diff(anchors,gt_boxes)

        argmax_overlaps = overlaps.argmax(axis=1) # max overlaps of anchor compared with gts
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        max_overlaps_angle_diff = an_gt_diffs[np.arange(len(inds_inside)), argmax_overlaps] # D
	

        gt_argmax_overlaps = overlaps.argmax(axis=0) # max overlaps of gt compared with anchors
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where((overlaps == gt_max_overlaps)&(an_gt_diffs<=cfg.TRAIN.R_POSITIVE_ANGLE_FILTER))[0]
	#mask1 = np.abs(anchors[:, 3] * 1.0 / anchors[:, 2] - 2.0) < 1.5
	#mask2 = np.abs(anchors[:, 3] * 1.0 / anchors[:, 2] - 5.0) < 1.5
	#mask3 = np.abs(anchors[:, 3] * 1.0 / anchors[:, 2] - 8.0) < 1.5
	

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
	    #labels[(max_overlaps < 0.3) & mask1] = 0
	    #labels[(max_overlaps < 0.27) & mask2] = 0
	    #labels[(max_overlaps < 0.13) & mask3] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

	# fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1 # D
	#labels[(max_overlaps >= 0.7) & mask1] = 1
	#labels[(max_overlaps >= 0.625) & mask2] = 1
	#labels[(max_overlaps >= 0.313) & mask3] = 1
        # fg label: above threshold IOU the angle diff abs must be less than 15

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[(max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP) | ((max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP) & (max_overlaps_angle_diff > cfg.TRAIN.R_NEGATIVE_ANGLE_FILTER))] = 0
	    #labels[((max_overlaps < 0.3) | ((max_overlaps >= 0.7) & (max_overlaps_angle_diff > cfg.TRAIN.R_NEGATIVE_ANGLE_FILTER))) & mask1] = 0
	    #labels[((max_overlaps < 0.27) | ((max_overlaps >= 0.625) & (max_overlaps_angle_diff > cfg.TRAIN.R_NEGATIVE_ANGLE_FILTER))) & mask2] = 0
	    #labels[((max_overlaps < 0.13) | ((max_overlaps >= 0.313) & (max_overlaps_angle_diff > cfg.TRAIN.R_NEGATIVE_ANGLE_FILTER))) & mask3] = 0
	
	
	
	
        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))

	#print 'num_fg',len(fg_inds),'num_bg',len(bg_inds)	

        bbox_targets = np.zeros((len(inds_inside), self.bbox_para_num), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), self.bbox_para_num), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_RBBOX_INSIDE_WEIGHTS) # D

        bbox_outside_weights = np.zeros((len(inds_inside), self.bbox_para_num), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, self.bbox_para_num)) * 1.0 / num_examples
            negative_weights = np.ones((1, self.bbox_para_num)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print 'rpn: num_positive avg', self._fg_sum / self._count
            print 'rpn: num_negative avg', self._bg_sum / self._count

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * self.bbox_para_num)).transpose(0, 3, 1, 2)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * self.bbox_para_num)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * self.bbox_para_num)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 5 # 
    assert gt_rois.shape[1] == 6

    return rbbox_transform(ex_rois, gt_rois[:, :5]).astype(np.float32, copy=False)

if __name__ == "__main__":
        anchor_scales = (8, 16, 32)
        _anchors = generate_anchors(scales=np.array(anchor_scales))
        _num_anchors = _anchors.shape[0]
        _feat_stride = 16
	print _anchors

	bbox_para_num = 5
	_allowed_border = 0

        height, width = (40,30)
        
        A = _num_anchors
        # labels
        top0 = np.zeros((1, 1, A * height, width))
        # bbox_targets
        top1 = np.zeros((1, A * bbox_para_num, height, width))
        # bbox_inside_weights
        top2 = np.zeros((1, A * bbox_para_num, height, width))
        # bbox_outside_weights
        top3 = np.zeros((1, A * bbox_para_num, height, width))

	#forward
        # GT boxes (x_ctr, y_ctr, height, width, theta, label)
        gt_boxes = np.array([[250.0,250.0,256.0,256.0,16,1.0]])#,[300.0,300.0,256.0,256.0,0.0,1.0]])
        # im_info
        im_info = [640,480]

 	# 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * _feat_stride
        shift_y = np.arange(0, height) * _feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
	#print shift_x.ravel().shape,shift_y.ravel().shape, np.zeros((3,width*height)).shape
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),np.zeros((3, width*height)))).transpose()
	#print shifts
        
	# add A anchors (1, A, 5) to
        # cell K shifts (K, 1, 5) to get
        # shift anchors (K, A, 5)
        # reshape to (K*A, 5) shifted anchors
        A = _num_anchors
        K = shifts.shape[0]
        all_anchors = (_anchors.reshape((1, A, bbox_para_num)) +
                       shifts.reshape((1, K, bbox_para_num)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, bbox_para_num))
        total_anchors = int(K * A)
	
	print all_anchors.shape

        # only keep anchors inside the image
        # inds_inside = np.where(
            # (all_anchors[:, 0] >= -self._allowed_border) &
            # (all_anchors[:, 1] >= -self._allowed_border) &
            # (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            # (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        # )[0]

	pt1, pt2, pt3, pt4 = condinate_rotate(all_anchors) # coodinate project
	inds_inside = np.array(ind_inside(pt1, pt2, pt3, pt4, im_info[0], im_info[1])) # inside index
	#+++

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
	print anchors.shape

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)

	######################
	# angle filter
	######################
	
	overlaps = rbbx_overlaps(anchors,gt_boxes,cfg.GPU_ID)

	an_gt_diffs = angle_diff(anchors,gt_boxes)

        argmax_overlaps = overlaps.argmax(axis=1) # max overlaps of anchor compared with gts
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        max_overlaps_angle_diff = an_gt_diffs[np.arange(len(inds_inside)), argmax_overlaps] # D
	

        gt_argmax_overlaps = overlaps.argmax(axis=0) # max overlaps of gt compared with anchors
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where((overlaps == gt_max_overlaps)&(an_gt_diffs<=15))[0]
	

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU the angle diff abs must be less than 15
        labels[(max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP) & (max_overlaps_angle_diff <= 15)] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[(max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP) | (max_overlaps_angle_diff > 15)] = 0
	

	
	######################
	# End angle filter
	######################



	######################
	# no angle filter
	######################
	'''
        overlaps = rbbx_overlaps(anchors,gt_boxes)

        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
	    print anchors[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP]
	    print overlaps[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP]

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1
	#print anchors[gt_argmax_overlaps]
        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
	#print anchors[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP]
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
	'''    
        ######################
	# End no angle filter
	######################
	
	# subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))

	print np.sum(labels==0)
	#print anchors[np.where(labels==0)]
	print np.sum(labels==1)
	print anchors[np.where(labels==1)]

        bbox_targets = np.zeros((len(inds_inside), bbox_para_num), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
	
	print np.sort(bbox_targets[:,4])

        bbox_inside_weights = np.zeros((len(inds_inside),bbox_para_num), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_RBBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), bbox_para_num), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, bbox_para_num)) * 1.0 / num_examples
            negative_weights = np.ones((1, bbox_para_num)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

	#print labels,labels.shape,np.sum(labels==1.0)
	#print bbox_targets,bbox_targets.shape
	#print bbox_inside_weights,bbox_inside_weights.shape
	#print bbox_outside_weights,bbox_outside_weights.shape

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        top0.reshape(*labels.shape)
        top0[...] = labels
	
	print np.sum(top0==0)

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * bbox_para_num)).transpose(0, 3, 1, 2)
        top1.reshape(*bbox_targets.shape)
        top1[...] = bbox_targets
	
        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * bbox_para_num)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top2.reshape(*bbox_inside_weights.shape)
        top2[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * bbox_para_num)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top3.reshape(*bbox_outside_weights.shape)
        top3[...] = bbox_outside_weights

