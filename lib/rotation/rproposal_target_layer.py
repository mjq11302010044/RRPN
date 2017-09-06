# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
#from fast_rcnn.bbox_transform import bbox_transform
#from utils.cython_bbox import bbox_overlaps
from rotation.rbbox import angle_diff
from rotation.rbbox_overlaps import rbbx_overlaps
from rbbox_transform import rbbox_transform

DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']
	
	#self._num_classes = 11

	self._bbox_para_num = 5

        # sampled rois (0, x_ctr, y_ctr, height, width, theta)
        top[0].reshape(1, self._bbox_para_num + 1)
        # labels
        top[1].reshape(1, 1)
        # bbox_targets
        top[2].reshape(1, self._num_classes * self._bbox_para_num) # D
        # bbox_inside_weights
        top[3].reshape(1, self._num_classes * self._bbox_para_num) # D
        # bbox_outside_weights
        top[4].reshape(1, self._num_classes * self._bbox_para_num) # D
        

    def forward(self, bottom, top):
        # Proposal ROIs (0, x_ctr, y_ctr, height, width, theta) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x_ctr, y_ctr, height, width, theta, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
	
	# all_rois shape: (proposal + gt, 6)
	# pattern: (0, x_ctr, y_ctr, height, width, theta)


        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        # sampled rois
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # bbox_targets
        top[2].reshape(*bbox_targets.shape)
        top[2].data[...] = bbox_targets

        # bbox_inside_weights
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = bbox_inside_weights

        # bbox_outside_weights
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)   
	

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th, da)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 5K blob of regression targets
        bbox_inside_weights (ndarray): N x 5K blob of loss weights
    """
    _bbox_para_num = 5

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, _bbox_para_num * num_classes), dtype=np.float32) # D	
    #print num_classes
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(_bbox_para_num * cls) # D
        end = start + _bbox_para_num # D
	#print start,end        
	bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.RBBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    _bbox_para_num = 5

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == _bbox_para_num # D
    assert gt_rois.shape[1] == _bbox_para_num # D

    targets = rbbox_transform(ex_rois, gt_rois) # D
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.RBBOX_NORMALIZE_MEANS)) # D
                / np.array(cfg.TRAIN.RBBOX_NORMALIZE_STDS)) # D
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """

    _bbox_para_num = 5

    # overlaps: (rois x gt_boxes)
    overlaps = rbbx_overlaps( # D
        np.ascontiguousarray(all_rois[:, 1: _bbox_para_num + 1], dtype=np.float32), # D
        np.ascontiguousarray(gt_boxes[:, :_bbox_para_num], dtype=np.float32) ,cfg.GPU_ID) # D

    an_gt_diffs = angle_diff(all_rois[:, 1: _bbox_para_num + 1],gt_boxes[:, :_bbox_para_num]) # D

    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)

    max_overlaps_angle_diff = an_gt_diffs[np.arange(len(gt_assignment)), gt_assignment] # D

    labels = gt_boxes[gt_assignment, 5] # D: label is in the last column

    # Select foreground RoIs as those with >= FG_THRESH overlap

    #################### angle filter
  #  print np.shape(max_overlaps_angle_diff)
    fg_inds = np.where((max_overlaps >= cfg.TRAIN.FG_THRESH) & (max_overlaps_angle_diff <= cfg.TRAIN.R_POSITIVE_ANGLE_FILTER))[0] # D
    ####################
  #  print 'anglediff',max_overlaps_angle_diff[fg_inds]
   # print gt_boxes

    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        # print(type(fg_inds))
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)

    ####################
    bg_inds = np.where(((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO)) | ((max_overlaps>=cfg.TRAIN.FG_THRESH)&(max_overlaps_angle_diff>cfg.TRAIN.R_NEGATIVE_ANGLE_FILTER)))[0]
    ####################
    # print 'proposal fg',len(fg_inds),'bg',len(bg_inds)
    # print 

    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:_bbox_para_num + 1], gt_boxes[gt_assignment[keep_inds], :_bbox_para_num], labels) # D

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights


if __name__ == "__main__":
        _num_classes = 3	
	_bbox_para_num = 5


 # Proposal ROIs (0, x_ctr, y_ctr, height, width, theta) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = np.array([[0,100.0,100.0,100.0,100.0,-20.0], # discard
				[0,150,150,100,100,0.0], # discard
				[0,100.0,100.0,100.0,100.0,20.0]])# keep
        # GT boxes (x_ctr, y_ctr, height, width, theta, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = np.array([[100.0,100.0,100.0,100.0,0.0,1],[100.0,100.0,100.0,100.0,30.0,2]])

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

	# all_rois shape: (proposal + gt, 6)
	# pattern: (0, x_ctr, y_ctr, height, width, theta)


        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

	##################
	# no angle filter
	##################
	'''
	#Sample ROI Test
	# overlaps: (rois x gt_boxes)
	overlaps = rbbx_overlaps( # D
	all_rois[:, 1: _bbox_para_num + 1],# D
	gt_boxes[:, :_bbox_para_num] ) # D

	print overlaps
	
	gt_assignment = overlaps.argmax(axis=1)
	max_overlaps = overlaps.max(axis=1)

	print max_overlaps

	print 'gt_boxes',gt_boxes.shape
        print 'gt_ass',gt_assignment.shape

	labels = gt_boxes[gt_assignment, 5] # D: label is in the last column
	print "label", labels.shape
	# Select foreground RoIs as those with >= FG_THRESH overlap


	fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
	
	# Guard against the case when an image has fewer than fg_rois_per_image
	# foreground RoIs
	'''
	##################
	# End no angle filter
	##################

	##################
	# angle filter
	##################
	
	# overlaps: (rois x gt_boxes)
    	overlaps = rbbx_overlaps( # D
        	all_rois[:, 1: _bbox_para_num + 1],# D
        	gt_boxes[:, :_bbox_para_num] ,cfg.GPU_ID) # D

    	an_gt_diffs = angle_diff(all_rois[:, 1: _bbox_para_num + 1],gt_boxes[:, :_bbox_para_num]) # D

    	gt_assignment = overlaps.argmax(axis=1)
   	max_overlaps = overlaps.max(axis=1)

	print "gt_assignment"
	print gt_assignment

    	max_overlaps_angle_diff = an_gt_diffs[np.arange(len(gt_assignment)), gt_assignment] # D
    
    	labels = gt_boxes[gt_assignment, 5] # D: label is in the last column

    	# Select foreground RoIs as those with >= FG_THRESH overlap

    	#################### angle filter
    	fg_inds = np.where((max_overlaps >= cfg.TRAIN.FG_THRESH) & (max_overlaps_angle_diff <= 15))[0] # D
	print "max_overlaps_angle_diff"
	print max_overlaps_angle_diff
	print "an_gt_diffs"
	print an_gt_diffs
	print fg_inds
	print "all_rois[fg_inds]", all_rois[fg_inds]
    	####################
	
	##################
	# End angle filter
	##################
	
	fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
	# Sample foreground regions without replacement
	if fg_inds.size > 0:
		fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

	# Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
	bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
		       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
	
	# Compute number of background RoIs to take from this image (guarding
	# against there being fewer than desired)
	bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
	bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
	# Sample background regions without replacement
	if bg_inds.size > 0:
		bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

	# The indices that we're selecting (both fg and bg)
	keep_inds = np.append(fg_inds, bg_inds)
	# Select sampled values from various arrays:
	labels = labels[keep_inds]

	# Clamp labels for the background RoIs to 0
	labels[fg_rois_per_this_image:] = 0
	rois = all_rois[keep_inds]
	#print all_rois
	print "rois", rois
	#print "label", labels.shape

	bbox_target_data = _compute_targets(
	rois[:, 1:_bbox_para_num + 1], gt_boxes[gt_assignment[keep_inds], :_bbox_para_num], labels) # D

	#print bbox_target_data
	#print

	bbox_targets, bbox_inside_weights = \
	_get_bbox_regression_labels(bbox_target_data, _num_classes)

	#print bbox_targets
	#print
	#print bbox_inside_weights	
