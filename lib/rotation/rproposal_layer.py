# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors

from rbbox_transform import rbbox_transform_inv
#from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
#from fast_rcnn.nms_wrapper import nms

#from rotate_cpu_nms import rotate_cpu_nms
#from rotation.rotate_cython_nms import rotate_cpu_nms
#from rotation.rotate_circle_nms import rotate_cpu_nms
#from rotation.rotate_gpu_nms import rotate_gpu_nms as rotate_cpu_nms
from rotation.rotate_polygon_nms import rotate_gpu_nms as rotate_cpu_nms

DEBUG = False

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._feat_stride = layer_params['feat_stride']
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]

	self._bbox_para_num = 5 # parameter number of bbox

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 6-tuple
        # (n, ctr_x, ctr_y, h, w, theta) specifying an image batch index n and a
        # rectangle (ctr_x, ctr_y, h, w, theta)
        top[0].reshape(1, self._bbox_para_num + 1) # D

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1, 1) # D

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :] # ??????
        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

	# ctr_x, ctr_y shift, the rest 3 colomns filled zero
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            np.zeros((3, width * height)))).transpose() 

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 5) to
        # cell K shifts (K, 1, 5) to get
        # shift anchors (K, A, 5)
        # reshape to (K*A, 5) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, self._bbox_para_num)) + shifts.reshape((1, K, self._bbox_para_num)).transpose((1, 0, 2)) # D
        anchors = anchors.reshape((K * A, self._bbox_para_num)) # D
	#print anchors

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 5 * A, H, W) format
        # transpose to (1, H, W, 5 * A)
        # reshape to (1 * H * W * A, 5) where rows are ordered by (h, w, a)
        # in slowest to fastest order

	# [ctr_x, ctr_y, height, width, angle]

        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5)) # D
	
        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations

	#print bbox_deltas[:, 4]
	#bbox_deltas[:, 4] = 0

        proposals = rbbox_transform_inv(anchors, bbox_deltas) # D

        # 2. clip predicted boxes to image
        # proposals = clip_boxes(proposals, im_info[:2]) # TODO

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2]) # D
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

	###
	#anchors = anchors[keep,:]
	#anchors = anchors[order,:]
	###
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
	import time
	tic = time.time()
        keep = rotate_cpu_nms(np.hstack((proposals, scores)), nms_thresh,cfg.GPU_ID) # D
	#print time.time() - tic
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]
	

	#anchors = anchors[keep,:]
	#for i in range(300):
	#	print anchors[i]
	#	print proposals[i]
	#	print scores[i]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

        # [Optional] output scores blob
        if len(top) > 1:
            top[1].reshape(*(scores.shape))
            top[1].data[...] = scores

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 3] # D
    hs = boxes[:, 2] # D
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


if __name__ == "__main__":
        anchor_scales = (8, 16, 32)
        _anchors = generate_anchors(scales=np.array(anchor_scales))
        _num_anchors = _anchors.shape[0]
        _feat_stride = 16
	print _anchors

	_bbox_para_num = 5

        height, width = (10,10)
        
        A = _num_anchors
        # labels
        #top0 = np.zeros((1, 6))

        # im_info
        im_info = [160,160,1]

	cfg_key = "TRAIN"

	pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

	print pre_nms_topN, post_nms_topN, nms_thresh, min_size

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want (1, A, H, W) 
        scores = np.zeros((1,_num_anchors,height,width))
	scores[0,19,5,5] = 1#87,87,128,128,0

        bbox_deltas = np.zeros((1,_num_anchors*5,height,width))
	bbox_deltas[0,19*5:20*5,5,5] = [0.1,0.1,0.1,0.1,0.1]

	print "bbox_delta", bbox_deltas.shape

        # Enumerate all shifts
        shift_x = np.arange(0, width) * _feat_stride
        shift_y = np.arange(0, height) * _feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

	# ctr_x, ctr_y shift, the rest 3 colomns filled zero
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            np.zeros((3, width * height)))).transpose()         


        A = _num_anchors
        K = shifts.shape[0]

	

        anchors = _anchors.reshape((1, A, _bbox_para_num)) + shifts.reshape((1, K, _bbox_para_num)).transpose((1, 0, 2)) # D
        anchors = anchors.reshape((K * A, _bbox_para_num)) # D

        print "A", A, "K", K

	
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5)) # D
	
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))


	

        proposals = rbbox_transform_inv(anchors, bbox_deltas)

        
        # proposals = clip_boxes(proposals, im_info[:2]) # TODO

       
        keep = _filter_boxes(proposals, min_size * im_info[2]) # D
        proposals = proposals[keep, :]
        scores = scores[keep]

       
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

      	print scores.shape
	print proposals.shape
	print 'ps',np.hstack((proposals, scores)).shape

        keep = rotate_cpu_nms(np.hstack((proposals, scores)), nms_thresh,cfg.GPU_ID) # D
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]
	print scores
	#print proposals.shape
	

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
	
	print blob.shape

        top0 = np.zeros(blob.shape)
        top0[...] = blob

       



