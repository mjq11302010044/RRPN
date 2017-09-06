
"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
#from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from rotation.rbbox_transform import rbbox_transform_inv # D
from rotation.generate_anchors import generate_anchors
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
#from fast_rcnn.nms_wrapper import nms # D 

from rotation.rotate_circle_nms import rotate_cpu_nms

import cPickle
from utils.blob import im_list_to_blob
import os

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def r_im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 5 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (5*K) array of predicted bounding boxes
    """
    # [x _ctr, y_ctr, height, width, angle]

    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))
    
    # do forward
    print blobs['data'].shape
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if False: # feature map anchor visualization
	# 800*1067
	# 800/16 = 50
	# 1067/16 = 67
	# 50*67

	# 2*54 A*S*R = 6*3*3 = [-30.0, 0.0, 30.0, 60.0, 90.0 , 120.0]*2**np.arange(3, 6) * ratios = [0.125, 0.2, 0.5]
	print 'kaishi'
	print net.blobs['data'].data.shape
	print im.shape
	cls = net.blobs['rpn_cls_prob_reshape'].data[0,54:108,:,:]
	#print select 
	#print cls.shape
	#s = cls.shape
	#a = cls.reshape(-1)
	#b = a.argsort()[::-1]
	#print 'a' ,a
	#a[b[6000:]] = -10.0
	#cls = a.reshape(s)
	#print cls.shape
#1

	amp = np.max(cls,axis=0)
	A = np.argmax(cls,axis=0)

	canvas = np.zeros((im_blob.shape[2], im_blob.shape[3], 3), dtype = np.int8)

	B = A%6
	A = -(-30+30*(A%6))/180.0*3.1415926
	amp =  (amp - np.min(amp))/ (np.max(amp)-np.min(amp))
	#canvas = cv2.resize(black_map,None,None,im_scales[0],im_scales[0])
	#canvas = cv2.cvtColor(im.copy(),cv2.COLOR_BGR2GRAY)  
	canvas = cv2.resize(im,None,None,im_scales[0],im_scales[0])
	canvas[...] = 0
	print 'im',im.shape
	print 'sh',canvas.shape
	#canvas = np.zeros((im2.shape[0],im2.shape[1],3),dtype=np.uint8)
	#canvas[:,:,0] = im2	
	#canvas[:,:,1] = im2
	#canvas[:,:,2] = im2

	stride = 16
	for i in range(0,canvas.shape[1],stride):
		for j in range(0,canvas.shape[0],stride):
			#cv2.rectangle(canvas,(i+stride/4,j+stride/4),(i+3*stride/4,j+3*stride/4),(0,255,0),-1)
			cx = i+stride/2
			cy = j+stride/2
			dx = np.cos(A[j/stride,i/stride]) * 10 * (amp[j/stride,i/stride])
			dy = np.sin(A[j/stride,i/stride]) * 10 * (amp[j/stride,i/stride])
			color = [(134,145,255),(136,215,255),(137,255,255),(151,255,184),(255,251,142),(246,195,154)]
			c =  color[int(amp[j/stride,i/stride]) * 6 - 1]
			#c = (int(255 - amp[j/stride,i/stride] * 200), 0, int(134 + amp[j/stride,i/stride] * 112))
			c = (255,255,255)
			cv2.line(canvas, (int(cx),int(cy)), (int(cx+dx),int(cy+dy)), c,2)
			

	
	cv2.imshow('win1',canvas)	
	#cv2.waitKey(0)

    if False: # feature map proposal visualization
	# 800*1067
	# 800/16 = 50
	# 1067/16 = 67
	# 50*67
	_bbox_para_num = 5

	anchor_scales = (8, 16, 32)
        _anchors = generate_anchors(scales=np.array(anchor_scales))

	_num_anchors = _anchors.shape[0]
	_feat_stride = 16

	bbox_deltas = net.blobs['rpn_bbox_pred'].data
	bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5)) # D

	scores = net.blobs['rpn_cls_prob_reshape']
	
	height = scores.shape[2]
	width = scores.shape[3]
	
	cls = scores.data[0,54:108,:,:]	

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

        #print "A", A, "K", K
        proposals = rbbox_transform_inv(anchors, bbox_deltas)

	# H * W * S * R * A
	
	proposals = proposals.reshape((1, height, width, 3 * 3 * 6, 5))
	anchors = anchors.reshape((1, height, width, 3 * 3 * 6, 5))
	#anchors = anchors.reshape((height * width * 3 * 3, 6, 5))
	#print anchors[:, 4, 4]
	#print proposals[:, 4, 4]

	#canvas = np.zeros((im_blob.shape[2], im_blob.shape[3], 3), dtype = np.int8)

	canvas = cv2.resize(im,None,None,im_scales[0],im_scales[0])
	canvas[...] = 0

	amp = np.max(cls,axis=0)
	angle_offs = np.argmax(cls,axis=0)

	print amp.shape

	for i in range(height):
		for j in range(width):
			#print "x", anchors[0][i][j][angle_offs[i,j]][0]
			#print "y", anchors[0][i][j][angle_offs[i,j]][1]
			#print "A", proposals[0][i][j][angle_offs[i,j]][4]
			
			cx = anchors[0][i][j][angle_offs[i,j]][0]
			cy = anchors[0][i][j][angle_offs[i,j]][1]
			an = proposals[0][i][j][angle_offs[i,j]][4]

			dx = 0
			dy = 0

			#if amp[i,j] > 0.6:
			dx = np.cos(an / 180 * 3.1415) * 10 * (amp[i,j])
			dy = -np.sin(an / 180 * 3.1415) * 10 * (amp[i,j])

			c = (255,255,255)
			cv2.line(canvas, (int(cx),int(cy)), (int(cx+dx),int(cy+dy)), c,2)

	cv2.imshow('win2',canvas)	
	cv2.waitKey(0)

    #if True:
	#print net.blobs['conv5_3'].data.shape

    # [0, x _ctr, y_ctr, height, width, angle]

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:6] / im_scales[0] # D
	boxes[:,4]*=im_scales[0] # D

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']

	#### Must Remove
	#print box_deltas[:,4]
	#box_deltas[:,4] = 0
	####

	

        pred_boxes = rbbox_transform_inv(boxes, box_deltas) # D 
        # pred_boxes = clip_boxes(pred_boxes, im.shape) # D
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    #print 'conv5',np.sum(net.blobs['conv5'].data)
    return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, box_proposals)
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*5:(j+1)*5]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                vis_detections(im, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)
