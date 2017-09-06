# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from rotation.data_extractor import get_rroidb

def r_get_rotate_minibatch(roidb, num_classes):

    bbox_para_num = 5

    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # generate random number of angle of images to argument the dataset with type of anti-clockwise angle
    angles = np.array(np.random.rand(num_images) * 360, dtype = np.int16)

    #print angles

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_rprocessed_image_blob(roidb, random_scale_inds, angles)

    blobs = {'data': im_blob}

    # bbox: [ctr_x, ctr_y, height, width, angle]

    if cfg.TRAIN.HAS_RPN: # D
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"

        # gt boxes: (ctr_x, ctr_y, height, width, angle, cls)

        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), bbox_para_num + 1), dtype=np.float32) # D initial the gt_scales
	#print roidb[0]['boxes'][gt_inds, :]
	gt_boxes[:, 0:bbox_para_num] = roidb[0]['boxes'][gt_inds, :]

	if roidb[0]['rotated']: # D
	    gt_boxes[:, 0:bbox_para_num] = rotate_gt_bbox(roidb[0], angles[0], gt_inds) # D
	    
        
        gt_boxes[:, 0:bbox_para_num-1] = gt_boxes[:, 0:bbox_para_num-1] * im_scales[0] # D
        gt_boxes[:, bbox_para_num] = roidb[0]['gt_classes'][gt_inds] # D
        blobs['gt_boxes'] = gt_boxes

	#im_info[height, width, scale]
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    return blobs

def rotate_gt_bbox(origin_gt_roidb, angle, gt_inds):

    rotated_gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    
    im_height = origin_gt_roidb["height"]
    im_width = origin_gt_roidb["width"]

    origin_gt_boxes = origin_gt_roidb["boxes"][gt_inds, :]

    # anti-clockwise to clockwise arc
    cos_cita = np.cos(-np.pi / 180 * angle)
    sin_cita = np.sin(-np.pi / 180 * angle)

    #print origin_gt_boxes[:, 0:2]

    # clockwise matrix
    rotation_matrix = np.array([[cos_cita, sin_cita], [-sin_cita, cos_cita]])

    pts_ctr = origin_gt_boxes[:, 0:2]

    pts_ctr = pts_ctr - np.tile((im_width / 2, im_height / 2), (len(gt_inds), 1))

    pts_ctr = np.array(np.dot(pts_ctr, rotation_matrix), dtype = np.int16)

    pts_ctr = pts_ctr + np.tile((im_width / 2, im_height / 2), (len(gt_inds), 1))

    origin_gt_boxes[:, 0:2] = pts_ctr 
    #print origin_gt_boxes[:, 0:2]

    len_of_gt = len(origin_gt_boxes)

    # rectificate the angle in the range of [-45, 45]
    
    for idx in range(len_of_gt):
        ori_angle = origin_gt_boxes[idx, 4]
	height = origin_gt_boxes[idx, 2]
	width = origin_gt_boxes[idx, 3]

        #step 1: normalize gt (-45,135)
        if width < height:
            ori_angle += 90
            width,height = height,width

        #step 2: rotate (-45,495)
        rotated_angle = ori_angle + angle

        #step 3: normalize rotated_angle       (-45,135)
        while rotated_angle > 135:
            rotated_angle = rotated_angle - 180

	rotated_gt_boxes[idx,0] = origin_gt_boxes[idx,0]
	rotated_gt_boxes[idx,1] = origin_gt_boxes[idx,1]
	rotated_gt_boxes[idx,2] = height * cfg.TRAIN.GT_MARGIN
	rotated_gt_boxes[idx,3] = width * cfg.TRAIN.GT_MARGIN
	rotated_gt_boxes[idx,4] = rotated_angle
    
    return rotated_gt_boxes

'''



	if ori_angle + angle > 45:
		origin_gt_boxes[idx, 4] = ori_angle + angle - 90
		origin_gt_boxes[idx, 2] = width
		origin_gt_boxes[idx, 3] = height

	elif ori_angle + angle < -45:
		origin_gt_boxes[idx, 4] = ori_angle + angle + 90
		origin_gt_boxes[idx, 2] = width
		origin_gt_boxes[idx, 3] = height
	
	elif ori_angle + angle <= 45 and ori_angle + angle >= -45:
		origin_gt_boxes[idx, 4] = ori_angle + angle
    
    

    #origin_gt_boxes[:, 4] = origin_gt_boxes[:, 4] + angle

    return origin_gt_boxes
'''

def _get_rprocessed_image_blob(roidb, scale_inds, angles):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
	
	if roidb[i]['rotated']:
	    # get the size of image
	    (h, w) = im.shape[:2] 
	    # set the rotation center
	    center = (w / 2, h / 2) 
	    # get the rotation matrix no scale changes
	    scale = 1.0
	    # anti-clockwise angle in the function
	    M = cv2.getRotationMatrix2D(center, angles[i], scale)
	    im = cv2.warpAffine(im,M,(w,h)) 
 
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales



def r_get_minibatch(roidb, num_classes):

    bbox_para_num = 5

    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)



    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    # bbox: [ctr_x, ctr_y, height, width, angle]

    if cfg.TRAIN.HAS_RPN:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"

        # gt boxes: (ctr_x, ctr_y, height, width, angle, cls)

        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), bbox_para_num + 1), dtype=np.float32) # D initial the gt_scales
        gt_boxes[:, 0:bbox_para_num] = roidb[0]['boxes'][gt_inds, :]
        gt_boxes[:, 0:bbox_para_num-1] = gt_boxes[:, 0:bbox_para_num-1] * im_scales[0] # D
        gt_boxes[:, bbox_para_num] = roidb[0]['gt_classes'][gt_inds] # D
        blobs['gt_boxes'] = gt_boxes

	#im_info[height, width, scale]
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # next not use
    #####################################################################################################
    else: # not using RPN
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, bbox_para_num + 1), dtype=np.float32) # D
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, bbox_para_num * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        # all_overlaps = []
        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes)

            # Add to RoIs blob
            rois = _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
            # all_overlaps = np.hstack((all_overlaps, overlaps))

        # For debug visualizations
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob

        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = \
                np.array(bbox_inside_blob > 0).astype(np.float32)

    return blobs


def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def im_rotate(im):
    
    return im


def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):

    bbox_para_num = 5
	
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 5-of-5*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 5K blob of regression targets
        bbox_inside_weights (ndarray): N x 5K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, bbox_para_num * num_classes), dtype=np.float32) # D
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = bbox_para_num * cls  # D
        end = start + bbox_para_num # D
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.RBBOX_INSIDE_WEIGHTS # D
    return bbox_targets, bbox_inside_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()

'''
if __name__ == "__main__":

    roidb = [get_rroidb("test")[0]]
    blobs = r_get_rotate_minibatch(roidb, 2)
    
    im = blobs['data'].transpose(0, 2, 3, 1)[0]

    #print blobs   

    gt_boxes = blobs['gt_boxes']
    info = blobs['im_info']
    
    s = 1.0 / info[0,2]
    im = cv2.resize(im, None, None, fx=s, fy=s,interpolation=cv2.INTER_LINEAR)
    im += cfg.PIXEL_MEANS
    #cv2.imshow('win',im/255.0)
    #cv2.waitKey(0)

    gt_boxes[:, 0:5-1] = gt_boxes[:, 0:5-1] * s

    for i in range(gt_boxes.shape[0]):
	gt_box = gt_boxes[i]
	cx,cy,h,w,angle = gt_box[:-1]
	lt = [cx - w/2, cy - h/2,1]
	rt = [cx + w/2, cy - h/2,1]
	lb = [cx - w/2, cy + h/2,1]
	rb = [cx + w/2, cy + h/2,1]

	pts = []

	pts.append(lt)
	pts.append(rt)
	pts.append(rb)
	pts.append(lb)
	
        print angle

	if angle != 0:
		cos_cita = np.cos(-np.pi / (180 / angle))
		sin_cita = np.sin(-np.pi / (180 / angle))

	else :
		cos_cita = 1
		sin_cita = 0
	
	M0 = np.array([[1,0,0],[0,1,0],[-cx,-cy,1]])
	M1 = np.array([[cos_cita, sin_cita,0], [-sin_cita, cos_cita,0],[0,0,1]])
	M2 = np.array([[1,0,0],[0,1,0],[cx,cy,1]])
	rotation_matrix = M0.dot(M1).dot(M2)

	rotated_pts = np.dot(pts, rotation_matrix)

	rotated_pts = np.array(rotated_pts, dtype=np.int16)

	cv2.line(im, (rotated_pts[0,0],rotated_pts[0,1]), (rotated_pts[1,0],rotated_pts[1,1]), (0, 0, 255))
	cv2.line(im, (rotated_pts[1,0],rotated_pts[1,1]), (rotated_pts[2,0],rotated_pts[2,1]), (0, 0, 255))
	cv2.line(im, (rotated_pts[2,0],rotated_pts[2,1]), (rotated_pts[3,0],rotated_pts[3,1]), (0, 0, 255))
	cv2.line(im, (rotated_pts[3,0],rotated_pts[3,1]), (rotated_pts[0,0],rotated_pts[0,1]), (0, 0, 255))
cv2.imshow('win',im/255.0)
cv2.waitKey(0)
'''
