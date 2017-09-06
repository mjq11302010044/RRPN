#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
#from fast_rcnn.nms_wrapper import nms
from rotation.rotate_polygon_nms import rotate_gpu_nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

from rotation.data_extractor import get_rroidb, test_rroidb, get_ICDAR2015_test
reload(cv2)

from rotation.rt_test import r_im_detect
from rotation.merge_box import merge

CLASSES = ('__background__',
           'text')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
	'rrpn': ('VGG16',
                  '813/exp0/vgg16_faster_rcnn_iter_195000.caffemodel')}
		


def vis_detections(im, class_name, dets, output_name, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    #print "proposal", dets
    for i in inds:
        bbox = dets[i, :5]
        score = dets[i, -1]
	
	cx,cy,h,w,angle = bbox[0:5]
	lt = [cx - w/2, cy - h/2,1]
	rt = [cx + w/2, cy - h/2,1]
	lb = [cx - w/2, cy + h/2,1]
	rb = [cx + w/2, cy + h/2,1]


	pts = []



	pts.append(lt)
	pts.append(rt)
	pts.append(rb)
	pts.append(lb)

	angle = -angle


	cos_cita = np.cos(np.pi / 180 * angle)
	sin_cita = np.sin(np.pi / 180 * angle)

	
	M0 = np.array([[1,0,0],[0,1,0],[-cx,-cy,1]])
	M1 = np.array([[cos_cita, sin_cita,0], [-sin_cita, cos_cita,0],[0,0,1]])
	M2 = np.array([[1,0,0],[0,1,0],[cx,cy,1]])
	rotation_matrix = M0.dot(M1).dot(M2)

	rotated_pts = np.dot(np.array(pts), rotation_matrix)



	
	cv2.line(im, (int(rotated_pts[0,0]),int(rotated_pts[0,1])), (int(rotated_pts[1,0]),int(rotated_pts[1,1])), (0, 0, 255),3)
	cv2.line(im, (int(rotated_pts[1,0]),int(rotated_pts[1,1])), (int(rotated_pts[2,0]),int(rotated_pts[2,1])), (0, 0, 255),3)
	cv2.line(im, (int(rotated_pts[2,0]),int(rotated_pts[2,1])), (int(rotated_pts[3,0]),int(rotated_pts[3,1])), (0, 0, 255),3)
	cv2.line(im, (int(rotated_pts[3,0]),int(rotated_pts[3,1])), (int(rotated_pts[0,0]),int(rotated_pts[0,1])), (0, 0, 255),3)

    im = cv2.resize(im, (int(im.shape[1] * 0.7), int(im.shape[0] * 0.7)))
    cv2.imwrite(output_name, im)
    #cv2.imshow("win", im)
    #cv2.waitKey(0)
    return output_name
 
def demo(net, image_name, output_name):#, result_dir, ori_result_dir):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = image_name
    im = cv2.imread(im_file)

    im_height = im.shape[0]
    im_width = im.shape[1]

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = r_im_detect(net, im)


    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.9
    NMS_THRESH = 0.3

    file_ret = ""

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 5*cls_ind:5*(cls_ind + 1)] # D
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = rotate_gpu_nms(dets, NMS_THRESH) # D
        dets = dets[keep, :]
	#dets = dets[0:20]
	#dets[:, 4] = dets[:, 4] * 0.45

	dets[:, 2] = dets[:, 2] / 1.4
	dets[:, 3] = dets[:, 3] / 1.4


    	file_ret = vis_detections(im, cls, dets, output_name, thresh=CONF_THRESH)

    return file_ret

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    pts = [[100,-5],[200,100],[200,200],[-5,100]]


