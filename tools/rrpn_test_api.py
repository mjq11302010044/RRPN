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
import math
from rotation.data_extractor import get_rroidb, test_rroidb
reload(cv2)

from rotation.rt_test import r_im_detect
from rotation.merge_box import merge

CLASSES = ('__background__',
           'text')

def demo(net, image_name, gt_boxes, result_dir, ori_result_dir, conf = 0.75):
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

    print "gt_margin: ", cfg.TEST.GT_MARGIN,  cfg.TRAIN.GT_MARGIN
    print "img_padding: ", cfg.IMG_PADDING

    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = conf
    NMS_THRESH = 0.3
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

	dets[:, 2] = dets[:, 2] / cfg.TEST.GT_MARGIN
	dets[:, 3] = dets[:, 3] / cfg.TEST.GT_MARGIN

	

	#if imdb_name == "icdar13":
	#write_result_ICDAR2013(im_file, dets, CONF_THRESH, ori_result_dir, im_height, im_width)
	#result_file = write_result_ICDAR2013(im_file, dets, CONF_THRESH, result_dir, im_height, im_width)
	 
            

	#if imdb_name == "icdar15":
	write_result_ICDAR(im_file, dets, CONF_THRESH, ori_result_dir, im_height, im_width)
	result_file = write_result_ICDAR(im_file, dets, CONF_THRESH, result_dir, im_height, im_width)
	
	


	#write_result(im_file, dets, CONF_THRESH, ori_result_dir, im_height, im_width)
	#result_file = write_result(im_file, dets, CONF_THRESH, result_dir, im_height, im_width)
	print "write done"
	#post_merge(result_file)
	print "merge done"
	#
	#print "merge done"
        #vis_detections(im, cls, dets, gt_boxes, thresh=CONF_THRESH)
	return result_file

def post_merge(result_file):

	f = open(result_file)
	#print result_file
	lines = f.readlines()
	bbox = []
	for line in lines:
		record = line.strip().split(' ')
		bbox.append([float(record[0]),float(record[1]),float(record[2]),float(record[3]),float(record[4]),float(record[5])])		
	
	res = np.array(bbox)

	res = merge(np.array(bbox))
	
	res = merge(np.array(res), 3.0, 1.3, 1.2, 1.0)
	res = merge(np.array(res), 3.0, 1.3, 1.2, 1.0)
	res = merge(np.array(res), 3.0, 1.3, 1.2, 1.0)	
	res = merge(np.array(res), 3.0, 1.3, 1.2, 1.0)	
	res = merge(np.array(res), 3.0, 1.3, 1.2, 1.0)	
	f.close()
	
	g = open(result_file,'w+')
	res = res.tolist()
	for r in res:
		print>>g,r[0],r[1],r[2],r[3],r[4],r[5]
	g.close()

def write_result_ICDAR(im_file, dets, threshold, result_dir, height, width):

    file_spl = im_file.split('/')
    file_name = file_spl[len(file_spl) - 1]
    file_name_arr = file_name.split(".")
	
    file_name_str = file_name_arr[0]

    result = os.path.join(result_dir, "res_" + file_name_str + ".txt")    

    return_bboxes = []

    if not os.path.isfile(result):
	os.mknod(result)
    result_file = open(result, "w")

    result_str = ""

    for idx in range(len(dets)):
	cx,cy,h,w,angle = dets[idx][0:5]
	lt = [cx - w/2, cy - h/2,1]
	rt = [cx + w/2, cy - h/2,1]
	lb = [cx - w/2, cy + h/2,1]
	rb = [cx + w/2, cy + h/2,1]


	pts = []

	#angle = angle * 0.45

	pts.append(lt)
	pts.append(rt)
	pts.append(rb)
	pts.append(lb)

	angle = -angle

	#if angle != 0:
	cos_cita = np.cos(np.pi / 180 * angle)
	sin_cita = np.sin(np.pi / 180 * angle)

	#else :
	#	cos_cita = 1
	#	sin_cita = 0

	M0 = np.array([[1,0,0],[0,1,0],[-cx,-cy,1]])
	M1 = np.array([[cos_cita, sin_cita,0], [-sin_cita, cos_cita,0],[0,0,1]])
	M2 = np.array([[1,0,0],[0,1,0],[cx,cy,1]])
	rotation_matrix = M0.dot(M1).dot(M2)

	rotated_pts = np.dot(np.array(pts), rotation_matrix)


	#print im
	#print im.shape
#			im = im.transpose(2,0,1)

	det_str = str(int(rotated_pts[0][0])) + "," + str(int(rotated_pts[0][1])) + "," + \
	 	str(int(rotated_pts[1][0])) + "," + str(int(rotated_pts[1][1])) + "," + \
	 	str(int(rotated_pts[2][0])) + "," + str(int(rotated_pts[2][1])) + "," + \
	 	str(int(rotated_pts[3][0])) + "," + str(int(rotated_pts[3][1])) + "\r\n"




	#rotated_pts = rotated_pts[:,0:2]
	
	if (dets[idx][5] > threshold):
		rotated_pts = over_bound_handle(rotated_pts, height, width)
		det_str = str(int(rotated_pts[0][0])) + "," + str(int(rotated_pts[0][1])) + "," + \
		str(int(rotated_pts[1][0])) + "," + str(int(rotated_pts[1][1])) + "," + \
	 	str(int(rotated_pts[2][0])) + "," + str(int(rotated_pts[2][1])) + "," + \
	 	str(int(rotated_pts[3][0])) + "," + str(int(rotated_pts[3][1])) + "\r\n"
	
		result_str = result_str + det_str
		return_bboxes.append(dets[idx])

	#print rotated_pts.shape

	

	
    result_file.write(result_str)
    result_file.close()

    return return_bboxes

def write_result_ICDAR2013(im_file, dets, threshold, result_dir, height, width):

    file_spl = im_file.split('/')
    file_name = file_spl[len(file_spl) - 1]
    file_name_arr = file_name.split(".")
	
    file_name_str = file_name_arr[0]

    result = os.path.join(result_dir, "res_" + file_name_str + ".txt")    

    if not os.path.isfile(result):
	os.mknod(result)
    result_file = open(result, "w")

    result_str = ""

    for idx in range(len(dets)):
	cx,cy,h,w,angle = dets[idx][0:5]
	lt = [cx - w/2, cy - h/2,1]
	rt = [cx + w/2, cy - h/2,1]
	lb = [cx - w/2, cy + h/2,1]
	rb = [cx + w/2, cy + h/2,1]


	pts = []

	#angle = angle * 0.45

	pts.append(lt)
	pts.append(rt)
	pts.append(rb)
	pts.append(lb)

	angle = -angle

	#if angle != 0:
	cos_cita = np.cos(np.pi / 180 * angle)
	sin_cita = np.sin(np.pi / 180 * angle)

	#else :
	#	cos_cita = 1
	#	sin_cita = 0

	M0 = np.array([[1,0,0],[0,1,0],[-cx,-cy,1]])
	M1 = np.array([[cos_cita, sin_cita,0], [-sin_cita, cos_cita,0],[0,0,1]])
	M2 = np.array([[1,0,0],[0,1,0],[cx,cy,1]])
	rotation_matrix = M0.dot(M1).dot(M2)

	rotated_pts = np.dot(np.array(pts), rotation_matrix)


	#print im
	#print im.shape
#			im = im.transpose(2,0,1)

	rotated_pts = over_bound_handle(rotated_pts, height, width)

	left = min(int(rotated_pts[0][0]), int(rotated_pts[1][0]), int(rotated_pts[2][0]), int(rotated_pts[3][0]))
	top = min(int(rotated_pts[0][1]), int(rotated_pts[1][1]), int(rotated_pts[2][1]), int(rotated_pts[3][1]))
	right = max(int(rotated_pts[0][0]), int(rotated_pts[1][0]), int(rotated_pts[2][0]), int(rotated_pts[3][0]))
	bottom = max(int(rotated_pts[0][1]), int(rotated_pts[1][1]), int(rotated_pts[2][1]), int(rotated_pts[3][1]))

	if (dets[idx][5] > threshold):

		det_str = str(left) + "," + \
		 	str(top) + "," + \
		 	str(right) + "," + \
		 	str(bottom) + "\r\n"

		result_str = result_str + det_str

	
    	#print dets[idx][5], threshold

	
    result_file.write(result_str)
    result_file.close()

    return result


def over_bound_handle(pts, img_height, img_width):

	pts[np.where(pts < 0)] = 1

	pts[np.where(pts[:,0] > img_width), 0] = img_width-1
	pts[np.where(pts[:,1] > img_height), 1] = img_height-1

	return pts



def rot_pts(det):
    cx,cy,h,w,angle = det[0:5]
    lt = [cx - w/2, cy - h/2,1]
    rt = [cx + w/2, cy - h/2,1]
    lb = [cx - w/2, cy + h/2,1]
    rb = [cx + w/2, cy + h/2,1]


    pts = []

    #angle = angle * 0.45

    pts.append(lt)
    pts.append(rt)
    pts.append(rb)
    pts.append(lb)

    angle = -angle

    #if angle != 0:
    cos_cita = np.cos(np.pi / 180 * angle)
    sin_cita = np.sin(np.pi / 180 * angle)

    #else :
    #	cos_cita = 1
    #	sin_cita = 0

    M0 = np.array([[1,0,0],[0,1,0],[-cx,-cy,1]])
    M1 = np.array([[cos_cita, sin_cita,0], [-sin_cita, cos_cita,0],[0,0,1]])
    M2 = np.array([[1,0,0],[0,1,0],[cx,cy,1]])
    rotation_matrix = M0.dot(M1).dot(M2)

    rotated_pts = np.dot(np.array(pts), rotation_matrix)

    lt = np.argmin(rotated_pts, axis=0)
    rb = np.argmax(rotated_pts, axis=0)
  
    left = rotated_pts[lt[0]]
    top = rotated_pts[lt[1]]
    right = rotated_pts[rb[0]]
    bottom = rotated_pts[rb[1]]

    return left, top, right, bottom
    

def write_result(im_file, dets, threshold, result_dir, im_height, im_width):
    #result_dir = "./result"
    file_spl = im_file.split('/')
    file_name = file_spl[len(file_spl) - 1]


    result = os.path.join(result_dir, file_name + ".result")    
    #print "result_name", result

    if not os.path.isfile(result):
	os.mknod(result)
    result_file = open(result, "w")
	    
    result_str = ""
    for det in dets:
	det_str = ""
	if det[5] > threshold:
	    #print det[len(det) - 1]
	    for ind in range(len(det) - 1):
	        det_str = det_str + str(det[ind]) + " "
	    det_str = det_str + str(det[len(det) - 1]) + "\n"
	    result_str = result_str + det_str

    result_file.write(result_str)
    result_file.close()

    return result

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

    print over_bound_handle(np.array(pts), 160, 150)
    '''
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    if args.demo_net == "rrpn":
	prototxt = os.path.join(cfg.RRPN_MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')

    

    print "shit prototxt",prototxt
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
  #  im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
  #  for i in xrange(2):
  #      _, _= r_im_detect(net, im)

    im_names = []#['IMG_0030.JPG','IMG_0059.JPG','IMG_0063.JPG']

    roidb = get_ICDAR2015_test("test")
    gt_boxes = []	
    
    for rdb in roidb:
	im_names.append(rdb['image'])
        gt_boxes.append(rdb['boxes'])
        

    for im_idx in range(len(im_names)):
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_names[im_idx])
	print im_names[im_idx], gt_boxes[im_idx]
        demo(net, im_names[im_idx], gt_boxes[im_idx], "./result")
	

    #plt.show()
    '''
