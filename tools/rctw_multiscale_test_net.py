#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
from rotation.data_extractor import get_ICDAR2013_test, test_ICDAR2015,get_ICDAR_RCTW17_10Fold
import rrpn_test_api
import caffe
import argparse
import pprint
import time, os, sys
import os.path as osp
import cv2
import numpy as np
from rotation.rotate_polygon_nms import rotate_gpu_nms

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    modelHome = args.caffemodel
    outputHome = osp.abspath(osp.join(cfg.ROOT_DIR, 'output', cfg.EXP_DIR,args.imdb_name+'result'))
    print outputHome
    if not os.path.exists(outputHome):
	    os.makedirs(outputHome)		
    roidb = get_ICDAR_RCTW17_10Fold("test")
    im_names = []
    gt_boxes = []
    output_dir = outputHome
    for rdb in roidb:
	im_names.append(rdb['image'])
	gt_boxes.append([0,0,0,0,0])
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    
    net = caffe.Net(args.prototxt, modelHome+'/vgg16_faster_rcnn_iter_450000.caffemodel', caffe.TEST)
    for im_idx in range(len(im_names)):	
    	im = cv2.imread(im_names[im_idx])
	im_height =im.shape[0]
	im_width = im.shape[1]
	dets = rrpn_test_api.multiscale_demo(net, im, gt_boxes[im_idx], args.imdb_name, output_dir, output_dir,0,0,im_width,im_height,0.5,args.gpu_id)
	#lt
	_dets = rrpn_test_api.multiscale_demo(net, im[0:im_height/2,0:im_width/2,:], gt_boxes[im_idx], args.imdb_name, output_dir, output_dir,0,0,im_width,im_height,0.5,args.gpu_id)
	dets = np.vstack((dets,_dets))
	#bt	
	_dets = rrpn_test_api.multiscale_demo(net, im[im_height/2:,0:im_width/2,:], gt_boxes[im_idx], args.imdb_name, output_dir, output_dir,0,0,im_width,im_height,0.5,args.gpu_id)
	_dets[:,1]+=im_height/2
	dets = np.vstack((dets,_dets))
	#rt
	_dets = rrpn_test_api.multiscale_demo(net, im[0:im_height/2,im_width/2:,:], gt_boxes[im_idx], args.imdb_name, output_dir, output_dir,0,0,im_width,im_height,0.5,args.gpu_id)
	_dets[:,0]+=im_width/2
	dets = np.vstack((dets,_dets))
	#rb
	_dets = rrpn_test_api.multiscale_demo(net, im[im_height/2:,im_width/2:,:], gt_boxes[im_idx], args.imdb_name, output_dir, output_dir,0,0,im_width,im_height,0.5,args.gpu_id)
	_dets[:,0]+=im_width/2
	_dets[:,1]+=im_height/2
	dets = np.vstack((dets,_dets))
	#mt
	_dets = rrpn_test_api.multiscale_demo(net, im[0:im_height/2,im_width/4:3*im_width/4,:], gt_boxes[im_idx], args.imdb_name, output_dir, output_dir,0,0,im_width,im_height,0.5,args.gpu_id)
	_dets[:,0]+=im_width/4
	dets = np.vstack((dets,_dets))
	#mb
	_dets = rrpn_test_api.multiscale_demo(net, im[im_height/2:,im_width/4:3*im_width/4,:], gt_boxes[im_idx], args.imdb_name, output_dir, output_dir,0,0,im_width,im_height,0.5,args.gpu_id)
	_dets[:,0]+=im_width/4
	_dets[:,1]+=im_height/2
	dets = np.vstack((dets,_dets))
	#ml
	_dets = rrpn_test_api.multiscale_demo(net, im[im_height/4:3*im_height/4,0:im_width/2,:], gt_boxes[im_idx], args.imdb_name, output_dir, output_dir,0,0,im_width,im_height,0.5,args.gpu_id)
	_dets[:,1]+=im_height/4
	dets = np.vstack((dets,_dets))
	#mr
	_dets = rrpn_test_api.multiscale_demo(net, im[im_height/4:3*im_height/4,im_width/2:,:], gt_boxes[im_idx], args.imdb_name, output_dir, output_dir,0,0,im_width,im_height,0.5,args.gpu_id)
	_dets[:,0]+=im_width/2
	_dets[:,1]+=im_height/4
	dets = np.vstack((dets,_dets))
	#mm
	_dets = rrpn_test_api.multiscale_demo(net, im[im_height/4:3*im_height/4,im_width/4:3*im_width/4,:], gt_boxes[im_idx], args.imdb_name, output_dir, output_dir,0,0,im_width,im_height,0.5,args.gpu_id)
	_dets[:,0]+=im_width/4
	_dets[:,1]+=im_height/4
	dets = np.vstack((dets,_dets))
	keep = rotate_gpu_nms(dets, 0.3,args.gpu_id) # D
        dets = dets[keep, :]
	#dets = dets[0:20]
	#dets[:, 4] = dets[:, 4] * 0.45

	dets[:, 2] = dets[:, 2] / cfg.TEST.GT_MARGIN
	dets[:, 3] = dets[:, 3] / cfg.TEST.GT_MARGIN

	rrpn_test_api.vis_detections(im, 'text', dets, thresh=0.5)
	cv2.waitKey(0)
	#rrpn_test_api.write_result_RCTW(im_names[im_idx], dets, output_dir, im_height, im_width)
	print 

	#rrpn_test_api.multiscale_demo(net, im, gt_boxes[im_idx], args.imdb_name, output_dir, output_dir,0,0,im_width,im_height,0.5,args.gpu_id)
	#rrpn_test_api.multiscale_demo(net, im, gt_boxes[im_idx], args.imdb_name, output_dir, output_dir,0,0,im_width,im_height,0.5,args.gpu_id)
	#rrpn_test_api.multiscale_demo(net, im, gt_boxes[im_idx], args.imdb_name, output_dir, output_dir,0,0,im_width,im_height,0.5,args.gpu_id)


