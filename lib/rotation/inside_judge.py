import os
import numpy as np
import time
from fast_rcnn.config import cfg

def condinate_rotate(all_anchors):

	

	left_top = np.array((- all_anchors[:, 2] / 2, - all_anchors[:, 3] / 2)).T # left top
	left_bottom = np.array([- all_anchors[:, 2] / 2, all_anchors[:, 3] / 2]).T # left bottom
	right_top = np.array([all_anchors[:, 2] / 2, - all_anchors[:, 3] / 2]).T # right top
	right_bottom = np.array([all_anchors[:, 2] / 2, all_anchors[:, 3] / 2]).T # right bottom
	
	
	
	theta = all_anchors[:, 4]

	#positive angle when anti-clockwise rotation

	cos_theta = np.cos(np.pi / 180 * theta) # D
	sin_theta = np.sin(np.pi / 180 * theta) # D

	# [2, 2, n] n is the number of anchors
	rotation_matrix = [cos_theta, sin_theta, -sin_theta, cos_theta]


	# coodinate rotation
	
	

	return pts_dot(left_top, rotation_matrix) + np.array((all_anchors[:, 0], all_anchors[:, 1])).T, \
		pts_dot(left_bottom, rotation_matrix) + np.array((all_anchors[:, 0], all_anchors[:, 1])).T, \
		pts_dot(right_top, rotation_matrix) + np.array((all_anchors[:, 0], all_anchors[:, 1])).T, \
		pts_dot(right_bottom, rotation_matrix) + np.array((all_anchors[:, 0], all_anchors[:, 1])).T

def pts_dot(pts, rotat_matrix):
	
	return np.array([pts[:, 0] * rotat_matrix[0] + pts[:, 1] * rotat_matrix[2], pts[:, 0] * rotat_matrix[1] + pts[:, 1] * rotat_matrix[3]]).T
	

	
def ind_inside(pt1, pt2, pt3, pt4, img_width, img_height):

	size = len(pt1)
	#print size
	
	#tic = time.time()
	#inside_ind = []
	
	padding_w = cfg.IMG_PADDING * img_width
	padding_h = cfg.IMG_PADDING * img_height
        iw = img_width+padding_w
        ih = img_height+padding_h
	
	#print type(pt1),pt1.shape
	pt = np.hstack((pt1,pt2,pt3,pt4))
        tmp = (pt[:,0:8:2]>-padding_w) & (pt[:,1:8:2]>-padding_h) & (pt[:,0:8:2]<iw) & (pt[:,1:8:2]<ih)
        ins = np.where(tmp[:,0]&tmp[:,1]&tmp[:,2]&tmp[:,3])[0].tolist()
        #print ins


	#for ind in range(size):
	#	
	#	t_pt1 = pt1[ind]
	#	t_pt2 = pt2[ind]
	#	t_pt3 = pt3[ind]
	#	t_pt4 = pt4[ind]
#
#		if t_pt1[0] > - padding_w and t_pt1[0] < iw and \
#		t_pt2[0] > - padding_w and t_pt2[0] < iw and \
		#t_pt3[0] > - padding_w and t_pt3[0] < iw and \
		#t_pt4[0] > - padding_w and t_pt4[0] < iw and \
		#t_pt1[1] > - padding_h and t_pt1[1] < ih and \
		#t_pt2[1] > - padding_h and t_pt2[1] < ih and \
		#t_pt3[1] > - padding_h and t_pt3[1] < ih and \
		#t_pt4[1] > - padding_h and t_pt4[1] < ih:
		#	inside_ind.append(ind)
        #print np.sum(np.array(inside_ind)-np.array(ins))
	#print time.time() - tic

	return ins

import time 

if __name__ == "__main__":
	tic = time.time()
	query_boxes = np.array([
			#[0, 0, 100, 100, 0], # outside
			#[20, 20, 100, 100, 45.0], # outside
			#[200, 200, 100, 100, 0], # outside
			#[195, 195, 100, 50, 45.0], # outside
			#[100, 100, 100, 100, 0], # outside
			[100, 100, 50, 50, -10] # inside
			])
	

	pt1, pt2, pt3, pt4 = condinate_rotate(query_boxes)
	print pt1
	print pt2 
	print pt3
	print pt4
	t = ind_inside(pt1, pt2, pt3, pt4, 200, 200)

	print time.time() - tic
	print t


