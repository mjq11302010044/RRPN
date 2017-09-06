#encoding:utf-8 
import numpy as np 
import cv2 

from rotation.data_extractor import get_rroidb
from rotation.r_roidb import add_rbbox_regression_targets
from rotation.r_minibatch import r_get_minibatch
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from rotation.rt_train import filter_roidb


cfg_from_file('../../experiments/cfgs/faster_rcnn_end2end.yml')
for c in cfg:
	print c,':',cfg[c]


roidb = get_rroidb()
print 'len',len(roidb)
#print roidb[0]
roidb = filter_roidb(roidb)
print 'len_after',len(roidb)
print roidb[0]
bbox_means,bbox_stds = add_rbbox_regression_targets(roidb)#To test
#print roidb[0]
#print 'bbox_means',bbox_means
#print 'bbox_stds',bbox_stds
blobs = r_get_minibatch(roidb, 2)

im = blobs['data'].transpose(0, 2, 3, 1)[0]
gt_boxes = blobs['gt_boxes']
info = blobs['im_info']

#print 'im:',im
#print 'gt',gt_boxes
#print 'info',info

print im.shape
s = 1.0/ info[0,2]
im = cv2.resize(im, None, None, fx=s, fy=s,interpolation=cv2.INTER_LINEAR)
print im.shape
im += cfg.PIXEL_MEANS
#cv2.imshow('win',im/255.0)
#cv2.waitKey(0)

print 'gt',gt_boxes
gt_boxes[:, 0:5-1] = gt_boxes[:, 0:5-1] * s
print 'gt',gt_boxes

print gt_boxes.shape

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
	
	if angle != 0:
		cos_cita = np.cos(np.pi / (180 / angle))
		sin_cita = np.sin(np.pi / (180 / angle))

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
pts = [(200, 250), (200, 400), (300, 400), (300, 250)]

cita = 45
scale = 1.25
shift = (20, 50)



add_rbbox_regression_targets(roidb)


	
image = cv2.imread("dog.jpg") 

(h,w) = image.shape[:2] 

bbox_w = pts[2][0] - pts[0][0]
bbox_h = pts[2][1] - pts[0][1]

bbox_ctr = (bbox_w / 2 + pts[0][0], bbox_h / 2 + pts[0][1])

relative_pts = []

for ind in pts:
	relative_pts.append(((ind[0] - bbox_ctr[0]), (ind[1] - bbox_ctr[1])))

print relative_pts

print bbox_ctr

if cita != 0:
	cos_cita = np.cos(np.pi / (180 / cita))
	sin_cita = np.sin(np.pi / (180 / cita))

else :
	cos_cita = 1
	sin_cita = 0

rel_pts_matrix = np.array(relative_pts)

scale_matrix = np.array([[scale, 0], [0, scale]])

rotation_matrix = np.array([[cos_cita, sin_cita], [-sin_cita, cos_cita]])

#matrix computation

bbox_ctr = (bbox_ctr[0] + shift[0], bbox_ctr[1] + shift[1])

rel_pts_matrix = np.dot(rel_pts_matrix, scale_matrix)
rotated_pts = np.dot(rel_pts_matrix, rotation_matrix)

rotated_pts = np.array(rotated_pts, dtype=np.int16)

print rotated_pts

abs_pts = []

for ind in rotated_pts:
	abs_pts.append((ind[0] + bbox_ctr[0], ind[1] + bbox_ctr[1]))

print abs_pts

#center = (h / 2, w / 2) 

cv2.line(image, abs_pts[0], abs_pts[1], (255, 0, 0))
cv2.line(image, abs_pts[1], abs_pts[2], (255, 0, 0))
cv2.line(image, abs_pts[2], abs_pts[3], (255, 0, 0))
cv2.line(image, abs_pts[3], abs_pts[0], (255, 0, 0))

cv2.line(image, pts[0], pts[1], (0, 0, 255))
cv2.line(image, pts[1], pts[2], (0, 0, 255))
cv2.line(image, pts[2], pts[3], (0, 0, 255))
cv2.line(image, pts[3], pts[0], (0, 0, 255))

cv2.imshow("Original",image) 
cv2.waitKey(0) 

#旋转45度，缩放0.75 

#M = cv2.getRotationMatrix2D(center, cita, scale)

#旋转缩放矩阵：(旋转中心，旋转角度，缩放因子) 

#rotated = cv2.warpAffine(image,M,(w,h)) 

#cv2.imshow("Rotated by 45 Degrees",rotated) 

#cv2.waitKey(0) 

#旋转-45度，缩放1.25 

#M = cv2.getRotationMatrix2D(center,-45,1.25)

#旋转缩放矩阵：(旋转中心，旋转角度，缩放因子) 

#rotated = cv2.warpAffine(image,M,(w,h)) 
#cv2.imshow("Rotated by -90 Degrees",rotated) 

#cv2.waitKey(0)
'''
