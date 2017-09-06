import numpy as np
import cv2
import time
import math
def rotate_cpu_nms(dets, threshold):
	'''
	Parameters
	----------------
	dets: (N, 6) --- x_ctr, y_ctr, height, width, angle, score
	threshold: 0.7 or 0.5 IoU
	----------------
	Returns
	---------------- 
	keep: keep the remaining index of dets
	'''
	keep = []
	scores = dets[:, -1]
	
	tic = time.time()
 
	order = scores.argsort()[::-1]
	ndets = dets.shape[0]
	print "nms start"
	print ndets
	suppressed = np.zeros((ndets), dtype = np.int)
	
	

	for _i in range(ndets):
		i = order[_i]
		if suppressed[i] == 1:
			continue
		keep.append(i)
		r1 = ((dets[i,0],dets[i,1]),(dets[i,3],dets[i,2]),dets[i,4])
		area_r1 = dets[i,2]*dets[i,3]
		for _j in range(_i+1,ndets):
			#tic = time.time()
			j = order[_j]
			if suppressed[j] == 1:
				continue
			r2 = ((dets[j,0],dets[j,1]),(dets[j,3],dets[j,2]),dets[j,4])
			area_r2 = dets[j,2]*dets[j,3]
			ovr = 0.0	
			#+++
			#d = math.sqrt((dets[i,0] - dets[j,0])**2 + (dets[i,1] - dets[j,1])**2)
			#d1 = math.sqrt(dets[i,2]**2 + dets[i,3]**2)
			#d2 = math.sqrt(dets[j,2]**2 + dets[j,3]**2)
			#if d<d1+d2:
				#+++
						
			int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
			if  None != int_pts:
				
				order_pts = cv2.convexHull(int_pts, returnPoints = True)
				#t2 = time.time()
    				int_area = cv2.contourArea(order_pts)				
				#t3 = time.time()
				ovr = int_area*1.0/(area_r1+area_r2-int_area)
				
			if ovr>=threshold:
				suppressed[j]=1
			#print t1 - tic, t2 - t1, t3 - t2
			#print 
	print time.time() - tic
	print "nms done"
	return keep



if __name__ == "__main__":

	boxes = np.array([
			[50, 50, 100, 100, 0,0.99],
			[60, 60, 100, 100, 0,0.88],#keep 0.68
			[50, 50, 100, 100, 45.0,0.66],#discard 0.70
			[200, 200, 100, 100, 0,0.77],#keep 0.0
			
		])

	#boxes = np.tile(boxes, (4500 / 4, 1))

	#for ind in range(4500):
	#	boxes[ind, 5] = 0

	a = rotate_cpu_nms(boxes, 0.7)

	print boxes[a]
