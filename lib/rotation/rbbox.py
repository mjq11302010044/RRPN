import numpy as np 
import cv2
from rotation.rbbox_overlaps import rbbx_overlaps as abc

from rotation.generate_anchors import generate_anchors

def rbbx_overlaps(boxes, query_boxes):

	'''
	Parameters
	----------------
	boxes: (N, 5) --- x_ctr, y_ctr, height, width, angle
	query: (K, 5) --- x_ctr, y_ctr, height, width, angle
	----------------
	Returns
	---------------- 
	Overlaps (N, K) IoU
	'''
	
	N = boxes.shape[0]
	K = query_boxes.shape[0]
	overlaps = np.zeros((N, K), dtype = np.float32)
	
	for k in range(K):

		query_area = query_boxes[k, 2] * query_boxes[k, 3]

		for n in range(N):
			
			box_area = boxes[n, 2] * boxes[n, 3]
			#IoU of rotated rectangle
			#loading data anti to clock-wise
			rn = ((boxes[n, 0], boxes[n, 1]), (boxes[n, 3], boxes[n, 2]), -boxes[n, 4])
			rk = ((query_boxes[k, 0], query_boxes[k, 1]), (query_boxes[k, 3], query_boxes[k, 2]), -query_boxes[k, 4])
			int_pts = cv2.rotatedRectangleIntersection(rk, rn)[1]
			#print type(int_pts)			
			if  None != int_pts:
        			order_pts = cv2.convexHull(int_pts, returnPoints = True)
    				int_area = cv2.contourArea(order_pts)
				
				overlaps[n, k] = int_area * 1.0 / (query_area + box_area - int_area)
	return overlaps


def angle_diff(boxes, query_boxes):
	
	
	'''
	Parameters
	----------------
	boxes: (N, 5) --- x_ctr, y_ctr, height, width, angle
	query: (K, 5) --- x_ctr, y_ctr, height, width, angle
	----------------
	Returns
	---------------- 
	diff (N, K) angles
	'''
	N = boxes.shape[0]
	K = query_boxes.shape[0]

	#angle_diff = np.zeros((N, K), dtype = np.float32)

	angles_pro = boxes[:, 4].reshape(N,1)
	angles_gt = query_boxes[:, 4].reshape(1,K)

	ret = np.abs(angles_pro - angles_gt)
	
	change = np.where(ret>150)
	ret[change] = np.abs(180 - ret[change])	
			
	return ret

if __name__ == "__main__":

	query_boxes = np.array([
			[1151.86, 537.293, 1244.822, 1436.03, 1],
			[1151.86, 637.293, 1234.822, 1446.03, 1],
			[450.0, 450.0, 100.0,150.0 , 2]
		], dtype = np.float32)



	boxes = np.array([
			[60.0, 60.0, 100.0,  100.0, 0.0], # 4 pts
			[50.0, 50.0, 100.0, 100.0, 45.0], # 8 pts
			[80.0, 50.0, 100.0, 100.0, 0.0], # overlap 4 edges
			[50.0, 50.0, 200.0, 50.0, 45.0], # 6 edges
			[200.0, 200.0, 100.0, 100.0, 0], # no intersection
			[60.0, 60.0, 100.0,  100.0, 0.0], # 4 pts
			[50.0, 50.0, 100.0, 100.0, 45.0], # 8 pts
			], dtype = np.float32)
	#boxes = np.tile(boxes,(10000,1));

	#boxes = generate_anchors()

	print boxes

	#for i in range(1000000):
	print abc(np.ascontiguousarray(query_boxes, dtype=np.float32), np.ascontiguousarray(query_boxes, dtype=np.float32))
	
