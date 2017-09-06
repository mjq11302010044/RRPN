import numpy as np
import cv2
def convert_region(image_path , region):

	angle = region[4];
	a_cos = np.cos(angle/180.0*3.1415926535);
	a_sin = -np.sin(angle/180.0*3.1415926535);# anti clock-wise

	ctr_x = region[0];
	ctr_y = region[1];
	h = region[2];
	w = region[3];

	pts = []

	pts_x = [];
	pts_y = [];

	pts_x.append(- w / 2);
	pts_x.append(- w / 2);
	pts_x.append(w / 2);
	pts_x.append(w / 2);

	pts_y.append(- h / 2);
	pts_y.append(h / 2);
	pts_y.append(h / 2);
	pts_y.append(- h / 2);

	for i in range(4) :
		#pts[2 * i] = a_cos * pts_x[i] - a_sin * pts_y[i] + ctr_x;
		pts.append(int(a_cos * pts_x[i] - a_sin * pts_y[i] + ctr_x))
		#pts[2 * i + 1] = a_sin * pts_x[i] + a_cos * pts_y[i] + ctr_y;
		pts.append(int(a_sin * pts_x[i] + a_cos * pts_y[i] + ctr_y))

	img = cv2.imread(image_path)

	cv2.line(img, (pts[0],pts[1]), (pts[2], pts[3]), (0, 0, 255),3)
	cv2.line(img, (pts[2],pts[3]), (pts[4], pts[5]), (0, 0, 255),3)
	cv2.line(img, (pts[4],pts[5]), (pts[6], pts[7]), (0, 0, 255),3)
	cv2.line(img, (pts[6],pts[7]), (pts[0], pts[1]), (0, 0, 255),3)
	
	img = cv2.resize(img, (1024, 768))
	cv2.imshow("image", img)
	cv2.waitKey(0)

def vis_image(image_path, boxes):

	img = cv2.imread(image_path)
	#cv2.namedWindow("image")	
	#cv2.setMouseCallback("image", trigger)
	for idx in range(len(boxes)):
		cx,cy,h,w,angle = boxes[idx]
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

		cv2.line(img, (int(rotated_pts[0,0]),int(rotated_pts[0,1])), (int(rotated_pts[1,0]),int(rotated_pts[1,1])), (0, 0, 255),3)
		cv2.line(img, (int(rotated_pts[1,0]),int(rotated_pts[1,1])), (int(rotated_pts[2,0]),int(rotated_pts[2,1])), (0, 0, 255),3)
		cv2.line(img, (int(rotated_pts[2,0]),int(rotated_pts[2,1])), (int(rotated_pts[3,0]),int(rotated_pts[3,1])), (0, 0, 255),3)
		cv2.line(img, (int(rotated_pts[3,0]),int(rotated_pts[3,1])), (int(rotated_pts[0,0]),int(rotated_pts[0,1])), (0, 0, 255),3)

	img = cv2.resize(img, (1024, 768))
	cv2.imshow("image", img)
	cv2.waitKey(0)


if __name__ == "__main__":
	convert_region("/home/shiki-alice/Downloads/MSRA-TD500/test/IMG_0059.JPG", [300, 675, 1232, 112, -0.053651 / 3.1415926 * 180])
	vis_image("/home/shiki-alice/Downloads/MSRA-TD500/test/IMG_0059.JPG", [[300, 675, 1232, 112, -0.053651 / 3.1415926 * 180]])
