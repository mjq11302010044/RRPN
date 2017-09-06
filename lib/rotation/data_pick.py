import numpy as np 
import os 
import cv2
from xml.dom.minidom import parse
import xml.dom.minidom
import data_extractor

trigger_int = 0

def get_rroidb(mode):	

	L_DATASET = []

	L_MSRA = get_MSRA("train")
	print "MSRA:", len(L_MSRA)
	L_HUST = get_HUST("train")
	print "HUST:", len(L_HUST)
	L_ICDAR2013 = get_ICDAR2013("train")
	print "L_ICDAR2013:", len(L_ICDAR2013)
	L_SVT_TRAIN = get_SVT("train")
	print "L_SVT_TRAIN:", len(L_SVT_TRAIN)
	L_SVT_TEST = get_SVT("test")
	print "L_SVT_TEST:", len(L_SVT_TEST)
	L_ICDAR2003_TRAIN = get_ICDAR2003("train")
	print "L_ICDAR2003_TRAIN:", len(L_ICDAR2003_TRAIN)
	L_ICDAR2003_TEST = get_ICDAR2003("test")
	print "L_ICDAR2003_TEST:", len(L_ICDAR2003_TEST)

	L_DATASET.extend(L_MSRA)
	L_DATASET.extend(L_HUST)
	L_DATASET.extend(L_ICDAR2013)
	L_DATASET.extend(L_SVT_TRAIN)
	L_DATASET.extend(L_SVT_TEST)
	L_DATASET.extend(L_ICDAR2003_TRAIN)
	L_DATASET.extend(L_ICDAR2003_TEST)

	return L_DATASET


def test_rroidb(mode):
	return get_MSRA(mode)



def pick_training_set(dataset_name, roidb, rel_db):
	
	global trigger_int

	pick_dir = "./line_pick/"

	PR_file = pick_dir + dataset_name

	if not os.path.isdir(pick_dir):
		os.mkdir(pick_dir)
	if not os.path.isfile(PR_file):
		os.mknod(PR_file)

	PR_obj = open(PR_file, "a+")
	PR_obj.write("[")
	PR_obj.flush()
	for i in range(len(roidb)):
		image_path = roidb[i]["image"]
		boxes =  roidb[i]["boxes"]

		vis_image(image_path, boxes)
		#print rel_db[i]["image"]
		#print roidb[i]["image"]
		#print trigger_int
		#print "reldb: ", str(rel_db[i])
		
		if trigger_int == 1:
			print "pick", rel_db[i]["image"]
			PR_obj.write(str(rel_db[i]) + ",")
			PR_obj.flush()
		else:
			print "discard"
	PR_obj.write("]")
	PR_obj.flush()
    	PR_obj.close()

def show_result_ICDAR(image_path):
	
	result_dir = "/home/shiki-alice/workspace/Rotation-Faster-RCNN/py-faster-rcnn/result/1210/exp0/vgg16_faster_rcnn_iter_270000.caffemodel/test_origin/"
	
	str_l = image_path.split("/")
	
	image_name = str_l[len(str_l) - 1]

	image_name_l = image_name.split(".")

	image_name = image_name_l[0]

	vis_image_ICDAR(image_path, read_box_ICDAR(result_dir + "res_" + image_name + ".txt"))
	

def show_result_ICDAR13(image_path):
	
	result_dir = "/home/shiki-alice/workspace/Rotation-Faster-RCNN/py-faster-rcnn/result/1210/exp0/vgg16_faster_rcnn_iter_270000.caffemodel_ICDAR2013/test_origin/"
	
	str_l = image_path.split("/")
	
	image_name = str_l[len(str_l) - 1]

	image_name_l = image_name.split(".")

	image_name = image_name_l[0]

	vis_image_ICDAR13(image_path, read_box_ICDAR(result_dir + "res_" + image_name + ".txt"))

def show_result(image_path):
	
	result_dir = "/home/shiki-alice/workspace/Rotation-Faster-RCNN/py-faster-rcnn/result/17-126/exp0/vgg16_faster_rcnn_iter_15000.caffemodel/test/"
	
	str_l = image_path.split("/")
	
	image_name = str_l[len(str_l) - 1]

	vis_image(image_path, read_box(result_dir + image_name + ".result"))

def show_result_ori(image_path):
	
	result_dir = "/home/shiki-alice/workspace/Rotation-Faster-RCNN/py-faster-rcnn/result/17-126/exp0/vgg16_faster_rcnn_iter_15000.caffemodel/test_origin/"
	
	str_l = image_path.split("/")
	
	image_name = str_l[len(str_l) - 1]

	vis_image(image_path, read_box(result_dir + image_name + ".result"))

def read_box(result_file):

	result_obj = open(result_file, "r")
	result_str = result_obj.read()
	results_lines = result_str.split("\n")
	
	#precision = 0
	#recall = 0
	result_boxes = []
	for result_line in results_lines:
		
		results = result_line.split(' ')
		result_len = len(results)
		result_box = []
		if result_len > 1:
			for ind in range(result_len - 2):
				result_box.append(int(float(results[ind])))
			result_box.append(float(results[result_len - 2]))
			result_boxes.append(result_box)
	return result_boxes

def read_box_ICDAR(result_file):

	result_obj = open(result_file, "r")
	result_str = result_obj.read()
	results_lines = result_str.split("\n")
	
	#precision = 0
	#recall = 0
	result_boxes = []
	print results_lines
	for result_line in results_lines:
		
		results = result_line.split(',')
		result_len = len(results)
		result_box = []
		if result_len > 1:
			for ind in range(result_len):
				result_box.append(int(float(results[ind])))
			result_boxes.append(result_box)
	return result_boxes


def vis_image(image_path, boxes):

	img = cv2.imread(image_path)
	cv2.namedWindow("image")	
	#cv2.setMouseCallback("image", trigger)
	for idx in range(len(boxes)):
		cx,cy,h,w,angle, _ = boxes[idx]
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

	# cv2.imwrite("a.jpg", img)

def vis_image_ICDAR(image_path, boxes):

	img = cv2.imread(image_path)
	cv2.namedWindow("image")	
	
	for idx in range(len(boxes)):
		

		cv2.line(img, (int(boxes[idx][0]),int(boxes[idx][1])), (int(boxes[idx][2]),int(boxes[idx][3])), (0, 255, 255),5)
		cv2.line(img, (int(boxes[idx][2]),int(boxes[idx][3])), (int(boxes[idx][4]),int(boxes[idx][5])), (0, 255, 255),5)
		cv2.line(img, (int(boxes[idx][4]),int(boxes[idx][5])), (int(boxes[idx][6]),int(boxes[idx][7])), (0, 255, 255),5)
		cv2.line(img, (int(boxes[idx][6]),int(boxes[idx][7])), (int(boxes[idx][0]),int(boxes[idx][1])), (0, 255, 255),5)

	print boxes

	img = cv2.resize(img, (1024, 768))
	cv2.imshow("image", img)
	cv2.waitKey(0)

def vis_image_ICDAR13(image_path, boxes):

	img = cv2.imread(image_path)
	cv2.namedWindow("image")	
	#cv2.setMouseCallback("image", trigger)
	for idx in range(len(boxes)):
		

		cv2.line(img, (int(boxes[idx][0]),int(boxes[idx][1])), (int(boxes[idx][2]),int(boxes[idx][1])), (0, 255, 255),3)
		cv2.line(img, (int(boxes[idx][2]),int(boxes[idx][1])), (int(boxes[idx][2]),int(boxes[idx][3])), (0, 255, 255),3)
		cv2.line(img, (int(boxes[idx][2]),int(boxes[idx][3])), (int(boxes[idx][0]),int(boxes[idx][3])), (0, 255, 255),3)
		cv2.line(img, (int(boxes[idx][0]),int(boxes[idx][3])), (int(boxes[idx][0]),int(boxes[idx][1])), (0, 255, 255),3)

	print boxes

	img = cv2.resize(img, (1024, 768))
	cv2.imshow("image", img)
	cv2.waitKey(0)

def trigger(event, x, y, flags, param):
	global trigger_int
	
	if event == cv2.EVENT_LBUTTONDOWN:
		trigger_int = 1
	elif event == cv2.EVENT_RBUTTONDOWN:
		trigger_int = 0

def get_KAIST(mode):
	DATASET_DIR = "/var/www/html/data/KAIST/English/"
	
	im_infos = []

	

	for f_eng in os.listdir(DATASET_DIR):
		if os.path.isdir(DATASET_DIR + f_eng):
			for fsituation in os.listdir(DATASET_DIR + f_eng):
				if os.path.isdir(DATASET_DIR + f_eng + "/" + fsituation):
					for gt_file in os.listdir(DATASET_DIR + f_eng + "/" + fsituation):
						gt_split = gt_file.split(".")
						if gt_split[len(gt_split) - 1] == "xml":
							gt_tree = xml.dom.minidom.parse(DATASET_DIR + f_eng + "/" + fsituation + "/" + gt_file)
							gt_collection = gt_tree.documentElement
							
							image_file = gt_split[0] + ".JPG"
							gt_image = gt_collection.getElementsByTagName("image")[0]
							print DATASET_DIR + f_eng + "/" + fsituation + "/" + image_file
							
							resolution = gt_image.getElementsByTagName("resolution")[0]
		
							img_width = int(resolution.getAttribute("x"))
							img_height = int(resolution.getAttribute("y"))
							#print img_width, img_height
							box_nodes = gt_image.getElementsByTagName("words")[0].getElementsByTagName("word")
							
							boxes = []

							for box_node in box_nodes:
								height = int(box_node.getAttribute("height"))
								width = int(box_node.getAttribute("width"))
								x_ctr = int(box_node.getAttribute("x")) + width / 2
								y_ctr = int(box_node.getAttribute("y")) + height / 2
								angle = 0 # no angle
								
								boxes.append([x_ctr, y_ctr, height, width, angle])
							#print boxes

							len_of_bboxes = len(boxes)
							gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)	
							gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
							overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32) #text or non-text
							seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)
		
							############################################
							"""
							img = cv2.imread(DATASET_DIR + f_eng + "/" + fsituation + "/" + image_file)
	
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

							cv2.imshow("win", img)
							cv2.waitKey(0)
							"""
							###############################################

							for idx in range(len(boxes)):
									gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
									gt_classes[idx] = 1 # cls_text
									overlaps[idx, 1] = 1.0 # cls_text
									seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

							max_overlaps = overlaps.max(axis=1)
							# gt class that had the max overlap
							max_classes = overlaps.argmax(axis=1)

							im_info = {
								'gt_classes': gt_classes,
								'max_classes': max_classes,
								'image': DATASET_DIR + f_eng + "/" + fsituation + "/" + image_file,
								'boxes': gt_boxes,
								'flipped' : False,
								'gt_overlaps' : overlaps,
								'seg_areas' : seg_areas,
								'height': img_height,
								'width': img_width,
								'max_overlaps' : max_overlaps,
								'rotated': True
								}
							im_infos.append(im_info)
	return im_infos

def get_ICDAR2003(mode):

	DATASET_DIR = "/var/www/html/data/ICDAR2003/ICDAR03/"
	RELATIVE_DIR = "../../../ICDAR2003/ICDAR03/"
	if mode == "train":
		DATASET_DIR = DATASET_DIR + "SceneTrialTrain/"
	elif mode == "test":
		DATASET_DIR = DATASET_DIR + "SceneTrialTest/"

	gt_xml_file = "locations.xml"

	gt_tree = xml.dom.minidom.parse(DATASET_DIR + gt_xml_file)

	gt_collection = gt_tree.documentElement

	im_infos = []
	relative_infos = []
	gt_images = gt_collection.getElementsByTagName("image")

	#print len(gt_images)

	for gt_image in gt_images:
		image_file = gt_image.getElementsByTagName("imageName")[0].childNodes[0].data		
		
		resolution = gt_image.getElementsByTagName("resolution")[0]
		
		img_width = int(resolution.getAttribute("x"))
		img_height = int(resolution.getAttribute("y"))

		box_nodes = gt_image.getElementsByTagName("taggedRectangles")[0].getElementsByTagName("taggedRectangle")

		boxes = []		

		for box_node in box_nodes:
			width = int(float(box_node.getAttribute("width")))
			height = int(float(box_node.getAttribute("height")))
			x_ctr = int(float(box_node.getAttribute("x")) + width / 2)
			y_ctr = int(float(box_node.getAttribute("y")) + height / 2)
			angle = float(box_node.getAttribute("rotation")) # rectangle
			boxes.append([x_ctr, y_ctr, height, width, angle])
		
		len_of_bboxes = len(boxes)
		gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)	
		gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
		overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32) #text or non-text
		seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)
		
		############################################
		"""
		img = cv2.imread(DATASET_DIR + image_file)
		
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
		cv2.imshow("win", img)
		cv2.waitKey(0)
		"""
		###############################################

		for idx in range(len(boxes)):
			gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
			gt_classes[idx] = 1 # cls_text
			overlaps[idx, 1] = 1.0 # cls_text
			seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

		max_overlaps = overlaps.max(axis=1)
		# gt class that had the max overlap
		max_classes = overlaps.argmax(axis=1)

		im_info = {
			'gt_classes': gt_classes,
			'max_classes': max_classes,
			'image': DATASET_DIR + image_file,
			'boxes': gt_boxes,
			'flipped' : False,
			'gt_overlaps' : overlaps,
			'seg_areas' : seg_areas,
			'height': img_height,
			'width': img_width,
			'max_overlaps' : max_overlaps,
			'rotated': True
			}
		im_infos.append(im_info)

		relative_info = {
			'gt_classes': gt_classes,
			'max_classes': max_classes,
			'image': RELATIVE_DIR + image_file,
			'boxes': gt_boxes,
			'flipped' : False,
			'gt_overlaps' : overlaps,
			'seg_areas' : seg_areas,
			'height': img_height,
			'width': img_width,
			'max_overlaps' : max_overlaps,
			'rotated': True
			}
		relative_infos.append(relative_info)

	return im_infos, relative_infos

def get_NEOCR(mode): # discard
	DATASET_DIR = "/var/www/html/data/NEOCR_SCENE_TEXT/neocr_dataset/"
		       
	gt_dir = "Annotations/users/pixtract/dataset/"
	          
	img_dir = "Images/users/pixtract/dataset/"
	
	
	gt_file_list = []
	
	
	for gt_file in os.listdir(DATASET_DIR + gt_dir):
		
		gt_tree = xml.dom.minidom.parse(DATASET_DIR + gt_dir + gt_file)
		
		annotation = gt_tree.documentElement
		im_infos = []
		
		image_file = annotation.getElementsByTagName("filename")[0].childNodes[0].data	

		image_properties = annotation.getElementsByTagName("properties")[0]

		img_height = int(image_properties.getElementsByTagName("height")[0].childNodes[0].data)
		img_width = int(image_properties.getElementsByTagName("width")[0].childNodes[0].data)

		

		object_list = annotation.getElementsByTagName("object")

		boxes = []

		for obj in object_list:
			rects = []
			polygon = obj.getElementsByTagName("polygon")[0]
			pts = polygon.getElementsByTagName("pt")
			for pt in pts:
				rects.append(int(pt.getElementsByTagName("x")[0].childNodes[0].data))
				rects.append(int(pt.getElementsByTagName("y")[0].childNodes[0].data))
			cx = (rects[0]+rects[2]+rects[4]+rects[6])/4
			cy = (rects[1]+rects[3]+rects[5]+rects[7])/4
			h = rects[7] - rects[1]
			w = rects[2] - rects[0]
			angle = 0
			
			boxes.append([cx, cy, h, w, angle])
		
		len_of_bboxes = len(boxes)
		gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)	
		gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
		overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32) #text or non-text
		seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)
		
		############################################
		
		img = cv2.imread(DATASET_DIR + img_dir + image_file)
		
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

		cv2.imshow("win", img)
		cv2.waitKey(0)
		
		###############################################

		

def get_SVT(mode):
	DATASET_DIR = "/var/www/html/data/SVT_SCENE_TEXT/svt/svt1/"
	RELATIVE_DIR = "../../../SVT_SCENE_TEXT/svt/svt1/"
	gt_xml_file = DATASET_DIR + mode + ".xml"

	gt_tree = xml.dom.minidom.parse(gt_xml_file)

	gt_collection = gt_tree.documentElement

	im_infos = []
	relative_infos = []
	gt_images = gt_collection.getElementsByTagName("image")

	for gt_image in gt_images:
		
		image_file = gt_image.getElementsByTagName("imageName")[0].childNodes[0].data		
		
		resolution = gt_image.getElementsByTagName("Resolution")[0]
		
		img_width = int(resolution.getAttribute("x"))
		img_height = int(resolution.getAttribute("y"))

		box_nodes = gt_image.getElementsByTagName("taggedRectangles")[0].getElementsByTagName("taggedRectangle")

		boxes = []		

		for box_node in box_nodes:
			width = int(box_node.getAttribute("width"))
			height = int(box_node.getAttribute("height"))
			x_ctr = int(box_node.getAttribute("x")) + width / 2
			y_ctr = int(box_node.getAttribute("y")) + height / 2
			angle = 0 # rectangle
			boxes.append([x_ctr, y_ctr, height, width, angle])

		len_of_bboxes = len(boxes)
		gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)	
		gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
		overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32) #text or non-text
		seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)
		
		############################################
		'''
		img = cv2.imread(DATASET_DIR + image_file)
		
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

		print image_file
		img = cv2.resize(img, (1024, 768))
		cv2.imshow("win", img)
		cv2.waitKey(0)
		'''
		###############################################

		for idx in range(len(boxes)):
			gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
			gt_classes[idx] = 1 # cls_text
			overlaps[idx, 1] = 1.0 # cls_text
			seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

		max_overlaps = overlaps.max(axis=1)
		# gt class that had the max overlap
		max_classes = overlaps.argmax(axis=1)
		
		im_info = {
			'gt_classes': gt_classes,
			'max_classes': max_classes,
			'image': os.path.join(DATASET_DIR, image_file),
			'boxes': gt_boxes,
			'flipped' : False,
			'gt_overlaps' : overlaps,
			'seg_areas' : seg_areas,
			'height': img_height,
			'width': img_width,
			'max_overlaps' : max_overlaps,
			'rotated': True
			}
		im_infos.append(im_info)

		relative_info = {
			'gt_classes': gt_classes,
			'max_classes': max_classes,
			'image': os.path.join(RELATIVE_DIR, image_file),
			'boxes': gt_boxes,
			'flipped' : False,
			'gt_overlaps' : overlaps,
			'seg_areas' : seg_areas,
			'height': img_height,
			'width': img_width,
			'max_overlaps' : max_overlaps,
			'rotated': True
			}
		relative_infos.append(relative_info)

		
	return im_infos, relative_infos

def get_ICDAR2013(mode):
	DATASET_DIR = "/home/shiki-alice/Downloads/ICDAR2013"

	RELATIVE_DIR = "../../../ICDAR2013"

	img_dir = "/Challenge2_Training_Task12_Images/"
	gt_dir = "/Challenge2_Training_Task1_GT/"


	#gt_list = []
	#img_list = []

	im_infos = []
	relative_infos = []

	gt_file_list = os.listdir(DATASET_DIR + gt_dir)
	
	
	for gt_file in gt_file_list:
		gt_fobj = open(DATASET_DIR + gt_dir + gt_file)
		gt_lines =  gt_fobj.read().split("\n")
	
		img_idx = (gt_file.split(".")[0]).split("_")[1]	
		img = cv2.imread(DATASET_DIR + img_dir + img_idx + ".jpg")

		boxes = []

		for gt_line in gt_lines:
			gt_idx = gt_line.split(" ")
			if len(gt_idx) > 1:
				width = int(gt_idx[2]) - int(gt_idx[0])
				height = int(gt_idx[3]) - int(gt_idx[1])
				x_ctr = int(gt_idx[0]) + width / 2 
				y_ctr = int(gt_idx[1]) + height / 2 
				angle = 0 # no orientation
				
				boxes.append([x_ctr, y_ctr, height, width, angle])

		len_of_bboxes = len(boxes)
		gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)	
		gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
		overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32) #text or non-text
		seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

		#if len_of_bboxes == 0:
 			#print gt_file
		
		############################################
		'''
		
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
		print img_idx + ".jpg"
		img = cv2.resize(img, (1024, 768))
		cv2.imshow("win", img)
		cv2.waitKey(0)
		'''
		###############################################

		for idx in range(len(boxes)):
			gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
			gt_classes[idx] = 1 # cls_text
			overlaps[idx, 1] = 1.0 # cls_text
			seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

		max_overlaps = overlaps.max(axis=1)
		# gt class that had the max overlap
		max_classes = overlaps.argmax(axis=1)

		im_info = {
			'gt_classes': gt_classes,
			'max_classes': max_classes,
			'image': DATASET_DIR + img_dir + img_idx + ".jpg",
			'boxes': gt_boxes,
			'flipped' : False,
			'gt_overlaps' : overlaps,
			'seg_areas' : seg_areas,
			'height': img.shape[0],
			'width': img.shape[1],
			'max_overlaps' : max_overlaps,
			'rotated': True
			}
		im_infos.append(im_info)

		relative_info = {
			'gt_classes': gt_classes,
			'max_classes': max_classes,
			'image': RELATIVE_DIR + img_dir + img_idx + ".jpg",
			'boxes': gt_boxes,
			'flipped' : False,
			'gt_overlaps' : overlaps,
			'seg_areas' : seg_areas,
			'height': img.shape[0],
			'width': img.shape[1],
			'max_overlaps' : max_overlaps,
			'rotated': True
			}
		relative_infos.append(relative_info)

	return im_infos, relative_infos

def get_HUST(mode):
	DATASET_DIR = "/home/shiki-alice/Downloads/HUST/HUST-TR400/"

	gt_list = []
	img_list = []

	im_infos = []

	for file_name in os.listdir(DATASET_DIR):
		if os.path.isfile(os.path.join(DATASET_DIR, file_name)):
			file_spl = file_name.split(".")
			if file_spl[len(file_spl) - 1] == "jpg":
				img_list.append(file_name)
				gt_list.append(file_spl[0] + ".gt")


	for i in range(len(img_list)):
		#print os.path.join(DATASET_DIR, gt_list[i])
		gt_file = open(os.path.join(DATASET_DIR, gt_list[i]))
		gt_content = gt_file.read()
		gt_li =  gt_content.split('\n')
		
		img = cv2.imread(os.path.join(DATASET_DIR, img_list[i]))	

		boxes = []

		for gt_line in gt_li:
			gts = gt_line.split(' ')

			if len(gts) > 1:
				width = int(gts[4])
				height = int(gts[5])
				x_ctr = int(gts[2]) + width / 2
				y_ctr = int(gts[3]) + height / 2
				angle = -float(gts[6]) * 180.0 / np.pi # arc to anti clock-wise angle
				boxes.append([x_ctr, y_ctr, height, width, angle]) # !!!!!!!!!!!!

		 		#print angle

		len_of_bboxes = len(boxes)
		gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)	
		gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
		overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32) #text or non-text
		seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

		#if len_of_bboxes == 0:
 			#print gt_file

		############################################
		'''
		
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

		print img_list[i]
		img = cv2.resize(img, (1024, 768))
		cv2.imshow("win", img)
		cv2.waitKey(0)
		'''
		###############################################

		for idx in range(len(boxes)):
			gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
			gt_classes[idx] = 1 # cls_text
			overlaps[idx, 1] = 1.0 # cls_text
			seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

		max_overlaps = overlaps.max(axis=1)
		# gt class that had the max overlap
		max_classes = overlaps.argmax(axis=1)

		im_info = {
			'gt_classes': gt_classes,
			'max_classes': max_classes,
			'image': os.path.join(DATASET_DIR, img_list[i]),
			'boxes': gt_boxes,
			'flipped' : False,
			'gt_overlaps' : overlaps,
			'seg_areas' : seg_areas,
			'height': img.shape[0],
			'width': img.shape[1],
			'max_overlaps' : max_overlaps,
			'rotated': True
			}
		im_infos.append(im_info)
	return im_infos

def get_MSRA(mode):

	DATASET_DIR = "/home/shiki-alice/Downloads/MSRA-TD500/" + mode + "/"

	gt_list = []
	img_list = []

	for file_name in os.listdir(DATASET_DIR):
		if os.path.isfile(os.path.join(DATASET_DIR, file_name)):
			file_spl = file_name.split(".")
			if file_spl[len(file_spl) - 1] == "JPG":
				img_list.append(file_name)
				gt_list.append(file_spl[0] + ".gt")

	im_infos = []
	
	#print gt_list
	for i in range(len(img_list)):
		#print os.path.join(DATASET_DIR, gt_list[i])
		gt_file = open(os.path.join(DATASET_DIR, gt_list[i]))
		gt_content = gt_file.read()
		gt_li =  gt_content.split('\n')
		hard_boxes = []
		easy_boxes = []
	
		img = cv2.imread(os.path.join(DATASET_DIR, img_list[i]))
	
		for gt_line in gt_li:

			gts = gt_line.split(' ')
			if len(gts) > 1:
				if int(gts[1]) == 1:
					width = int(gts[4])
					height = int(gts[5])
					x_ctr = int(gts[2]) + width / 2
					y_ctr = int(gts[3]) + height / 2
					angle = -float(gts[6]) * 180.0 / np.pi # anti clock-wise angle
					hard_boxes.append([x_ctr, y_ctr, height, width, angle])
				
				elif int(gts[1]) == 0: 
					width = int(gts[4])
					height = int(gts[5])
					x_ctr = int(gts[2]) + width / 2
					y_ctr = int(gts[3]) + height / 2
					angle = -float(gts[6]) * 180.0 / np.pi # anti clock-wise angle
					easy_boxes.append([x_ctr, y_ctr, height, width, angle])

	

		len_of_bboxes = len(hard_boxes) + len(easy_boxes)

		boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)	
		gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
		overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32) #text or non-text
		seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

		for idx in range(len(hard_boxes)):
			boxes[idx, :] = [hard_boxes[idx][0], hard_boxes[idx][1], hard_boxes[idx][2], hard_boxes[idx][3], hard_boxes[idx][4]]
			gt_classes[idx] = 1 # cls_text
			overlaps[idx, 1] = 1.0 # cls_text
			seg_areas[idx] = (hard_boxes[idx][2]) * (hard_boxes[idx][3])
		for idx in range(len(easy_boxes)):	
			boxes[idx + len(hard_boxes), :] = [easy_boxes[idx][0], easy_boxes[idx][1], easy_boxes[idx][2], easy_boxes[idx][3], easy_boxes[idx][4]]
			gt_classes[idx + len(hard_boxes)] = 1 # cls_text
			overlaps[idx + len(hard_boxes), 1] = 1.0 # cls_text
			seg_areas[idx + len(hard_boxes)] = (easy_boxes[idx][2]) * (easy_boxes[idx][3])
		
		max_overlaps = overlaps.max(axis=1)
		# gt class that had the max overlap
		max_classes = overlaps.argmax(axis=1)
	
		im_info = {
			'gt_classes': gt_classes,
			'max_classes': max_classes,
			'image': os.path.join(DATASET_DIR, img_list[i]),
			'boxes': boxes,
			'flipped' : False,
			'gt_overlaps' : overlaps,
			'seg_areas' : seg_areas,
			'height': img.shape[0],
			'width': img.shape[1],
			'max_overlaps' : max_overlaps,
			'rotated': True
			}

	
		im_infos.append(im_info)

	return im_infos

def get_ICDAR2015_new(mode):
	dir_path = "/home/shiki-alice/Downloads/ICDAR-2015-COMPETITION/new_training_data/"
	img_file_type = "png"

	RELATIVE_DIR = "../../../ICDAR-2015-COMPETITION/new_training_data/"

	cls = "gt"


	file_list = os.listdir(dir_path)

	gt_list = []
	img_list = []		

	

	for file_name in file_list:
		split = file_name.split(".")
		if split[len(split) - 1] == img_file_type:
			img_list.append(file_name)

	

	for img_ind in range(len(img_list)):
		split = img_list[img_ind].split(".")	
		gt_list.append(split[0] + "." + split[1] + "." + cls)

	im_infos = []
	relative_infos = []

	for idx in range(len(img_list)):

		img_name = dir_path + img_list[idx]
		gt_name = dir_path + gt_list[idx]

		boxes = []

		print gt_name
		gt_obj = open(gt_name, 'r')

		gt_txt = gt_obj.read()

		gt_split = gt_txt.split('\n')

		img = cv2.imread(img_name)

	

		for gt_line in gt_split:
			gt_ind = gt_line.split('\t')
			if len(gt_ind) > 3:
				condinate_list = gt_ind[2].split(' ')
				pt1 = (float(condinate_list[0]), float(condinate_list[1]))
				pt2 = (float(condinate_list[2]), float(condinate_list[3]))
				pt3 = (float(condinate_list[4]), float(condinate_list[5]))
				pt4 = (float(condinate_list[6]), float(condinate_list[7]))
			
				edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
				edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

				angle = 0
			
				if edge1 > edge2:
				
					width = edge1
					height = edge2
					if pt1[0] - pt2[0] != 0:
						angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
					else:
						angle = 90.0
				elif edge2 >= edge1:
					width = edge2
					height = edge1
					#print pt2[0], pt3[0]
					if pt2[0] - pt3[0] != 0:
						angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
					else:
						angle = 90.0
				if angle < -45.0:
					angle = angle + 180

				x_ctr = float(pt1[0] + pt3[0]) / 2#pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
				y_ctr = float(pt1[1] + pt3[1]) / 2#pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

				boxes.append([x_ctr, y_ctr, height, width, angle])
			
		len_of_bboxes = len(boxes)
		gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)	
		gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
		overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32) #text or non-text
		seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)
		
		for idx in range(len(boxes)):
			gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
			gt_classes[idx] = 1 # cls_text
			overlaps[idx, 1] = 1.0 # cls_text
			seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])


		max_overlaps = overlaps.max(axis=1)
		# gt class that had the max overlap
		max_classes = overlaps.argmax(axis=1)

		im_info = {
			'gt_classes': gt_classes,
			'max_classes': max_classes,
			'image': img_name,
			'boxes': gt_boxes,
			'flipped' : False,
			'gt_overlaps' : overlaps,
			'seg_areas' : seg_areas,
			'height': img.shape[0],
			'width': img.shape[1],
			'max_overlaps' : max_overlaps,
			'rotated': True
			}
		im_infos.append(im_info)

		relative_info = {
			'gt_classes': gt_classes,
			'max_classes': max_classes,
			'image': RELATIVE_DIR + img_list[idx],
			'boxes': gt_boxes,
			'flipped' : False,
			'gt_overlaps' : overlaps,
			'seg_areas' : seg_areas,
			'height': img.shape[0],
			'width': img.shape[1],
			'max_overlaps' : max_overlaps,
			'rotated': True
			}
		relative_infos.append(relative_info)
	return im_infos, relative_infos

def get_ICDAR2015(mode):
	dir_path = "/home/shiki-alice/Downloads/ICDAR-2015-COMPETITION/training_data/"
	img_file_type = "png"

	RELATIVE_DIR = "../../../ICDAR-2015-COMPETITION/training_data/"

	cls = "gt"

	file_list = os.listdir(dir_path)

	gt_list = []
	img_list = []		

	

	for file_name in file_list:
		split = file_name.split(".")
		if split[len(split) - 1] == img_file_type:
			img_list.append(file_name)

	

	for img_ind in range(len(img_list)):
		split = img_list[img_ind].split(".")	
		gt_list.append(split[0] + "." + split[1] + "." + cls)

	im_infos = []
	relative_infos = []

	for idx in range(len(img_list)):

		img_name = dir_path + img_list[idx]
		gt_name = dir_path + gt_list[idx]

		boxes = []

		print gt_name
		gt_obj = open(gt_name, 'r')

		gt_txt = gt_obj.read()

		gt_split = gt_txt.split('\n')

		img = cv2.imread(img_name)

	

		for gt_line in gt_split:
			gt_ind = gt_line.split('\t')
			if len(gt_ind) > 3:
				condinate_list = gt_ind[2].split(' ')
				pt1 = (float(condinate_list[0]), float(condinate_list[1]))
				pt2 = (float(condinate_list[2]), float(condinate_list[3]))
				pt3 = (float(condinate_list[4]), float(condinate_list[5]))
				pt4 = (float(condinate_list[6]), float(condinate_list[7]))
			
				edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
				edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

				angle = 0
			
				if edge1 > edge2:
				
					width = edge1
					height = edge2
					if pt1[0] - pt2[0] != 0:
						angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
					else:
						angle = 90.0
				elif edge2 >= edge1:
					width = edge2
					height = edge1
					#print pt2[0], pt3[0]
					if pt2[0] - pt3[0] != 0:
						angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
					else:
						angle = 90.0
				if angle < -45.0:
					angle = angle + 180

				x_ctr = float(pt1[0] + pt3[0]) / 2#pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
				y_ctr = float(pt1[1] + pt3[1]) / 2#pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

				boxes.append([x_ctr, y_ctr, height, width, angle])
			
		len_of_bboxes = len(boxes)
		gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)	
		gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
		overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32) #text or non-text
		seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)
		
		for idx in range(len(boxes)):
			gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
			gt_classes[idx] = 1 # cls_text
			overlaps[idx, 1] = 1.0 # cls_text
			seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])


		max_overlaps = overlaps.max(axis=1)
		# gt class that had the max overlap
		max_classes = overlaps.argmax(axis=1)

		im_info = {
			'gt_classes': gt_classes,
			'max_classes': max_classes,
			'image': img_name,
			'boxes': gt_boxes,
			'flipped' : False,
			'gt_overlaps' : overlaps,
			'seg_areas' : seg_areas,
			'height': img.shape[0],
			'width': img.shape[1],
			'max_overlaps' : max_overlaps,
			'rotated': True
			}
		im_infos.append(im_info)

		relative_info = {
			'gt_classes': gt_classes,
			'max_classes': max_classes,
			'image': RELATIVE_DIR + img_list[idx],
			'boxes': gt_boxes,
			'flipped' : False,
			'gt_overlaps' : overlaps,
			'seg_areas' : seg_areas,
			'height': img.shape[0],
			'width': img.shape[1],
			'max_overlaps' : max_overlaps,
			'rotated': True
			}
		relative_infos.append(relative_info)
	return im_infos, relative_infos

def get_ICDAR2015_test(mode):
	dir_path = "/home/shiki-alice/Downloads/ICDAR-2015-COMPETITION/public_test_data/"
	img_file_type = "png"

	RELATIVE_DIR = "../../../ICDAR-2015-COMPETITION/training_data/"

	cls = "gt"

	file_list = os.listdir(dir_path)

	gt_list = []
	img_list = []		

	

	for file_name in file_list:
		split = file_name.split(".")
		if split[len(split) - 1] == img_file_type:
			img_list.append(file_name)

	

	for img_ind in range(len(img_list)):
		split = img_list[img_ind].split(".")	
		gt_list.append(split[0] + "." + split[1] + "." + cls)

	im_infos = []
	relative_infos = []

	for idx in range(len(img_list)):

		img_name = dir_path + img_list[idx]
		gt_name = dir_path + gt_list[idx]

		boxes = []

		print gt_name
		gt_obj = open(gt_name, 'r')

		gt_txt = gt_obj.read()

		gt_split = gt_txt.split('\n')

		img = cv2.imread(img_name)

	

		for gt_line in gt_split:
			gt_ind = gt_line.split('\t')
			if len(gt_ind) > 3:
				condinate_list = gt_ind[2].split(' ')
				pt1 = (float(condinate_list[0]), float(condinate_list[1]))
				pt2 = (float(condinate_list[2]), float(condinate_list[3]))
				pt3 = (float(condinate_list[4]), float(condinate_list[5]))
				pt4 = (float(condinate_list[6]), float(condinate_list[7]))
			
				edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
				edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

				angle = 0
			
				if edge1 > edge2:
				
					width = edge1
					height = edge2
					if pt1[0] - pt2[0] != 0:
						angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
					else:
						angle = 90.0
				elif edge2 >= edge1:
					width = edge2
					height = edge1
					#print pt2[0], pt3[0]
					if pt2[0] - pt3[0] != 0:
						angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
					else:
						angle = 90.0
				if angle < -45.0:
					angle = angle + 180

				x_ctr = float(pt1[0] + pt3[0]) / 2#pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
				y_ctr = float(pt1[1] + pt3[1]) / 2#pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

				boxes.append([x_ctr, y_ctr, height, width, angle])
			
		len_of_bboxes = len(boxes)
		gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)	
		gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
		overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32) #text or non-text
		seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)
		
		for idx in range(len(boxes)):
			gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
			gt_classes[idx] = 1 # cls_text
			overlaps[idx, 1] = 1.0 # cls_text
			seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])


		max_overlaps = overlaps.max(axis=1)
		# gt class that had the max overlap
		max_classes = overlaps.argmax(axis=1)

		im_info = {
			'gt_classes': gt_classes,
			'max_classes': max_classes,
			'image': img_name,
			'boxes': gt_boxes,
			'flipped' : False,
			'gt_overlaps' : overlaps,
			'seg_areas' : seg_areas,
			'height': img.shape[0],
			'width': img.shape[1],
			'max_overlaps' : max_overlaps,
			'rotated': True
			}
		im_infos.append(im_info)

		relative_info = {
			'gt_classes': gt_classes,
			'max_classes': max_classes,
			'image': RELATIVE_DIR + img_list[idx],
			'boxes': gt_boxes,
			'flipped' : False,
			'gt_overlaps' : overlaps,
			'seg_areas' : seg_areas,
			'height': img.shape[0],
			'width': img.shape[1],
			'max_overlaps' : max_overlaps,
			'rotated': True
			}
		relative_infos.append(relative_info)
	return im_infos, relative_infos


def test_ICDAR2015(mode):
	dir_path = "/home/shiki-alice/Downloads/ICDAR2015/ch4_test_images/"
	file_list = os.listdir(dir_path)
	img_file_type = "jpg"
	im_infos = []

	img_list = []		
	for file_name in file_list:
		split = file_name.split(".")
		if split[len(split) - 1] == img_file_type:
			img_list.append(file_name)

			im_info = {
			'image': dir_path + file_name
			}
			im_infos.append(im_info)

	return im_infos

def get_bot_img(mode):
	
	bot_img_dir = {
			'noodle': '/var/www/html/data/BOT2/bot_train1/Train/Instant Noodle/',
			'shampoo': '/var/www/html/data/BOT2/bot_train1/Train/Shampoo',
			'potato': '/var/www/html/data/BOT2/bot_train1/Train/Potato Chips'
			}

	im_infos = []
	file_list = os.listdir(bot_img_dir[mode])

	for file_name in file_list:
		split = file_name.split(".")
		
		
		im_info = {
		'set': mode,
		'image': bot_img_dir[mode] + file_name
		}
		im_infos.append(im_info)
	return im_infos

def get_test_coco_img(mode):
	coco_train_dir = '/var/www/html/data/MSCOCO/train2014/'
	coco_val_dir = '/var/www/html/data/MSCOCO/val2014/'
	
	train_list = os.listdir(coco_train_dir)
	val_list = os.listdir(coco_val_dir)

	img_file_type = "jpg"
	im_infos = []

	if mode == 'train':	
		for file_name in train_list:
			split = file_name.split(".")
			if split[len(split) - 1] == img_file_type:
			

				im_info = {
				'set': 'train',
				'image': coco_train_dir + file_name
				}
				im_infos.append(im_info)
	if mode == 'val':
		for file_name in val_list:
			split = file_name.split(".")
			if split[len(split) - 1] == img_file_type:
			
				im_info = {
				'set': 'val',
				'image': coco_val_dir + file_name
				}
				im_infos.append(im_info)

	return im_infos

def angle_stat(roidb):

	angles = []

	for roi_infos in roidb:
		boxes = roi_infos["boxes"]
		for box in boxes:
			angles.append(box[4])

	for angle in angles:
		print angle

if __name__ == '__main__':
	#get_rroidb("train")
	roidb = data_extractor.get_ICDAR2013_test("test")
	for roi in roidb:
		print roi["image"]
		#show_result_ICDAR13(roi["image"])
		show_result_ICDAR(roi["image"])
	#roidb, reldb = get_ICDAR2015("train")
	#pick_training_set("ICDAR2015_train", roidb, reldb)
