import numpy as np 
import os 
import cv2
from xml.dom.minidom import parse
import xml.dom.minidom
from PIL import Image
import pickle

trigger_int = 0

def get_rroidb():	

	L_DATASET = []

	L_ICDAR2003_TRAIN = get_ICDAR2003("train")
	print "L_ICDAR2003_TRAIN:", len(L_ICDAR2003_TRAIN)
	
	L_DATASET.extend(L_ICDAR2003_TRAIN)
	print "total", len(L_DATASET)

	count = 0
	for roi in L_DATASET:
		count += len(roi["boxes"])
	print "the text instance of total: ", count

	return L_DATASET


def test_rroidb(mode):
	return get_MSRA(mode)


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


def pick_training_set(dataset_name, roidb):
	
	global trigger_int

	pick_dir = "./line_pick/"

	PR_file = pick_dir + dataset_name

	if not os.path.isdir(pick_dir):
		os.mkdir(pick_dir)
	if not os.path.isfile(PR_file):
		os.mknod(PR_file)

	PR_obj = open(PR_file, "a+")

	for roi_data in roidb:
		image_path = roi_data["image"]
		boxes =  roi_data["boxes"]

		vis_image(image_path, boxes)
		print trigger_int

		PR_obj.write(str(roi_data) + "\n")

    	PR_obj.close()

def vis_image(image_path, boxes):

	img = cv2.imread(image_path)
	cv2.namedWindow("image")	
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

	cv2.imshow("image", cv2.resize(img, (1024, 768)))
	cv2.waitKey(0)


def trigger(event, x, y, flags, param):
	global trigger_int
	
	if event == cv2.EVENT_LBUTTONDOWN:
		trigger_int = 1
	elif event == cv2.EVENT_RBUTTONDOWN:
		trigger_int = 0


def get_ICDAR2017_mlt(mode, task, prefetched):

	DATASET_DIR = '/var/www/html/data/ICDAR2017/'

	im_infos = []

	data_list = []
	gt_list = []
	img_type = ['jpg', 'png', 'gif']
	cls_list = {'background':0, 'Arabic':1, 'English':2, 'Japanese':3, 'French':4, 'German':5, 'Chinese':6, 'Korean':7, 'Italian':8, 'Bangla':9}

	if not prefetched:
		# training set contains 7200 images with 
		if mode == "train":
			for i in range(7200):
				img_candidate_path = DATASET_DIR + "ch8_training_images_" + str((i / 1000) + 1) + "/" + 'img_' + str(i+1) + "."
				if os.path.isfile(img_candidate_path + img_type[0]):
					img_candidate_path += img_type[0]	
				elif os.path.isfile(img_candidate_path + img_type[1]):
					img_candidate_path += img_type[1]	
				elif os.path.isfile(img_candidate_path + img_type[2]):
					im = Image.open(img_candidate_path + img_type[2])
					im = im.convert('RGB')
					im.save(img_candidate_path + "jpg","jpeg")
					img_candidate_path = img_candidate_path + "jpg"
				data_list.append(img_candidate_path)
				#print ("data_list:", len(data_list))	
			
				gt_candidate_path = DATASET_DIR + "ch8_training_localization_transcription_gt/" + 'gt_img_' + str(i+1) + ".txt"	
				if os.path.isfile(gt_candidate_path):
					gt_list.append(gt_candidate_path)	
				#print ("gt_list:", len(gt_list))

				f_gt = open(gt_candidate_path)
				f_content = f_gt.read()
			
				lines = f_content.split('\n')
				print (img_candidate_path)
				img = cv2.imread(img_candidate_path)
				boxes = []
				for gt_line in lines:
					#print (gt_line)
					gt_ind = gt_line.split(',')
					
					if len(gt_ind) > 3:
					
						pt1 = (int(gt_ind[0]), int(gt_ind[1]))
						pt2 = (int(gt_ind[2]), int(gt_ind[3]))
						pt3 = (int(gt_ind[4]), int(gt_ind[5]))
						pt4 = (int(gt_ind[6]), int(gt_ind[7]))
			
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

						boxes.append([x_ctr, y_ctr, height, width, angle, gt_ind[8]])

				print ("line_size:", len(lines))
				

				cls_num = 2
				if task == "multi_class":
					cls_num = len(cls_list.keys())

				len_of_bboxes = len(boxes)
				gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)	
				gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
				overlaps = np.zeros((len_of_bboxes, cls_num), dtype=np.float32) #text or non-text
				seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)
		
				if task == "multi_class":
					gt_boxes = [] #np.zeros((len_of_bboxes, 5), dtype=np.int16)	
					gt_classes = [] #np.zeros((len_of_bboxes), dtype=np.int32)
					overlaps = [] #np.zeros((len_of_bboxes, cls_num), dtype=np.float32) #text or non-text
					seg_areas = [] #np.zeros((len_of_bboxes), dtype=np.float32)

				for idx in range(len(boxes)):
				
					if task == "multi_class":
						if not boxes[idx][5] in cls_list:
							print (boxes[idx][5] + " not in list")
							continue
						gt_classes.append(cls_list[boxes[idx][5]]) # cls_text
						overlap = np.zeros((cls_num))
						overlap[cls_list[boxes[idx][5]]] = 1.0 # prob
						overlaps.append(overlap)
						seg_areas.append((boxes[idx][2]) * (boxes[idx][3]))
						gt_boxes.append([boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]])
					else:
						gt_classes[idx] = 1 # cls_text
						overlaps[idx, 1] = 1.0 # prob
						seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])
						gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]

				if task == "multi_class":
					gt_classes = np.array(gt_classes)
					overlaps = np.array(overlaps) 
					seg_areas = np.array(seg_areas) 
					gt_boxes = np.array(gt_boxes)

				print ("boxes_size:", gt_boxes.shape[0])
				if gt_boxes.shape[0] > 0:
					max_overlaps = overlaps.max(axis=1)
					# gt class that had the max overlap
					max_classes = overlaps.argmax(axis=1)

				im_info = {
					'gt_classes': gt_classes,
					'max_classes': max_classes,
					'image': img_candidate_path,
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
			
			f_save_pkl = open('ICDAR2017_training_cache.pkl', 'wb')
			pickle.dump(im_infos, f_save_pkl)
			f_save_pkl.close()
			print ("Save pickle done.")						
		elif mode == "validation":
			for i in range(1800):
				img_candidate_path = DATASET_DIR + "ch8_validation_images/" + 'img_' + str(i+1) + "."
				if os.path.isfile(img_candidate_path + img_type[0]):
					img_candidate_path += img_type[0]	
				elif os.path.isfile(img_candidate_path + img_type[1]):
					img_candidate_path += img_type[1]	
				elif os.path.isfile(img_candidate_path + img_type[2]):
					im = Image.open(img_candidate_path + img_type[2])
					im = im.convert('RGB')
					im.save(img_candidate_path + "jpg","jpeg")
					img_candidate_path = img_candidate_path + "jpg"
				data_list.append(img_candidate_path)
				#print ("data_list:", len(data_list))	
			
				gt_candidate_path = DATASET_DIR + "ch8_validation_localization_transcription_gt/" + 'gt_img_' + str(i+1) + ".txt"	
				if os.path.isfile(gt_candidate_path):
					gt_list.append(gt_candidate_path)	
				#print ("gt_list:", len(gt_list))

				f_gt = open(gt_candidate_path)
				f_content = f_gt.read()
			
				lines = f_content.split('\n')
				print (img_candidate_path)
				img = cv2.imread(img_candidate_path)
				boxes = []
		
				for gt_line in lines:
					#print (gt_line)
					gt_ind = gt_line.split(',')
					if len(gt_ind) > 3:
					
						pt1 = (int(gt_ind[0]), int(gt_ind[1]))
						pt2 = (int(gt_ind[2]), int(gt_ind[3]))
						pt3 = (int(gt_ind[4]), int(gt_ind[5]))
						pt4 = (int(gt_ind[6]), int(gt_ind[7]))
			
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

						boxes.append([x_ctr, y_ctr, height, width, angle, gt_ind[8]])
			
				cls_num = 2
				if task == "multi_class":
					cls_num = len(cls_list.keys())

				len_of_bboxes = len(boxes)
				gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)	
				gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
				overlaps = np.zeros((len_of_bboxes, cls_num), dtype=np.float32) #text or non-text
				seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)
		
				for idx in range(len(boxes)):
				
					if task == "multi_class":
						if not boxes[idx][5] in cls_list:
							break
						gt_classes[idx] = cls_list[boxes[idx][5]] # cls_text
						overlaps[idx, cls_list[boxes[idx][5]]] = 1.0 # prob
					else:
						gt_classes[idx] = 1 # cls_text
						overlaps[idx, 1] = 1.0 # prob
					seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])
					gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]

				max_overlaps = overlaps.max(axis=1)
				# gt class that had the max overlap
				max_classes = overlaps.argmax(axis=1)

				im_info = {
					'gt_classes': gt_classes,
					'max_classes': max_classes,
					'image': img_candidate_path,
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
			
			f_save_pkl = open('ICDAR2017_validation_cache.pkl', 'wb')
			pickle.dump(im_infos, f_save_pkl)
			f_save_pkl.close()
			print ("Save pickle done.")
	else:
		if mode == "train":
			f_pkl = open('ICDAR2017_training_cache.pkl', 'rb')
			im_infos = pickle.load(f_pkl)
		if mode == "validation":
			f_pkl = open('ICDAR2017_validation_cache.pkl', 'rb')
			im_infos = pickle.load(f_pkl)
	return im_infos
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

	DATASET_DIR = "./data/datasets/ICDAR03/"

	if mode == "train":
		DATASET_DIR = DATASET_DIR + "SceneTrialTrain/"
	elif mode == "test":
		DATASET_DIR = DATASET_DIR + "SceneTrialTest/"

	gt_xml_file = "locations.xml"

	gt_tree = xml.dom.minidom.parse(DATASET_DIR + gt_xml_file)

	gt_collection = gt_tree.documentElement

	im_infos = []

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
	return im_infos

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

	gt_xml_file = DATASET_DIR + mode + ".xml"

	gt_tree = xml.dom.minidom.parse(gt_xml_file)

	gt_collection = gt_tree.documentElement

	im_infos = []

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
		
	return im_infos

def get_ICDAR2013(mode):
	DATASET_DIR = "/home/shiki-alice/Downloads/ICDAR2013"

	img_dir = "/Challenge2_Training_Task12_Images/"
	gt_dir = "/Challenge2_Training_Task1_GT/"


	#gt_list = []
	#img_list = []

	im_infos = []
	
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
	return im_infos

def get_ICDAR2013_test(mode):
	DATASET_DIR = "/home/shiki-alice/Downloads/ICDAR2013"

	img_dir = "/Challenge2_Test_Task12_Images/"
	
	


	im_infos = []
	file_list = os.listdir(DATASET_DIR + img_dir)

	for file_name in file_list:
		split = file_name.split(".")
		
		file_type = split[1]
		
		if file_type == "gif":
			im = Image.open(DATASET_DIR + img_dir + file_name)
			#im.show()
			im = im.convert('RGB')
			im.save(DATASET_DIR + img_dir + split[0] + ".jpg","jpeg")
			file_name = split[0] + ".jpg"
		im_info = {
		'set': mode,
		'image': DATASET_DIR + img_dir + file_name
		}
		im_infos.append(im_info)

	
	return im_infos

def get_ICDAR2013_test_ch1(mode):
	DATASET_DIR = "/home/shiki-alice/Downloads"

	img_dir = "/Challenge1_Test_Task12_Images/"
	
	


	im_infos = []
	file_list = os.listdir(DATASET_DIR + img_dir)

	for file_name in file_list:
		split = file_name.split(".")
		
		file_type = split[1]
		
		if file_type == "gif":
			im = Image.open(DATASET_DIR + img_dir + file_name)
			#im.show()
			im = im.convert('RGB')
			im.save(DATASET_DIR + img_dir + split[0] + ".jpg","jpeg")
			file_name = split[0] + ".jpg"
		im_info = {
		'set': mode,
		'image': DATASET_DIR + img_dir + file_name
		}
		im_infos.append(im_info)

	
	return im_infos


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

def get_ICDAR2015(mode):
	dir_path = "/home/shiki-alice/Downloads/ICDAR-2015-COMPETITION/training_data/"
	img_file_type = "png"

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

		
	return im_infos

def get_ICDAR2015_test(mode):
	dir_path = "/home/shiki-alice/Downloads/ICDAR-2015-COMPETITION/public_test_data/"
	img_file_type = "png"

	

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

		
	return im_infos

def get_ICDAR2015_RRC_TRAIN(mode):
	dir_path = "/home/shiki-alice/Downloads/ICDAR2015/ch4_training_images/"
	img_file_type = "jpg"

	gt_dir = "/home/shiki-alice/Downloads/ICDAR2015/ch4_training_localization_transcription_gt/"

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
		gt_list.append(cls + "_" + split[0] + ".txt")

	im_infos = []

	for idx in range(len(img_list)):

		img_name = dir_path + img_list[idx]
		gt_name = gt_dir + gt_list[idx]

		boxes = []

		#print gt_name
		gt_obj = open(gt_name, 'r')

		gt_txt = gt_obj.read()

		gt_split = gt_txt.split('\n')

		img = cv2.imread(img_name)

	
		f = False
		for gt_line in gt_split:
			if not f:
				gt_ind = gt_line.split('\\')
				#print gt_ind
#$.split(',')
				f = True
			else:
				gt_ind = gt_line.split(',')
			if len(gt_ind) > 3:
				#condinate_list = gt_ind[2].split(',')
				#print gt_ind
				pt1 = (int(gt_ind[0]), int(gt_ind[1]))
				pt2 = (int(gt_ind[2]), int(gt_ind[3]))
				pt3 = (int(gt_ind[4]), int(gt_ind[5]))
				pt4 = (int(gt_ind[6]), int(gt_ind[7]))
			
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

		
	return im_infos

def get_ICDAR2015_RRC_PICK_TRAIN(mode):
	dir_path = "/home/shiki-alice/Downloads/ICDAR2015/ch4_training_images/"
	img_file_type = "jpg"

	gt_dir = "/home/shiki-alice/Downloads/ICDAR2015/ch4_training_localization_transcription_gt/"

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
		gt_list.append(cls + "_" + split[0] + ".txt")

	im_infos = []

	for idx in range(len(img_list)):

		img_name = dir_path + img_list[idx]
		gt_name = gt_dir + gt_list[idx]

		easy_boxes = []
		hard_boxes = []

		boxes = []
		#print gt_name
		gt_obj = open(gt_name, 'r')

		gt_txt = gt_obj.read()

		gt_split = gt_txt.split('\n')

		img = cv2.imread(img_name)

	
		f = False
		#print '-------------'
		for gt_line in gt_split:
			
			if not f:
				gt_ind = gt_line.split('\\')
				#print gt_ind
#$.split(',')
				f = True
			else:
				gt_ind = gt_line.split(',')
			if len(gt_ind) > 3 and gt_ind[8] != '###\r':
				#condinate_list = gt_ind[2].split(',')
				#print "easy: ", gt_ind

				pt1 = (int(gt_ind[0]), int(gt_ind[1]))
				pt2 = (int(gt_ind[2]), int(gt_ind[3]))
				pt3 = (int(gt_ind[4]), int(gt_ind[5]))
				pt4 = (int(gt_ind[6]), int(gt_ind[7]))
			
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

				easy_boxes.append([x_ctr, y_ctr, height, width, angle])

			if len(gt_ind) > 3 and gt_ind[8] == '###\r':
				#condinate_list = gt_ind[2].split(',')
				
				#print "hard: ", gt_ind

				pt1 = (int(gt_ind[0]), int(gt_ind[1]))
				pt2 = (int(gt_ind[2]), int(gt_ind[3]))
				pt3 = (int(gt_ind[4]), int(gt_ind[5]))
				pt4 = (int(gt_ind[6]), int(gt_ind[7]))
			
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

				hard_boxes.append([x_ctr, y_ctr, height, width, angle])
		
		boxes.extend(easy_boxes)
		
		boxes.extend(hard_boxes[0 : len(hard_boxes) / 2])
			
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

		
	return im_infos

def get_VEDAI_vihicle(mode, group_num, resolution):
	anno_dir = "/var/www/html/data/VEDAI/Annotations" + str(resolution) + "/"
	dataset_dir = "/var/www/html/data/VEDAI/Vehicules" + str(resolution) + "/";
	train_content = "";
	test_content = "";

	num = group_num

	#{'Car', 'Truck', 'Ship', 'Tractor', 'Camping Car', 'van', 'vehicle', 'pick-up', 'plane'}
	cls_map = {1:1, 2:2, 23:3, 4:4, 5:5, 9:6, 10:7, 11:8, 31:9, 7:10}

	if num < 10:
		if mode == "train":
			train_set_file = "fold0" + str(num) + ".txt"
			train_content = open(anno_dir + train_set_file, "r").read()

			print "exp" ,num , "num of train data: ", len(train_content.split("\n"))
		if mode == "test":
			test_set_file = "fold0" + str(num) + "test.txt"
			test_content = open(anno_dir + test_set_file, "r").read()

			print "exp" ,num , "num of test data: ", len(test_content.split("\n"))
	else:

		if mode == "train":
			train_set_file = "fold" + str(num) + ".txt"
			train_content = open(anno_dir + train_set_file, "r").read()

			print "exp" ,num , "num of train data: ", len(train_content.split("\n"))
		if mode == "test":
			test_set_file = "fold" + str(num) + "test.txt"
			test_content = open(anno_dir + test_set_file, "r").read()

			print "exp" ,num , "num of test data: ", len(test_content.split("\n"))

	im_infos = []		

	if train_content != "" or test_content != "":
	
		for img_prefix in train_content.split("\n"):
			if img_prefix != "":
				gt_file = img_prefix + ".txt"
				gt_lines = open(anno_dir + gt_file, "r").read().split("\n")

				boxes = []

				for line in gt_lines:
					if line != "":
											
						gt_ids = line.split(" ")
						
						x_ctr = float(gt_ids[0])
						y_ctr = float(gt_ids[1])

						pt1 = (int(gt_ids[6]), int(gt_ids[10]))
						pt2 = (int(gt_ids[7]), int(gt_ids[11]))
						pt3 = (int(gt_ids[8]), int(gt_ids[12]))
						pt4 = (int(gt_ids[9]), int(gt_ids[13]))
	
						edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
						edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))
						edge3 = np.sqrt((pt3[0] - pt4[0]) * (pt3[0] - pt4[0]) + (pt3[1] - pt4[1]) * (pt3[1] - pt4[1]))
						edge4 = np.sqrt((pt4[0] - pt1[0]) * (pt4[0] - pt1[0]) + (pt4[1] - pt1[1]) * (pt4[1] - pt1[1]))
						

						edge1 = (edge1 + edge3) / 2
						edge2 = (edge2 + edge4) / 2
						
						height = 0
						width = 0

						if edge1 > edge2:
			
							width = edge1
							height = edge2
							if pt1[0] - pt2[0] != 0 and pt3[0] - pt4[0] != 0:
								angle1 = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
								angle2 = -np.arctan(float(pt3[1] - pt4[1]) / float(pt3[0] - pt4[0])) / 3.1415926 * 180
								angle = (angle1 + angle2) / 2
							else:
								angle = 90.0
						elif edge2 >= edge1:
							width = edge2
							height = edge1
							#print pt2[0], pt3[0]
							if pt2[0] - pt3[0] != 0 and pt1[0] - pt4[0] != 0:
								angle1 = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
								angle2 = -np.arctan(float(pt1[1] - pt4[1]) / float(pt1[0] - pt4[0])) / 3.1415926 * 180
								angle = (angle1 + angle2) / 2
							else:
								angle = 90.0

						#angle = -float(gt_ids[2]) / 3.1415926 * 180

						if angle < -45.0:
							angle = angle + 180
						elif angle > 135.0:
							angle = angle - 180

						boxes.append([x_ctr, y_ctr, height, width, angle])

				len_of_bboxes = len(boxes)
				gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)	
				gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
				overlaps = np.zeros((len_of_bboxes, 11), dtype=np.float32) #text or non-text
				seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

				for idx in range(len(boxes)):
					
					gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
					gt_classes[idx] = cls_map[int(gt_ids[3])] # cls_text
					overlaps[idx, cls_map[int(gt_ids[3])]] = 1.0 # cls_text
					seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

					if int(gt_ids[3]) == 7:
						print dataset_dir + img_prefix + "_co.png"
						print "cls: ", int(gt_ids[3])

				max_overlaps = overlaps.max(axis=1)
				# gt class that had the max overlap
				max_classes = overlaps.argmax(axis=1)

				img_name = dataset_dir + img_prefix + "_co.png"

				img = cv2.imread(img_name)

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
	#roidb = get_VEDAI_vihicle("train", 1, 1024)
	#print len(roidb)
	import random
	roidb = get_ICDAR2017_mlt('train', 'multi_class', False)

	L_bboxes = []
	random.shuffle(roidb)
	for roi in roidb:
		L_bboxes.extend(roi["boxes"])
		print (roi["image"])
		vis_image(roi["image"], roi["boxes"])

	out_data = ""

	for i in range(len(L_bboxes)):
		
		out_data += str(i) + "\t" + str(L_bboxes[i][3] / float(L_bboxes[i][2])) + "\r\n"



	f = file("blogdata.txt", "w+")
	f.write(out_data)
	f.close()
