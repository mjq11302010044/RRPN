import numpy as np 
import os 
import cv2
import json
from xml.dom.minidom import parse
import xml.dom.minidom


trigger_int = 0

def get_picked_roidbs(roi_file_name):
	
	rois_dir = "./line_pick/"

	roi_fo = open(rois_dir + roi_file_name,'r')
	roi_str = roi_fo.read()
	roi_str = roi_str.replace("array", "np.array")
	roi_str = roi_str.replace("int16", "np.int16")
	roi_str = roi_str.replace("int32", "np.int32")
	roi_str = roi_str.replace("float32", "np.float32")
	return eval(roi_str)

if __name__ == '__main__':
	get_picked_roidbs("SVT_test")
	
	
