import os
import numpy as np
from rotation.rbbox_overlaps import rbbx_overlaps
from rotation.data_pick import vis_image


def center_dir(bbox,thres,thres2,thres_height):# thres_height > 1
	xx = bbox[:,0].copy().reshape(-1,1)
	yy = bbox[:,1].copy().reshape(-1,1)
	xx2 = bbox[:,0].copy().reshape(1,-1)
	yy2 = bbox[:,1].copy().reshape(1,-1)
	
	dx = xx - xx2
	dy = yy - yy2
	da = -np.arctan(dy/dx)
	
	idx = np.where(da<-45/180.0*np.pi)
	da[idx]+=np.pi

	idx = np.where(da>135/180.0*np.pi)
	da[idx]-=np.pi

	gt_a = bbox[:,4]/180.0*np.pi
	angle_diff  = np.abs(da-gt_a)
	same_line =  angle_diff<thres
	same_line = same_line & same_line.T
	
	dist = np.sqrt(dx*dx+dy*dy)
	ww = (bbox[:,3]/2).copy().reshape(-1,1)
	ww2 = ww.copy().reshape(1,-1)
	dw = ww + ww2
	
	hh = (bbox[:,2]).copy().reshape(-1,1)
	hh2 = hh.copy().reshape(1,-1)
	dh = np.array(hh, np.float32) / hh2

	dist = dist < dw*thres2

	h_sameline = (dh < thres_height) & (dh > 1.0 / thres_height)

	same_line = same_line&dist&h_sameline
	
	dict ={}
	sz = same_line.shape[0]
	num = 0
	for i in range(sz):
		for j in range(i+1,sz,1):
			if same_line[i,j]:
				if dict.has_key(i):
					if dict.has_key(j):
						pass
					else:
						dict[j] = dict[i]
				else:
					if dict.has_key(j):
						dict[i] = dict[j]
					else:
						dict[i] = dict[j] = num
						num+=1
	keep_set = []	
	for i in range(sz):
		if not dict.has_key(i):
			keep_set.append(i)

	inv_dict = {}
	for key in dict:
		ikey = dict[key]
		ival = key
		if inv_dict.has_key(ikey):
			inv_dict[ikey].append(ival)
		else:
			inv_dict[ikey] = [ival]
	return inv_dict,keep_set		


def mul(bbox,ratio):
	bbox[:,2] = bbox[:,2]*ratio
	bbox[:,3] = bbox[:,3]*ratio
	return bbox

def merge_box(merge_set,bbox,keep_set):
	#print merge_set,keep_set
	res = []
	for idx in keep_set:
		res.append(bbox[idx].tolist())
	#print merge_set
	#print bbox
	
	for k in merge_set:
		indice = merge_set[k]
		pts = []
		a = 0
		s = 0
		h = 0
		for idx in indice:
			box = bbox[idx]
			a+=box[4]
			s+=box[5]
			h+=box[2]
		a/=len(indice)
		s/=len(indice)
		h/=len(indice)
		boxes = bbox[np.array(indice)]
		#print 'boxes',boxes.shape
		#print boxes[:,0]
		if boxes[0,4]<45.0:
			order = boxes[:,0].argsort()
		else:
			order = boxes[:,1].argsort()
		#print order,indice
		order = np.array(indice)[order]
		#print order
		
		xx = bbox[:,0].copy().reshape(-1,1)
		yy = bbox[:,1].copy().reshape(-1,1)
		xx2 = bbox[:,0].copy().reshape(1,-1)
		yy2 = bbox[:,1].copy().reshape(1,-1)
		dx = xx - xx2
		dy = yy - yy2
		dist = np.sqrt(dx*dx+dy*dy)
		ww = bbox[:,3]/2
		
		l = len(indice)
		w0 = ww[order[0]]
		w2 = ww[order[l-1]]
		w1 = 0
		for i in range(l-1):
			w1 += dist[order[i],order[i+1]]
		w = w0+w1+w2
		c1 = bbox[order[0],:2]
		c2 = bbox[order[l-1],:2]
		c = (w1*(c1+c2)+w0*(c1-c2)+w2*(c2-c1) )/ (2*w1)
		
		res.append([c[0],c[1],h,w,a,s])

	
	return res


def merge(bbox, angle=9, thres_width=1.1, thres_height=1.5, padding=1.0):
	if bbox.shape[0] == 0:
		return bbox
	bbox = mul(bbox,padding)
	merge_set,keep_set = center_dir(bbox,angle/180.0*np.pi,thres_width, thres_height)
	merged_res = merge_box(merge_set,bbox,keep_set)
	ret_box = mul(np.array(merged_res),1.0/padding)
	return ret_box



def merge_write(root):
	files =  os.listdir(root)
	for _file in files:
		f = open(root+'/'+_file)
		print _file
		lines = f.readlines()
		bbox = []
		for line in lines:
			record = line.strip().split(' ')
			bbox.append([float(record[0]),float(record[1]),float(record[2]),float(record[3]),float(record[4]),float(record[5])])		
		im_l = _file.split(".")

		#image_path = "/home/shiki-alice/Downloads/MSRA-TD500/test/" + im_l[0] + "." + im_l[1]
	 
		res = merge(np.array(bbox))
		f.close()
	
		g = open(root+'/'+_file,'w+')
		res = res.tolist()
		for r in res:
			print>>g,r[0],r[1],r[2],r[3],r[4],r[5]
		g.close()
if __name__ == "__main__":
	root = '/home/shiki-alice/workspace/Rotation-Faster-RCNN/py-faster-rcnn/result/729/exp3/vgg16_faster_rcnn_iter_180000.caffemodel/test'
	merge_write(root)
