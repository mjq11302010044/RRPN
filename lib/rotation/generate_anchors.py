import numpy as np
from fast_rcnn.config import cfg

def generate_anchors(base_size = 16, ratios = [0.2, 0.5, 1.0],
                     scales = 2**np.arange(3, 6), angle = [-30.0, 0.0, 30.0, 60.0, 90.0, 120.0]):


	######################################
	# Parameter:
	######################################
	#	ratio: [0.5, 1, 2]
	#	scales: 2^3, 2^4, 2^5
	#	angle: [-45, 0, 45]
	######################################
	#	(project in original patch)
	######################################

	# [x_ctr, y_ctr, height, width, theta] anti-clock-wise angle

	if len(cfg.TEST.RATIO_GROUP) != 0:
		ratios = cfg.TEST.RATIO_GROUP

	if len(cfg.TEST.SCALE_GROUP) != 0:
		ratios = cfg.TEST.SCALE_GROUP

	if len(cfg.TEST.ANGLE_GROUP) != 0:
		ratios = cfg.TEST.ANGLE_GROUP

	base_anchor = np.array([base_size / 2 - 1, base_size / 2 - 1, base_size, base_size, 0], dtype = np.float32)
	
	ratio_anchors = _ratio_enum(base_anchor, ratios)
	
	scale_anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in xrange(ratio_anchors.shape[0])])

	anchors =  np.vstack([_angle_enum(scale_anchors[i, :], angle) for i in xrange(scale_anchors.shape[0])])

	anchors[:, 2:4] = anchors[:, 3:1:-1]
	return anchors


def _ratio_enum(anchor, ratios):

	x_ctr, y_ctr, width, height, theta = anchor
	
	
	size = width * height

    	size_ratios = size / ratios
    	ws = np.round(np.sqrt(size_ratios))
    	hs = np.round(ws * ratios)
	
	

	ws = ws[:, np.newaxis]
    	hs = hs[:, np.newaxis]

	

	x_ctr_list = []
	y_ctr_list = []

	
	ctr = np.tile([x_ctr, y_ctr], (ws.shape[0], 1))
	theta = np.tile([theta], (ws.shape[0], 1))
	
        return np.hstack((ctr, ws, hs, theta))                 
                         

    	#return [x_ctr,y_ctr,ws,hs]

def _scale_enum(anchor, scales):
	

	x_ctr, y_ctr, width, height, theta = anchor

	ws = width * scales
	hs = height * scales
	
	x_ctr_list = []
	y_ctr_list = []

	
	ctr = np.tile([x_ctr, y_ctr], (ws.shape[0], 1))
	theta = np.tile([theta], (ws.shape[0], 1))

	ws = ws[:, np.newaxis]
    	hs = hs[:, np.newaxis]
	
	return np.hstack((ctr, ws, hs, theta))
	

def _angle_enum(anchor, angle):

	x_ctr, y_ctr, width, height, theta = anchor	
	
	ctr = np.tile([x_ctr, y_ctr, width, height], (len(angle), 1))

	angle = [[ele] for ele in angle]

	return np.hstack((ctr, angle))


if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print time.time() - t
    print a
    

