#from rotate_circle_nms import rotate_cpu_nms as cpu_nms
#from rotate_gpu_nms import rotate_gpu_nms as cpu_nms
from rotation.rotate_polygon_nms import rotate_gpu_nms as cpu_nms
import numpy as np



boxes = np.array([
			[50, 50, 100, 100, 0,0.99],
			[60, 60, 100, 100, 0,0.88],#keep 0.68
			[50, 50, 100, 100, 45.0,0.66],#discard 0.70
			[200, 200, 100, 100, 0,0.77],#keep 0.0
			
		], dtype=np.float32)

	#boxes = np.tile(boxes, (4500 / 4, 1))

	#for ind in range(4500):
	#	boxes[ind, 5] = 0

a = cpu_nms(boxes, 0.7)


print a
