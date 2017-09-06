import os
import numpy as np

#[ctr_x, ctr_y, height, width, angle] anti-clock-wise arc
def rbbox_transform(ex_rois, gt_rois):
	ex_widths = ex_rois[:, 3] 
	ex_heights = ex_rois[:, 2] 
	ex_ctr_x = ex_rois[:, 0]
	ex_ctr_y = ex_rois[:, 1]
	ex_angle = ex_rois[:, 4] 	

	gt_widths = gt_rois[:, 3]
	gt_heights = gt_rois[:, 2]
	gt_ctr_x = gt_rois[:, 0]
	gt_ctr_y = gt_rois[:, 1]
	gt_angle = gt_rois[:, 4]

	targets_dx = (gt_ctr_x - ex_ctr_x)*1.0 / ex_widths
    	targets_dy = (gt_ctr_y - ex_ctr_y)*1.0 / ex_heights
    	targets_dw = np.log(gt_widths*1.0 / ex_widths)
    	targets_dh = np.log(gt_heights*1.0 / ex_heights)

	#ex_angle = np.pi / 180 * ex_angle
	#gt_angle = np.pi / 180 * gt_angle
	
	targets_da = gt_angle - ex_angle

	targets_da[np.where((gt_angle<=-30) & (ex_angle>=120))]+=180
	targets_da[np.where((gt_angle>=120) & (ex_angle<=-30))]-=180

	targets_da = 3.14159265358979323846264338327950288/180*targets_da
	

	targets = np.vstack(
		 (targets_dx, targets_dy, targets_dw, targets_dh, targets_da)
	).transpose()

	return targets

def rbbox_transform_inv(boxes, deltas):

	if boxes.shape[0] == 0:
        	return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    	boxes = boxes.astype(deltas.dtype, copy=False)

	widths = boxes[:, 3]
	heights = boxes[:, 2]
	ctr_x = boxes[:, 0]
	ctr_y = boxes[:, 1]

	angle = boxes[:, 4]
	
	dx = deltas[:, 0::5]
    	dy = deltas[:, 1::5]
    	dw = deltas[:, 2::5]
   	dh = deltas[:, 3::5]
	da = deltas[:, 4::5]

	
    	pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    	pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    	pred_w = np.exp(dw) * widths[:, np.newaxis]
    	pred_h = np.exp(dh) * heights[:, np.newaxis]
	
	da = da*1.0 / np.pi * 180 # arc to angle

	pred_angle = da + angle[:, np.newaxis] 

        #step 6: bounding judge (-45,135)
	pred_angle[np.where(pred_angle<-45)] += 180
	pred_angle[np.where(pred_angle>135)] -= 180
	
    	pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    	# ctr_x1
    	pred_boxes[:, 0::5] = pred_ctr_x
    	# ctr_y1
    	pred_boxes[:, 1::5] = pred_ctr_y
  	# height
    	pred_boxes[:, 2::5] = pred_h
    	# width
    	pred_boxes[:, 3::5] = pred_w
	# angle
	pred_boxes[:, 4::5] = pred_angle

    	return pred_boxes

if __name__ == "__main__":
	ex_rois = np.array([[100,100,100,100,45],[33,34,76,2,123]])
	gt_rois = np.array([[101,99,50,50,30],[123,545,3,5,-23]])
	targets = rbbox_transform(ex_rois,gt_rois)
	print targets
	print rbbox_transform_inv(ex_rois,targets)
	

	
		

























	






