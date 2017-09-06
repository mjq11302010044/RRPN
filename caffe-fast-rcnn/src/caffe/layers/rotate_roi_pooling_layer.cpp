#include <cfloat>
#include <cmath>

#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void RotateROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  RotateROIPoolingParameter rotate_pool_param = this->layer_param_.rotate_pooling_param();
  CHECK_GT(rotate_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(rotate_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = rotate_pool_param.pooled_h();
  pooled_width_ = rotate_pool_param.pooled_w();
  spatial_scale_ = rotate_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void RotateROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
}

template <typename Dtype>
void RotateROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  const Dtype* image_info = bottom[2]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);
  int imageWidth = int(image_info[1]*spatial_scale_+0.5);
  int imageHeight = int(image_info[0]*spatial_scale_+0.5);

  // For each ROI R = [batch_index Cx Cy height width angle]: max pool over R 
  for (int n = 0; n < num_rois; ++n) {
    // Points
    int roi_batch_ind = bottom_rois[0];
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);
    Dtype cx = bottom_rois[1];
    Dtype cy = bottom_rois[2];
    Dtype h = bottom_rois[3];
    Dtype w = bottom_rois[4];
    Dtype angle = bottom_rois[5]/180.0*3.1415926535;
    
    //TransformPrepare
    Dtype dx = -pooled_width_/2.0;
    Dtype dy = -pooled_height_/2.0;
    Dtype Sx = w*spatial_scale_/pooled_width_;
    Dtype Sy = h*spatial_scale_/pooled_height_;
    Dtype Alpha = cos(angle);
    Dtype Beta = sin(angle);
    Dtype Dx = cx*spatial_scale_;
    Dtype Dy = cy*spatial_scale_;
	 
    Dtype M[2][3]; 
    M[0][0] = Alpha*Sx;
    M[0][1] = Beta*Sy;
    M[0][2] = Alpha*Sx*dx+Beta*Sy*dy+Dx;
    M[1][0] = -Beta*Sx;
    M[1][1] = Alpha*Sy;
    M[1][2] = -Beta*Sx*dx+Alpha*Sy*dy+Dy;

    /*std::cout<<M[0][0]<<std::endl;
    std::cout<<M[0][1]<<std::endl;
    std::cout<<M[0][2]<<std::endl;
    std::cout<<M[1][0]<<std::endl;
    std::cout<<M[1][1]<<std::endl;
    std::cout<<M[1][2]<<std::endl;    
*/
    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          const int pool_index = ph * pooled_width_ + pw;
          Dtype P[8];
          P[0] = M[0][0]*pw+M[0][1]*ph+M[0][2];
          P[1] = M[1][0]*pw+M[1][1]*ph+M[1][2];
          P[2] = M[0][0]*pw+M[0][1]*(ph+1)+M[0][2];
          P[3] = M[1][0]*pw+M[1][1]*(ph+1)+M[1][2];
          P[4] = M[0][0]*(pw+1)+M[0][1]*ph+M[0][2];
          P[5] = M[1][0]*(pw+1)+M[1][1]*ph+M[1][2];
          P[6] = M[0][0]*(pw+1)+M[0][1]*(ph+1)+M[0][2];
          P[7] = M[1][0]*(pw+1)+M[1][1]*(ph+1)+M[1][2];
	  
 	  std::cout<<imageWidth<<imageHeight<<std::endl;
          int leftMost = int(max(round(min(min(P[0],P[2]),min(P[4],P[6]))),0.0));
          int rightMost= int(min(round(max(max(P[0],P[2]),max(P[4],P[6]))),imageWidth-1.0));
          int topMost= int(max(round(min(min(P[1],P[3]),min(P[5],P[7]))),0.0));
          int bottomMost= int(min(round(max(max(P[1],P[3]),max(P[5],P[7]))),imageHeight-1.0));
	  //bool is_empty = (rightMost<= leftMost) || (bottomMost <= topMost);         

          //std::cout<<leftMost<<rightMost<<topMost<<bottomMost<<std::endl;
          Dtype AB[2];
          AB[0] = P[2] - P[0];
          AB[1] = P[3] - P[1];	
			 Dtype ABAB = AB[0]*AB[0] +AB[1]*AB[1];
			 
			 Dtype AC[2];
          AC[0] = P[4] - P[0];
          AC[1] = P[5] - P[1];
          Dtype ACAC = AC[0]*AC[0] + AC[1]*AC[1];

	  top_data[pool_index] = 0;
	  argmax_data[pool_index] = -1;
          for (int h = topMost; h < bottomMost+1; ++h) {
            for (int w = leftMost; w < rightMost+1; ++w) {
              Dtype AP[2];
              AP[0] = w - P[0];
              AP[1] = h - P[1];
              Dtype ABAP = AB[0]*AP[0] +AB[1]*AP[1];
              Dtype ACAP = AC[0]*AP[0] + AC[1]*AP[1];
              if(ABAB>ABAP&&ABAP>=0&&ACAC>ACAP&&ACAP>=0){
		      	const int index = h * width_ + w;
		      	if (batch_data[index] > top_data[pool_index]) {
		        		top_data[pool_index] = batch_data[index];
		        		argmax_data[pool_index] = index;
		      	}
              }
             }
           }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void RotateROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(RotateROIPoolingLayer);
#endif

INSTANTIATE_CLASS(RotateROIPoolingLayer);
REGISTER_LAYER_CLASS(RotateROIPooling);

}  // namespace caffe
