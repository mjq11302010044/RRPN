#include <cfloat>
#include <cmath>
//#include <cstdio>

#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void RotateROIPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data,const Dtype* info) {
    
    int imageWidth = int(info[1]*spatial_scale+0.5);
    int imageHeight = int(info[0]*spatial_scale+0.5);
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 6;
    int roi_batch_ind = bottom_rois[0];
    Dtype cx = bottom_rois[1];
    Dtype cy = bottom_rois[2];
    Dtype h = bottom_rois[3];
    Dtype w = bottom_rois[4];
    Dtype angle = bottom_rois[5]/180.0*3.1415926535;

	 //TransformPrepare
    Dtype dx = -pooled_width/2.0;
    Dtype dy = -pooled_height/2.0;
    Dtype Sx = w*spatial_scale/pooled_width;
    Dtype Sy = h*spatial_scale/pooled_height;
    Dtype Alpha = cos(angle);
    Dtype Beta = sin(angle);
    Dtype Dx = cx*spatial_scale;
    Dtype Dy = cy*spatial_scale;

    Dtype M[2][3]; 
    M[0][0] = Alpha*Sx;
    M[0][1] = Beta*Sy;
    M[0][2] = Alpha*Sx*dx+Beta*Sy*dy+Dx;
    M[1][0] = -Beta*Sx;
    M[1][1] = Alpha*Sy;
    M[1][2] = -Beta*Sx*dx+Alpha*Sy*dy+Dy;

    Dtype P[8];
    P[0] = M[0][0]*pw+M[0][1]*ph+M[0][2];
    P[1] = M[1][0]*pw+M[1][1]*ph+M[1][2];
    P[2] = M[0][0]*pw+M[0][1]*(ph+1)+M[0][2];
    P[3] = M[1][0]*pw+M[1][1]*(ph+1)+M[1][2];
    P[4] = M[0][0]*(pw+1)+M[0][1]*ph+M[0][2];
    P[5] = M[1][0]*(pw+1)+M[1][1]*ph+M[1][2];
    P[6] = M[0][0]*(pw+1)+M[0][1]*(ph+1)+M[0][2];
    P[7] = M[1][0]*(pw+1)+M[1][1]*(ph+1)+M[1][2];

    int leftMost = int(max(round(min(min(P[0],P[2]),min(P[4],P[6]))),0.0));
    int rightMost= int(min(round(max(max(P[0],P[2]),max(P[4],P[6]))),imageWidth-1.0));
    int topMost= int(max(round(min(min(P[1],P[3]),min(P[5],P[7]))),0.0));
    int bottomMost= int(min(round(max(max(P[1],P[3]),max(P[5],P[7]))),imageHeight-1.0));
    
    Dtype maxval = 0;
    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;

 	 Dtype AB[2];
    AB[0] = P[2] - P[0];
    AB[1] = P[3] - P[1];	
	 Dtype ABAB = AB[0]*AB[0] +AB[1]*AB[1];
	 Dtype AC[2];
    AC[0] = P[4] - P[0];
    AC[1] = P[5] - P[1];
    Dtype ACAC = AC[0]*AC[0] + AC[1]*AC[1];

    for (int h = topMost; h < bottomMost+1; ++h) {
      for (int w = leftMost; w < rightMost+1; ++w) {
	    Dtype AP[2];
       AP[0] = w - P[0];
       AP[1] = h - P[1];
       Dtype ABAP = AB[0]*AP[0] +AB[1]*AP[1];
       Dtype ACAP = AC[0]*AP[0] + AC[1]*AP[1];
	    if(ABAB>ABAP&&ABAP>=0&&ACAC>ACAP&&ACAP>=0){
        int bottom_index = h * width + w;
		  if (bottom_data[bottom_index] > maxval) {
		       maxval = bottom_data[bottom_index];
		       maxidx = bottom_index;
		  }
       }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename Dtype>
void RotateROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();
  const Dtype* image_info = bottom[2]->gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  RotateROIPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data,image_info);
  CUDA_POST_KERNEL_CHECK;
}



template <typename Dtype>
__global__ void RotateROIPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 6;
    int roi_batch_ind = bottom_rois[0];
    bottom_diff += (roi_batch_ind * channels + c) * height * width;

    int bottom_index = argmax_data[index];
    if(bottom_index!=-1)
    bottom_diff[bottom_index]+=top_diff[index] ;
  }
}

template <typename Dtype>
void RotateROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();
  int counter = top[0]->count();
  //NOLINT_NEXT_LINE(whitespace/operators)
  RotateROIPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(counter), CAFFE_CUDA_NUM_THREADS>>>(
     count, top_diff, argmax_data, top[0]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
  //std::cout<<top_diff[0]<<std::endl;
}

INSTANTIATE_LAYER_GPU_FUNCS(RotateROIPoolingLayer);

}  // namespace caffe
