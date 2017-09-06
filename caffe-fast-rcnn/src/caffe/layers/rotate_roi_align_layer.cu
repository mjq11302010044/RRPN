#include <cfloat>
#include <cmath>
//#include <cstdio>

#include "caffe/fast_rcnn_layers.hpp"
#include "stdio.h"

using std::max;
using std::min;

namespace caffe {

//Definite equinox
template <typename Dtype>
__device__ inline float DexX(const Dtype* bottom_rois, int i_int, int j_int, const int pooled_height_int, const int pooled_width_int) {

  Dtype i = float(i_int);
  Dtype j = float(j_int);
  Dtype pooled_width = float(pooled_width_int);
  Dtype pooled_height = float(pooled_height_int);

  return (pooled_height - i) / pooled_height * (
	(pooled_width - j) / pooled_width * bottom_rois[1] + j / pooled_width * bottom_rois[3]) + i / pooled_height * (
	(pooled_width - j) / pooled_width * bottom_rois[7] + j / pooled_width * bottom_rois[5]);
}

template <typename Dtype>
__device__ inline float DexY(const Dtype* bottom_rois, int i_int, int j_int, const int pooled_height_int, const int pooled_width_int) {

  Dtype i = float(i_int);
  Dtype j = float(j_int);
  Dtype pooled_width = float(pooled_width_int);
  Dtype pooled_height = float(pooled_height_int);

  return (pooled_width - j) / pooled_width * (
	(pooled_height - i) / pooled_height * bottom_rois[2] + i / pooled_height * bottom_rois[8]) + j / pooled_width * (
	(pooled_height - i) / pooled_height * bottom_rois[4] + i / pooled_height * bottom_rois[6]);
}

template <typename Dtype>
__device__ inline Dtype cross_mul(Dtype *pt1,Dtype * pt2,Dtype *pt3){
  return pt2[0]*pt3[1]+pt3[0]*pt1[1]+pt1[0]*pt2[1]-pt2[0]*pt1[1]-pt3[0]*pt2[1]-pt1[0]*pt3[1];
}

template <typename Dtype>
__device__ inline bool inpoly(Dtype pt_x, Dtype pt_y, Dtype * pts) {
  bool flag = true;
  int cur_sign;
  Dtype pt[2];
  pt[0] = pt_x;
  pt[1] = pt_y;
  int sign;
  for(int i = 0 ;i<4;i++){
     Dtype val = cross_mul(pts+i*2,pts+((i+1)%4*2),pt);
     if(val<0.0f){
        cur_sign = -1;
     }else if(val>0.0f){
        cur_sign = 1;
     }else{
        cur_sign =0;
     }
     if(cur_sign !=0){
        if(flag){
            flag = false;
            sign = cur_sign;
        }else{
            if(sign!=cur_sign) return false;
        }
     }
  }
  return true;
}



template <typename Dtype>
__global__ void RotateROIAlignForward(const int nthreads, const Dtype* bottom_data,
    const Dtype or_spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, Dtype* con_idx_x, Dtype* con_idx_y,const Dtype* info) {

    // The real spatial_scale should be depended on the true scale

    Dtype im_height = info[0];
    Dtype im_width = info[1];
    Dtype spatial_scale_h = float(height) / im_height;
    Dtype spatial_scale_w = float(width) / im_width;

    //Dtype spatial_scale = (spatial_scale_w + spatial_scale_h) / 2.0;

    int imageWidth = int(info[1]*spatial_scale_w+0.5);
    int imageHeight = int(info[0]*spatial_scale_h+0.5);
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // The input boxes are ready in the form of 4 pts in a array of length 8

    bottom_rois += n * 9;
    int roi_batch_ind = bottom_rois[0];
    //Dtype cx = bottom_rois[1];
    //Dtype cy = bottom_rois[2];
    //Dtype h = bottom_rois[3];
    //Dtype w = bottom_rois[4];
    //Dtype angle = bottom_rois[5]/180.0*3.1415926535;



	//TransformPrepare
    //Dtype dx = -pooled_width/2.0;
    //Dtype dy = -pooled_height/2.0;
    //Dtype Sx = w*spatial_scale/pooled_width;
    //Dtype Sy = h*spatial_scale/pooled_height;
    //Dtype Alpha = cos(angle);
    //Dtype Beta = sin(angle);
    //Dtype Dx = cx*spatial_scale;
    //Dtype Dy = cy*spatial_scale;

    //Dtype M[2][3];
    //M[0][0] = Alpha*Sx;
    //M[0][1] = Beta*Sy;
    //M[0][2] = Alpha*Sx*dx+Beta*Sy*dy+Dx;
    //M[1][0] = -Beta*Sx;
    //M[1][1] = Alpha*Sy;
    //M[1][2] = -Beta*Sx*dx+Alpha*Sy*dy+Dy;

    // order of lt, rt, rb, lb
    Dtype P[8];
    P[0] = DexX(bottom_rois, ph, pw, pooled_height, pooled_width) * spatial_scale_w;
    P[1] = DexY(bottom_rois, ph, pw, pooled_height, pooled_width) * spatial_scale_h;
    P[2] = DexX(bottom_rois, ph, pw + 1, pooled_height, pooled_width) * spatial_scale_w;
    P[3] = DexY(bottom_rois, ph, pw + 1, pooled_height, pooled_width) * spatial_scale_h;
    P[4] = DexX(bottom_rois, ph + 1, pw + 1, pooled_height, pooled_width) * spatial_scale_w;
    P[5] = DexY(bottom_rois, ph + 1, pw + 1, pooled_height, pooled_width) * spatial_scale_h;
    P[6] = DexX(bottom_rois, ph + 1, pw, pooled_height, pooled_width) * spatial_scale_w;
    P[7] = DexY(bottom_rois, ph + 1, pw, pooled_height, pooled_width) * spatial_scale_h;

    //int leftMost = int(max(round(min(min(P[0],P[2]),min(P[4],P[6]))),0.0));
    //int rightMost= int(min(round(max(max(P[0],P[2]),max(P[4],P[6]))),imageWidth-1.0));
    //int topMost= int(max(round(min(min(P[1],P[3]),min(P[5],P[7]))),0.0));
    //int bottomMost= int(min(round(max(max(P[1],P[3]),max(P[5],P[7]))),imageHeight-1.0));

    // Exact position on feature map in type float
    Dtype leftMost = fmax(fmin(fmin(P[0],P[2]),fmin(P[4],P[6])),0.0);
    Dtype rightMost = fmin(fmax(fmax(P[0],P[2]),fmax(P[4],P[6])),imageWidth-1.0);
    Dtype topMost = fmax(fmin(fmin(P[1],P[3]),fmin(P[5],P[7])),0.0);
    Dtype bottomMost = fmin(fmax(fmax(P[1],P[3]),fmax(P[5],P[7])),imageHeight-1.0);

    float maxval = 0.0;

    float max_con_x = -1.0;
    float max_con_y = -1.0;

    bottom_data += (roi_batch_ind * channels + c) * height * width;

    //Dtype AB[2];
    //AB[0] = P[2] - P[0];
    //AB[1] = P[3] - P[1];	
    //Dtype ABAB = AB[0]*AB[0] + AB[1]*AB[1];
    //Dtype AC[2];
    //AC[0] = P[4] - P[0];
    //AC[1] = P[5] - P[1];
    //Dtype ACAC = AC[0]*AC[0] + AC[1]*AC[1];

    Dtype h = topMost;
    
    while (h < bottomMost+1) {
      Dtype w = leftMost;
      while (w < rightMost+1) {
           
           if(inpoly(w, h, P)){
               //Performing blinear interpolation
               int bin_xs = int(floor(w));
               int bin_ys = int(floor(h));

               float rx = w - floor(w);
               float ry = h - floor(w);

               float wlt = (1.0 - rx) * (1.0 - ry);
               float wrt = rx * (1.0 - ry);
               float wrb = rx * ry;
               float wlb = (1.0 - rx) * ry;

               float inter_val = 0.0;

               int min_x = min(max(bin_xs, 0), width - 1);
               int min_y = min(max(bin_ys, 0), height - 1);
               int max_x = max(min(bin_xs + 1, width - 1), 0);
               int max_y = max(min(bin_ys + 1, height - 1), 0);

               int lt = min_y * width + min_x;
               int rt = min_y * width + max_x;
               int rb = max_y * width + max_x;
               int lb = max_y * width + max_x;               

               inter_val += bottom_data[lt] * wlt;
               inter_val += bottom_data[rt] * wrt;
               inter_val += bottom_data[rb] * wrb;
               inter_val += bottom_data[lb] * wlb;
               
               //inter_val = bottom_data[bin_ys * width + bin_xs];

               if (inter_val > maxval) {
                   maxval = inter_val;
                   
                   max_con_x = w;
                   max_con_y = h;
               }
          }
       
          w = w + 1.0;
      }
      h = h + 1.0;
    }
     
 

    top_data[index] = maxval;
    con_idx_x[index] = max_con_x;
    con_idx_y[index] = max_con_y;
  }
}

template <typename Dtype>
void RotateROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();

  const Dtype* cpu_bottom_data = bottom[0]->cpu_data();

  std::cout<<bottom[0]->count()<<std::endl;
  std::cout<<bottom[1]->count()<<std::endl;
  std::cout<<bottom[2]->count()<<std::endl;
  Dtype* top_data = top[0]->mutable_gpu_data();
  //int* argmax_data = max_idx_.mutable_gpu_data();
  Dtype* con_idx_x = continuous_idx_x.mutable_gpu_data();
  Dtype* con_idx_y = continuous_idx_y.mutable_gpu_data();
  std::cout<<"cpu_bottom_data"<<std::endl;
  std::cout<<cpu_bottom_data[1]<<std::endl;
  const Dtype* image_info = bottom[2]->gpu_data();
  const Dtype* cpu_image_info = bottom[2]->cpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)

  std::cout<<"spatial_scale_"<<std::endl;
  std::cout<<spatial_scale_<<std::endl;
  std::cout<<(height_ / cpu_image_info[0])<<std::endl;

  std::cout<<count<<std::endl;
  std::cout<<bottom[0]->count()<<std::endl;
  std::cout<<bottom[1]->count()<<std::endl;
  std::cout<<bottom[2]->count()<<std::endl;
  

  RotateROIAlignForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data, con_idx_x, con_idx_y, image_info);
  CUDA_POST_KERNEL_CHECK;
 
  //const Dtype* top_gpu_data = top[0]->gpu_data();

  //std::cout<<top_gpu_data[0]<<std::endl;

}



template <typename Dtype>
__global__ void RotateROIAlignBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* con_idx_x, const Dtype* con_idx_y, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* backbone_diff, Dtype* proposal_diff,
    const Dtype* bottom_data, const Dtype* bottom_rois, const Dtype* info) {
  CUDA_KERNEL_LOOP(index, nthreads) {

    //backbone_diff is original bottom_diff && argmax_data decomposites to to parts of continuous coodinations
    //And now we have a new branch to perform backprop

    

    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    Dtype im_height = info[0];
    Dtype im_width = info[1];
    Dtype spatial_scale_h = float(height) / im_height;
    Dtype spatial_scale_w = float(width) / im_width;
    
    //Take an offset
    bottom_rois += n * 9;
    proposal_diff += n * 9;

    

    int roi_batch_ind = bottom_rois[0];
    backbone_diff += (roi_batch_ind * channels + c) * height * width;
    bottom_data += (roi_batch_ind * channels + c) * height * width;

    //////////////////// backbone branch //////////////////////
    //Performing backprop for blinear interpolation

    Dtype w = con_idx_x[index];
    Dtype h = con_idx_y[index];

    int bin_xs = int(floor(w));
    int bin_ys = int(floor(h));

    Dtype rx = w - float(bin_xs);
    Dtype ry = h - float(bin_ys);

    Dtype wlt = (1.0 - rx) * (1.0 - ry);
    Dtype wrt = rx * (1.0 - ry);
    Dtype wrb = rx * ry;
    Dtype wlb = (1.0 - rx) * ry;

    //Dtype inter_val = 0;

    int min_x = min(max(bin_xs, 0), width - 1);
    int min_y = min(max(bin_ys, 0), height - 1);
    int max_x = max(min(bin_xs + 1, width - 1), 0);
    int max_y = max(min(bin_ys + 1, height - 1), 0);

    //if(bin_xs >= 0 && bin_ys >= 0) {
    backbone_diff[min_y * width + min_x] += wlt * top_diff[index];
    //}
    //if(bin_xs + 1 < width && bin_ys >= 0) {
    backbone_diff[min_y * width + max_x] += wrt * top_diff[index];
    //}
    //if(bin_xs + 1 < width && bin_ys + 1 < height) {
    backbone_diff[max_y * width + max_x] += wrb * top_diff[index];
    //}
    //if(bin_xs >= 0 && bin_ys + 1 < height) {
    backbone_diff[max_y * width + min_x] += wlb * top_diff[index];
    //}

    

    ///////////////////////////////////////////////////////////////////////

    
    
    /////////////////////////  proposal branch  ///////////////////////////

    // pick the value from feature map when the coods are inside boundaries
    //Dtype val_lt = (bin_xs >= 0 && bin_ys >= 0) ? bottom_data[bin_ys * width + bin_xs] : 0.0;
    //Dtype val_rt = (bin_xs + 1 < width && bin_ys >= 0) ? bottom_data[bin_ys * width + (bin_xs + 1)] : 0.0;
    //Dtype val_rb = (bin_xs + 1 < width && bin_ys + 1 < height) ? bottom_data[(bin_ys + 1) * width + (bin_xs + 1)] : 0.0;
    //Dtype val_lb = (bin_xs >= 0 && bin_ys + 1 < height) ? bottom_data[(bin_ys + 1) * width + bin_xs] : 0.0;
    
    Dtype val_lt = bottom_data[min_y * width + min_x];
    Dtype val_rt = bottom_data[min_y * width + max_x];
    Dtype val_rb = bottom_data[max_y * width + max_x];
    Dtype val_lb = bottom_data[max_y * width + min_x];

    // Compute the loss of h & w on pts of bilinear interpolation

    Dtype d_wlt_w = -(1.0 - h + bin_ys);
    Dtype d_wlt_h = -(1.0 - w + bin_xs);

    Dtype d_wrt_w = (1.0 - h + bin_ys);
    Dtype d_wrt_h = -(w - bin_xs);

    Dtype d_wrb_w = (h - bin_ys);
    Dtype d_wrb_h = (w - bin_xs);

    Dtype d_wlb_w = -(h - bin_ys);
    Dtype d_wlb_h = (1 - w + bin_xs);

    Dtype dw = d_wlt_w * val_lt + d_wrt_w * val_rt + d_wrb_w * val_rb + d_wlb_w * val_lb;
    Dtype dh = d_wlt_h * val_lt + d_wrt_h * val_rt + d_wrb_h * val_rb + d_wlb_h * val_lb;

    Dtype loss_w = dw * top_diff[index];
    Dtype loss_h = dh * top_diff[index];
    
    // order of lt, rt, rb, lb
    Dtype P[8];
    P[0] = DexX(bottom_rois, ph, pw, pooled_height, pooled_width) * spatial_scale_w;;
    P[1] = DexY(bottom_rois, ph, pw, pooled_height, pooled_width) * spatial_scale_h;;
    P[2] = DexX(bottom_rois, ph, pw + 1, pooled_height, pooled_width) * spatial_scale_w;;
    P[3] = DexY(bottom_rois, ph, pw + 1, pooled_height, pooled_width) * spatial_scale_h;
    P[4] = DexX(bottom_rois, ph + 1, pw + 1, pooled_height, pooled_width) * spatial_scale_w;;
    P[5] = DexY(bottom_rois, ph + 1, pw + 1, pooled_height, pooled_width) * spatial_scale_h;
    P[6] = DexX(bottom_rois, ph + 1, pw, pooled_height, pooled_width) * spatial_scale_w;;
    P[7] = DexY(bottom_rois, ph + 1, pw, pooled_height, pooled_width) * spatial_scale_h;
    
    Dtype loss_P[8];
    //backprop to the pts of pooling bin
    loss_P[0] = (P[0] >= 0.0 && P[0] < P[2] && P[0] < P[4] && P[0] < P[6]) ? loss_w : 0.0;
    loss_P[1] = (P[1] >= 0.0 && P[1] < P[3] && P[1] < P[5] && P[1] < P[7]) ? loss_h : 0.0;

    loss_P[2] = (P[2] >= 0.0 && P[2] < P[0] && P[2] < P[4] && P[2] < P[6]) ? loss_w : 0.0;
    loss_P[3] = (P[3] >= 0.0 && P[3] < P[1] && P[3] < P[5] && P[3] < P[7]) ? loss_h : 0.0;

    loss_P[4] = (P[4] >= 0.0 && P[4] < P[0] && P[4] < P[2] && P[4] < P[6]) ? loss_w : 0.0;
    loss_P[5] = (P[5] >= 0.0 && P[5] < P[1] && P[5] < P[3] && P[5] < P[7]) ? loss_h : 0.0;

    loss_P[6] = (P[6] >= 0.0 && P[6] < P[0] && P[6] < P[2] && P[6] < P[4]) ? loss_w : 0.0;
    loss_P[7] = (P[7] >= 0.0 && P[7] < P[1] && P[7] < P[3] && P[7] < P[5]) ? loss_h : 0.0;

    int trigger_w = 0;
    int trigger_h = 0;

    //find x & y position
    for(int i = 0;i < 4;i++) {
        if(fabs(loss_P[i*2]) > 0.0) {
            if(i*2 == 2 || i*2 == 4) {
                trigger_w = 1;
            }
        }
        if(fabs(loss_P[i*2+1]) > 0.0) {
            if(i*2 == 5 || i*2 == 7) {
                trigger_h = 1;
            }
        }
    }

    int h_idx = (trigger_h == 1) ? ph + 1 : ph;
    int w_idx = (trigger_w == 1) ? pw + 1 : pw;

    proposal_diff[1] += (float(pooled_height - h_idx) / pooled_height) * (float(pooled_width - w_idx) / pooled_width) * loss_w;
    proposal_diff[2] += (float(pooled_width - w_idx) / pooled_width) * (float(pooled_height - h_idx) / pooled_height) * loss_h;

    proposal_diff[3] += (float(pooled_height - h_idx) / pooled_height) * (float(w_idx) / pooled_width) * loss_w;
    proposal_diff[4] += (float(w_idx) / pooled_width) * (float(pooled_height - h_idx) / pooled_height) * loss_h;

    proposal_diff[5] += (float(h_idx) / pooled_height) * (float(w_idx) / pooled_width) * loss_w;
    proposal_diff[6] += (float(w_idx) / pooled_width) * (float(h_idx) / pooled_height) * loss_h;

    proposal_diff[7] += (float(h_idx) / pooled_height) * (float(pooled_width - w_idx) / pooled_width) * loss_w;
    proposal_diff[8] += (float(pooled_width - w_idx) / pooled_width) * (float(h_idx) / pooled_height) * loss_h;

    ///////////////////////////////////////////////////////////////////////

    //int bottom_index = argmax_data[index];
    //if(bottom_index!=-1)
    //backbone_diff[bottom_index]+=top_diff[index];
    /**/
  }
}

template <typename Dtype>
void RotateROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
 
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  //std::cout<<top_diff[0]<<std::endl;
  Dtype* backbone_diff = bottom[0]->mutable_gpu_diff();
  Dtype* proposal_diff = bottom[1]->mutable_gpu_diff();
  
  //const int count = bottom[0]->count();
  const int backbone_count = bottom[0]->count();
  const int proposal_count = bottom[1]->count();
  
  //caffe_gpu_set(count, Dtype(0.), backbone_diff);
  caffe_gpu_set(backbone_count, Dtype(0.), backbone_diff);
  caffe_gpu_set(proposal_count, Dtype(0.), proposal_diff);
  
  //const int* argmax_data = max_idx_.gpu_data();
  const Dtype* con_idx_x = continuous_idx_x.gpu_data();
  const Dtype* con_idx_y = continuous_idx_y.gpu_data();
  
  const Dtype* cpu_con_idx_x = continuous_idx_x.cpu_data();
  const Dtype* cpu_con_idx_y = continuous_idx_y.cpu_data();

  std::cout<<cpu_con_idx_x[0]<<std::endl;
  std::cout<<cpu_con_idx_x[1]<<std::endl;

  const Dtype* image_info = bottom[2]->gpu_data();

  int counter = top[0]->count();
  std::cout<<counter<<std::endl;
  //NOLINT_NEXT_LINE(whitespace/operators)
  RotateROIAlignBackward<Dtype><<<CAFFE_GET_BLOCKS(counter), CAFFE_CUDA_NUM_THREADS>>>(
     backbone_count, top_diff, con_idx_x, con_idx_y, top[0]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, backbone_diff, proposal_diff, bottom_data, bottom_rois, image_info);
  CUDA_POST_KERNEL_CHECK;
  
  //const Dtype* cpu_arr = top[0]->cpu_diff();
  std::cout<<"Backprop down"<<std::endl;
}


INSTANTIATE_LAYER_GPU_FUNCS(RotateROIAlignLayer);

}  // namespace caffe
