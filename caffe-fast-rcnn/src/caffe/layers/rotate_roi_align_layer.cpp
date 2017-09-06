#include <cfloat>
#include <cmath>

#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void RotateROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  RotateROIAlignParameter rotate_align_param = this->layer_param_.rotate_align_param();
  CHECK_GT(rotate_align_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(rotate_align_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = rotate_align_param.pooled_h();
  pooled_width_ = rotate_align_param.pooled_w();
  spatial_scale_ = rotate_align_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void RotateROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  continuous_idx_x.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  continuous_idx_y.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
}

template <typename Dtype>
void RotateROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  NOT_IMPLEMENTED;
}

template <typename Dtype>
void RotateROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(RotateROIAlignLayer);
#endif

INSTANTIATE_CLASS(RotateROIAlignLayer);
REGISTER_LAYER_CLASS(RotateROIAlign);

}  // namespace caffe
