### Paper source

# Arbitrary-Oriented Scene Text Detection via Rotation Proposals

https://arxiv.org/abs/1703.01086

### License

RRPN is released under the MIT License (refer to the LICENSE file for details). This project is for research purpose only, further use for RRPN should contact authors.

### Citing RRPN

If you find RRPN useful in your research, please consider citing:

    @article{Jianqi17RRPN,
        Author = {Jianqi Ma and Weiyuan Shao and Hao Ye and Li Wang and Hong Wang and Yingbin Zheng and Xiangyang Xue},
        Title = {Arbitrary-Oriented Scene Text Detection via Rotation Proposals},
        journal = {arXiv:1703.01086},
        Year = {2017}
    }

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```
  You can download my [Makefile.config](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/Makefile.config) for reference.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Requirements: hardware

1. For training the end-to-end version of RRPN with VGG16, 4~5G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Clone the RRPN repository
  ```Shell
  # git clone https://github.com/mjq11302010044/RRPN.git
  ```

2. We'll call the directory that you cloned RRPN into `RRPN_ROOT`

  
3. Build the Cython modules
    ```Shell
    cd $RRPN_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $RRPN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j4 && make pycaffe
    ```

5. Download pre-computed RRPN detectors
    ```Shell
    download link: TODO
    
    ```

   Then move the model into `$RRPN_ROOT/data/faster_rcnn_models`.

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $RRPN_ROOT
python ./tools/rotation_demo.py
```
The txt results will be saved in `$RRPN_ROOT/result`

### Beyond the demo: installation for training and testing models

You can using the function `get_rroidb()` in `$RRPN_ROOT/lib/rotation/data_extractor.py` to manage your training data:

	Each training sample should be managed in a python dict like:

	im_info = {
		'gt_classes': # Set to 1(Only text)
		'max_classes': # Set to 1(Only text)
		'image': # image path to access
		'boxes': # ground truth box
		'flipped' : # Flip an image or not (Not implemented)
		'gt_overlaps' : # overlap of a class(text)
		'seg_areas' : # area of an ground truth region
		'height': # height of an image data
		'width': # width of an image data
		'max_overlaps' : # max overlap with each gt-proposal
		'rotated': # Random angle to rotate an image
	}


### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $RRPN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

Then you can train RRPN by typing:
```
./experiment/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] rrpn
```
[NET] usually takes `VGG16`

Trained RRPN networks are saved under:(We set the directory to './' by default.)

```
./
```
One can change the directory in variable `output_dir` in `$RRPN_ROOT/tools/train_net.py`

Any question about this project please send message to Jianqi Ma(mjq11302010044@gmail.com), and enjoy i
