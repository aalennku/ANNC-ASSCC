1. Requirement:
    caffe: should be compiled with "WITH_PYTHON_LAYER := 1" in "Makefile.config"
    python2: packages required: numpy, scipy, python-lmdb, pycaffe (which is compiled with caffe)
    
    To make your compiled caffe and pycaffe work, add $path_to_ur_caffe/build/tools$
    to your $PATH$ environment and $path_to_ur_caffe/python$ to your $PYTHONPATH$ environment.

2. User should copy the files paviaU.mat and paviaU_gt.mat to ./data/  . 
3. Please cite the following paper in your publications if the codes help your research: 
@article{guo2017spectral,
  title={Spectral-Spatial Feature Extraction and Classification by ANN Supervised with Center Loss in Hyperspectral Imagery},
  author={Guo, Alan JX and Zhu, Fei},
  journal={arXiv preprint arXiv:1711.07141},
  year={2017}
}
