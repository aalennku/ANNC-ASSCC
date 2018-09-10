Requirement:
    caffe: should be compiled with "WITH_PYTHON_LAYER := 1" in "Makefile.config"
    python2: packages required: numpy, scipy, python-lmdb, pycaffe (which is compiled with caffe)
    
    To make your compiled caffe and pycaffe work, add $path_to_ur_caffe/build/tools$
    to your $PATH$ environment and $path_to_ur_caffe/python$ to your $PYTHONPATH$ environment.

    User should copy the files paviaU.mat and paviaU_gt.mat to ./data/  . 
