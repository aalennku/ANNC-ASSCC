1. Requirement:
    caffe: should be compiled with "WITH_PYTHON_LAYER := 1" in "Makefile.config"
    python2: packages required: numpy, scipy, python-lmdb, pycaffe (which is compiled with caffe)
    
    To make your compiled caffe and pycaffe work, add $path_to_ur_caffe/build/tools$
    to your $PATH$ environment and $path_to_ur_caffe/python$ to your $PYTHONPATH$ environment.

2. User should copy the files paviaU.mat and paviaU_gt.mat to ./data/  . 
3. Please cite the following paper in your publications if the codes help your research: 
@article{guo2019spectral,
	author = {A. J. X. {Guo} and F. {Zhu}},
	doi = {10.1109/TGRS.2018.2869004},
	issn = {0196-2892},
	journal = {IEEE Transactions on Geoscience and Remote Sensing},
	keywords = {Feature extraction; Training; Neurons; Hyperspectral imaging; Teﬆing; Iron; Artiﬁcial neural networks (ANN); deep learning; feature extraction; hyperspectral image classiﬁcation},
	month = {March},
	number = {3},
	pages = {1755–1767},
	title = {Spectral-Spatial Feature Extraction and Classiﬁcation by ANN Supervised With Center Loss in Hyperspectral Imagery},
	volume = {57},
	year = {2019}
}

