#!/bin/sh
# -*- coding: utf-8 -*-
rm -rf ./*_lmdb
python2 generate_random_data.py
cd train
rm -f ./models/*
caffe train -solver solver.prototxt
cd ..
python2 feature_extraction_and_testing.py
