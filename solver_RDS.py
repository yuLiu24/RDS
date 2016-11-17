#!/usr/bin/env python

import numpy as np
import os
import sys
import string

# Make sure that caffe is on the python path:
caffe_root = '../caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

# base net -- follow the editing model parameters example to make
# a fully convolutional VGG16 net.
# http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/net_surgery.ipynb
#base_weights = caffe_root + 'examples/hed_release/models/hed_bsds.caffemodel'
#base_weights = '/home/tesla/caffe-20150813/models/VGGnet/VGG_ILSVRC_16_layers_full_conv.caffemodel'
base_weights = './examples/VGG_ILSVRC_16_layers_full_conv.caffemodel'

# init
#caffe.set_mode_gpu()
#caffe.set_device(0)

solver = caffe.SGDSolver(caffe_root + 'examples/RDS_release/RDS_solver.prototxt')
solver.net.set_mode_gpu()
#solver.net.set_device(0)

# do net surgery to set the deconvolution weights for bilinear interpolation
interp_layers = [k for k in solver.net.params.keys() if 'upsample' in k]
interp_surgery(solver.net, interp_layers)

# copy base weights for fine-tuning
solver.net.copy_from(base_weights)

# solve straight through -- a better approach is to define a solving loop to
# 1. take SGD steps
# 2. score the model by the test net `solver.test_nets[0]`
# 3. repeat until satisfied
solver.step(500)
