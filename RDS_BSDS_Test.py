#!/usr/bin/env python

import numpy as np
import os
import sys
import string
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = 'hed_release-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#caffe.set_phase_test()
#caffe.set_mode_gpu()

# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)

def main(argv):

	# set net config
	net = caffe.Classifier(caffe_root+'examples/RDS_release/RDS_deploy.prototxt',
		               caffe_root+'examples/RDS_release/RDSwithSE_BSDS.caffemodel',
                               channel_swap=(2,1,0), raw_scale=255,gpu=True)       
         
	net.set_phase_test()
	net.set_mode_gpu()

	# get all images in the directory
	#images = os.listdir(sys.argv[1])
	#images.sort()
        dirpath = '/home/tesla/Datasets/BSDS500/BSDS500/data/images/test'

        # get validation image names
	#valfile = open('/home/titan/Datasets/PASCAL-VOC/development_kit/VOC2012/val_seg.txt','r')
        valfile = open('/home/tesla/Datasets/BSDS500/BSDS500/data/images/test_imgnames.txt','r')

        # variable 
	feat = []
        score = []
        nimg = 200      

	for counters in range(0,nimg):
            imgarr = valfile.readline().split()
            img = imgarr[0]
	    path = dirpath + '/' + img

	    # open a file to save features with append mode
	    fw = open(caffe_root + 'examples/RDS_release/RDS_SE_New/'+img+'.txt','w')

	    #exclude . and ..
	    if(img[0] == '.'):
		pass
	    else:
		print counters+1, img
		input_image = caffe.io.load_image(path) # load image
                #print input_image.shape

                # reshape input size
                net.blobs['data'].reshape(1, 3, input_image.shape[0], input_image.shape[1])

                # mean file
                #net.set_mean('data', np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'))

                # 'data' is the input blob name in the model definition, so we preprocess for that input.
                caffe_input = np.asarray([net.preprocess('data', input_image)])


                # mean pixel 
                caffe_input[:,0,:,:] -= 104.00698793
                caffe_input[:,1,:,:] -= 116.66876762
                caffe_input[:,2,:,:] -= 122.67891434

		# call forward, not predict!! 
		net.forward(data=caffe_input)  
  
                # Visulizing feature maps               
                #for k, v in net.blobs.items():
                    #print (k, v.data.shape) 
                #for k, v in net.params.items():
                    #print (k, v[0].data.shape) 
	        # show inter layers
                #plt.imshow(net.deprocess('data', net.blobs['data'].data[0]))
                #plt.show()

                # Compute the final score
                feat = net.blobs['sigmoid-upscore-fuse'].data[0,0]

                for nrow in range(0,input_image.shape[0]):
                    feat[nrow].tofile(fw,sep=" ", format="%s")
                    fw.write('\n')
		
	        # close the out stream
	        fw.close()

        # close file
        valfile.close()

if __name__ == '__main__':
    main(sys.argv)


