import os
os.environ['GLOG_minloglevel'] = '2'  #no caffe log output

import caffe
import numpy as np
import cv2
import time
import argparse

img_path = './test.jpg'   #picture to predict
picsize = 224

root = './data/'
net_path = root + 'deploy_mobilenetv2.prototxt'
weight_path = root + 'place365.caffemodel'
mean_path = root + 'places365CNN_mean.npy'


scenes = []
net = None

def parse_args():
   
    parser = argparse.ArgumentParser(description='inference picture')
    parser.add_argument('--solver', dest='solver',
                        help='solver',
                        default='solver_MobileNetV2.prototxt', type=str)
    parser.add_argument('--weights', dest='weights',
                        help='weights',
                        default=None, type=str)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

def predict(img_path):

    img_mean = np.load(mean_path)  #(C,H,W)
    img = cv2.resize(cv2.imread(img_path), (img_mean.shape[1], img_mean.shape[1]))
    img = img - img_mean.transpose(1,2,0) #(H,W,C)
    img = cv2.resize(img, (picsize, picsize)).transpose(2,0,1) #(C,H,W)
    net.blobs['data'].data[...] = img[...]

    start_time = time.time()
    net.forward()
    end_time = time.time()
    processing_time = end_time - start_time
    print ('processing time: %d ms') %(processing_time*1000)

    prob = net.blobs['prob'].data[...].flatten()
    top5_idx = np.argsort(prob)[::-1][:5]
    print(str(img_path))
    for idx in top5_idx:
        print('{}:{}'.format(prob[idx],scenes[idx]))

if __name__ == '__main__':
    
    with open(root+'scenes.txt') as f:
        for s in f.readlines():
          scenes.append(s.rstrip('\n'))
          
    net = caffe.Net(net_path, weight_path, caffe.TEST)
    predict(img_path)
              