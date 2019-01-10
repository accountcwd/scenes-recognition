import os
os.environ['GLOG_minloglevel'] = '2'  #no caffe log output

import caffe
import numpy as np
import cv2
import time
import argparse

img_path = './test.jpg'   #picture to predict
picsize = 224
threshold = 0.95
root = './data/'
net_path = root + 'deploy_attr_MobileNetV2.prototxt'
weight_path = root + 'place365&SUN_attribute.caffemodel'
mean_path = root + 'mean_attribute.npy'

scenes = []
attribute = []
net = None


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

    print(str(img_path) + ' sences:')
    prob = net.blobs['prob'].data[...].flatten()
    top5_idx = np.argsort(prob)[::-1][:5]
    for idx in top5_idx:
        print('{}:{}'.format(prob[idx], scenes[idx]))

    print('attritue:')
    attr_scores = net.blobs['fc7_102'].data[...].flatten()
    top10_idx = np.argsort(attr_scores)[::-1][:10]
    for idx in top10_idx:
        if attr_scores[idx] < threshold : break
        print('{}:{}'.format(attr_scores[idx], attribute[idx]))

if __name__ == '__main__':

    with open(root+'scenes.txt') as f:
        for s in f.readlines():
           scenes.append(s.rstrip('\n'))  

    with open(root+'attribute.txt') as f:
        for s in f.readlines():
            attribute.append(s.rstrip('\n'))
            
    net = caffe.Net(net_path, weight_path, caffe.TEST)
    predict(img_path)
                