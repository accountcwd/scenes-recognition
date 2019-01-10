    # -*- coding: UTF-8 -*-
# caffe_root = '/home/beauty/caffe'
# import sys
# sys.path.insert(0, caffe_root + 'python')
# sys.path.append('./layer')

# from ImageServer import ImageServer

import caffe
import argparse
import numpy as np
import os
from matplotlib import pyplot as plt

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a caffe network')

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


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process.
    """

    def __init__(self, solver_prototxt,
                 pretrained_model=None):
        self.errors = np.load('./tmp/errors.npy').tolist()
        self.errorsTrain = np.load('./tmp/errorsTrain.npy').tolist()
        """Initialize the SolverWrapper."""

        # use Adam solver
        self.solver = caffe.SGDSolver(solver_prototxt)

        # load pretrained model
        if pretrained_model is not None:
            if '.caffemodel' in pretrained_model:
                print ('Loading pretrained model '
                    'weights from {:s}').format(pretrained_model)
                self.solver.net.copy_from(pretrained_model)
            elif '.solverstate' in pretrained_model:
                print ('Restore state from {:s}').format(pretrained_model)
                self.solver.restore(pretrained_model)

        # parse solver.prototxt
        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)


    def drawErrors(self):
        trainStep = self.solver_param.display
        testStep = self.solver_param.test_interval

        plt.plot(np.arange(0, len(self.errorsTrain) * trainStep, trainStep),
                                                              self.errorsTrain)
        plt.plot(np.arange(0, len(self.errors) * testStep, testStep),
                                                                   self.errors)
        ymin = np.min(self.errorsTrain)
        ymax = np.max(self.errorsTrain)

        plt.ylim(ymin, ymax)
        #plt.savefig('./errors%d.jpg' %self.solver.iter)
        plt.savefig('./iters_{:d}.jpg'.format(self.solver.iter))
        plt.clf()

    def saveErrors(self):
        np.save("./tmp/errors.npy", self.errors)
        np.save("./tmp/errorsTrain.npy", self.errorsTrain)

    def snapshot(self):
        """Take a snapshot of the network. This enables easy use at test-time.
        """
        net = self.solver.net
        filename = (self.solver_param.snapshot_prefix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)
        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)
        return filename

    def train_model(self):
        """Network training loop."""
        train_loss = 0
        train_loss_single = 0
        test_loss = 0
        test_loss_single = 0
        max_iters = self.solver_param.max_iter
        while self.solver.iter < max_iters:
            # train and update one iter
            self.solver.step(1)

            # calculate and draw train/test error
            train_loss_single += self.solver.net.blobs['loss'].data

            if self.solver.iter % self.solver_param.display == 0:
                train_loss = train_loss_single / self.solver_param.display
                self.errorsTrain.append(train_loss)
                train_loss_single = 0

            if self.solver.iter % self.solver_param.test_interval == 0:
                for test_it in range(self.solver_param.test_iter[0]):
                    self.solver.test_nets[0].forward()
                    test_loss_single += self.solver.test_nets[0].blobs['loss'].data
                test_loss = test_loss_single / self.solver_param.test_iter[0]
                self.errors.append(test_loss)
                test_loss_single = 0
                
                self.drawErrors()
                self.saveErrors()




def train_net(solver_prototxt,
              pretrained_model=None):
    """Train a Fast R-CNN network."""

    sw = SolverWrapper(solver_prototxt,
                       pretrained_model=pretrained_model)
    

    print 'Solving...'
    sw.train_model()
    print 'done solving'

    net = sw.solver.net


if __name__ == '__main__':
    args = parse_args()
    caffe.set_mode_gpu()
    caffe.set_device(0)

    solver_prototxt = args.solver #'solver_vgg16.prototxt'
    # pretrained_model = args.weights
    pretrained_model = '/home/beauty/project/places365-caffe/models/MobileNetV2_1epoch__iter_610000.solverstate'

    train_net(solver_prototxt,
              pretrained_model=pretrained_model)
              





