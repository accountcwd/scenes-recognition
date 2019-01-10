import argparse
import numpy as np
import os
from matplotlib import pyplot as plt

errors_train_path = './tmp/errors.npy'
errors_path = './tmp/errors.npy'

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a caffe network')

    parser.add_argument('--start', dest='start',
                        default=0, type=int)
    parser.add_argument('--end', dest='end',
                        default=-1, type=int)
    parser.add_argument('--display', dest='display',
                        default=100, type=int)                                               
    parser.add_argument('--interval', dest='interval',
                        default=2000, type=int)                        

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

def draw_errors():
    errors_train = np.load(errors_train_path)
    errors = np.load(errors_path)
    print(errors)
    display = args.display
    start = args.start
    start_idx = start / display
    end = errors_train.shape[0] * display if args.end == -1 else args.end
    end_idx = end / display
    print(start, end,start_idx,end_idx)
    plt.plot(np.arange(start, end, display), errors_train[start_idx : end_idx])
#    plt.plot(np.arange(start, end_iter, args.interval), errors)
    ymin = np.min(errors_train[start_idx : end_idx])
    ymax = np.max(errors_train[start_idx : end_idx])
    plt.ylim(ymin, ymax)
    plt.show()
    plt.savefig('./tmp/nperr.jpg')
    plt.clf()
    min = np.argmin(errors[46:])
    print('min ',min ,errors[min+46])
if __name__ == '__main__':
    args = parse_args()
    draw_errors()
