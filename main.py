from __future__ import print_function
import argparse
import numpy as np
import os
import sys


class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class peak:
    def __init__(self, mu, sigma, amp):
        self.mu = mu
        self.sigma = sigma
        self.amp = amp

class model:
    def __init__(self, n, data):
        self.SIGMA = 0.1
        self.AMP = 10
        self.xrange = range(-256, 256)
        self.yrange = range(-256, 256)

        self.n = n

        if n < 3:
            raise ValueError("n is too small")
        if n < 5:
            self.init_x = [len(xrange)*1/4, len(xrange)*3/4, len(xrange)*1/4, len(xrange)*3/4]
            self.init_y = [len(yrange)*1/4, len(yrange)*1/4, len(yrange)*3/4, len(yrange)*3/4]
        elif n < 7:
            self.init_x = [len(xrange)*1/4, len(xrange)*3/4, len(xrange)*1/4, len(xrange)*3/4, len(xrange)*1/4, len(xrange)*3/4]
            self.init_y = [len(yrange)*1/6, len(yrange)*1/6, len(yrange)*3/6, len(yrange)*3/6, len(yrange)*5/6, len(yrange)*5/6]
        elif n < 9:
            self.init_x = [len(xrange)*1/6, len(xrange)*3/6, len(xrange)*5/6, len(xrange)*1/6, len(xrange)*3/6, len(xrange)*5/6, len(xrange)*1/6, len(xrange)*3/6, len(xrange)*5/6]
            self.init_y = [len(yrange)*1/6, len(yrange)*1/6, len(yrange)*1/6, len(yrange)*3/6, len(yrange)*3/6, len(yrange)*3/6, len(yrange)*5/6, len(yrange)*5/6, len(yrange)*5/6]
        else:
            raise ValueError("n is too big")

        self.peaks = []
        for i in range(n):
            self.peaks.append(peak(point(init_x[i], init_y[i]), SIGMA, AMP))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=str, default=None, help="directory of input files")
    parser.add_argument("-v", "--visual", action="store_true", help="show visual process for debugging")
    parser.add_argument("-n", "--num-of-peaks", type=int, default=None, help="number of peaks")

    return parser.parse_args()

def parse_inputs(args):
    input_dir = args.input_dir
    # data = np.array([])
    data = []
    angle = []

    for filename in sorted(os.listdir(input_dir)):
        if filename.startswith("hybrid"):
            angle.append(int(filename.split("hybrid")[1].split('.')[0]))

            d = []
            fd = open(os.path.join(input_dir, filename), 'r')
            fd.next()
            print(os.path.join(input_dir, filename))
            for line in fd:
                d.append(line.split()[1])
            data.append(d)
    
    print("Parsing inputs is finished...")
    npdata = np.array(data)
    print(angle)
    print(npdata)
    return (npdata, angle)

def infer_peak(data):
    infer_peaks = []

    for i in range(data)
        

    return infer_peaks
def infer_n(data, angle):
    for i in range(angle):
        
    
    return

def main():
    args = parse_arguments()
    (data, angle) = parse_inputs(args)
    if not args.num_of_peaks:
        n = infer_n(args)

    return



if __name__ == "__main__":
    sys.exit(main())
