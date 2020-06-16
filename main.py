from __future__ import print_function
import argparse
import os
import sys
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# import shapely.geometry import Point
# import shapely.geometry import LineString

matplotlib.use("TkAgg")


class Peak:
    def __init__(self, mu, sigma, amp):
        self.mu = mu
        self.sigma = sigma
        self.amp = amp

class Model:
    def __init__(self, n, data):
        self.SIGMA = 0.1
        self.AMP = 10
        self.xrange = range(-256, 256)
        self.yrange = range(-256, 256)

        self.n = n
        self.data = data

        if n < 3:
            raise ValueError("n is too small")
        if n < 5:
            self.init_x = [len(self.xrange)*1/4, len(self.xrange)*3/4, len(self.xrange)*1/4, len(self.xrange)*3/4]
            self.init_y = [len(self.yrange)*1/4, len(self.yrange)*1/4, len(self.yrange)*3/4, len(self.yrange)*3/4]
        elif n < 7:
            self.init_x = [len(self.xrange)*1/4, len(self.xrange)*3/4, len(self.xrange)*1/4, len(self.xrange)*3/4, len(self.xrange)*1/4, len(self.xrange)*3/4]
            self.init_y = [len(self.yrange)*1/6, len(self.yrange)*1/6, len(self.yrange)*3/6, len(self.yrange)*3/6, len(self.yrange)*5/6, len(self.yrange)*5/6]
        elif n < 9:
            self.init_x = [len(self.xrange)*1/6, len(self.xrange)*3/6, len(self.xrange)*5/6, len(self.xrange)*1/6, len(self.xrange)*3/6, len(self.xrange)*5/6, len(self.xrange)*1/6, len(self.xrange)*3/6, len(self.xrange)*5/6]
            self.init_y = [len(self.yrange)*1/6, len(self.yrange)*1/6, len(self.yrange)*1/6, len(self.yrange)*3/6, len(self.yrange)*3/6, len(self.yrange)*3/6, len(self.yrange)*5/6, len(self.yrange)*5/6, len(self.yrange)*5/6]
        else:
            raise ValueError("n is too big")
        self.init_x = [elem - 256 for elem in self.init_x]
        self.init_y = [elem - 256 for elem in self.init_y]

        self.peaks = []
        for i in range(n):
            self.peaks.append(Peak([self.init_x[i], self.init_y[i]], self.SIGMA, self.AMP))

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
                d.append(float(line.split()[1]))
            data.append(d)
    
    print("Parsing inputs is finished...")
    npdata = np.array(data)
    print(angle)
    print(npdata)
    return (npdata, angle)

def infer_peak(npdata):
    peaks = []
    for data in npdata:
        threshold = np.average(data) * 3
        infer_peaks = []
        grad1 = []
        grad2 = []

        for (i, val) in enumerate(data):
            if i >= 2:
                grad1.append(data[i] - data[i-1])
                grad2.append((data[i] - data[i-2]) / 2)
            else:
                grad1.append(0)
                grad2.append(0)

            if i >= 4:
                if data[i-2] > threshold and grad1[i-2] > 0 and grad1[i-1] > 0 and grad1[i] < 0:
                    infer_peaks.append((i-1-256, data[i-1]))
        
        peaks.append(infer_peaks)

    return peaks

def infer_n(peaks):
    return max([len(p) for p in peaks])

def generate_cdata(data, i_peaks, model):
    cdata = []

    SIGMA = 0.01

    x = np.arange(model.xrange[0], model.xrange[-1], 0.1)
    # yrange = np.arange(model.yrange[0], model.yrange[-1], 0.1)

    for i, d in enumerate(data):
        peaks = i_peaks[i]
        cd = np.zeros(len(x))
        for j, p in enumerate(peaks):
            print(p)
            cd = cd + p[1]*np.exp(-(x-p[0])**2/2*SIGMA)
        print(cd)
        # cdata.append(np.append(x.reshape(len(x), 1), cd.reshape(len(cd), 1), axis=1))
        cdata.append(cd)

    return cdata

def grad(x1, x2, d):
    return (x1-x2)/d

def proj_angle(a, angle):
    b = [1, math.tan(math.radians(angle))]
    return np.dot(a, b)

def infer_mu(data, cdata, angle, i_peaks, model):
    while True:
        for p in model.peaks:
            mu = p.mu
            for i, ang in enumerate(angle):
                cd = cdata[i]
                xx = proj_angle(mu, ang)

            
            
def update_vis(data, cdata, angle, i_peaks, model):
    global fig2, fig3, axes2, axes3

    xn = 2
    yn = len(angle)/2

    fig2, axes2 = plt.subplots(xn, yn)
    x = np.arange(model.xrange[0], model.xrange[-1], 0.1)
    for i, cd in enumerate(cdata):
        axes2[i/yn][i%yn].plot(x, cd, 'k')
        for p in model.peaks:
            xx = proj_angle(p.mu, angle[i])
            print(angle[i])
            print(xx)
            axes2[i/yn][i%yn].plot(xx, cd[int((xx+256)/0.1)], 'go')
    for i, peaks in enumerate(i_peaks):
        for p in peaks:
            axes2[i/yn][i%yn].plot(p[0], p[1], 'rx')


    fig3, axes3 = plt.subplots(1, 1)
    axes3.plot([p.mu[0] for p in model.peaks], [p.mu[1] for p in model.peaks], 'go')
    axes3.set_xlim([-256, 256])
    axes3.set_ylim([-256, 256])

    return

def visualize(data, cdata, angle, i_peaks, model):
    global fig, axes

    xn = 2
    yn = len(angle)/2
    fig, axes = plt.subplots(xn, yn)
    for i, d in enumerate(data):
        axes[i/yn][i%yn].plot(model.xrange, d, 'k')
    for i, peaks in enumerate(i_peaks):
        for p in peaks:
            axes[i/yn][i%yn].plot(p[0], p[1], 'rx')

    update_vis(data, cdata, angle, i_peaks, model)

    plt.show(block=True)

    return

def main():
    args = parse_arguments()
    (data, angle) = parse_inputs(args)
    i_peaks = infer_peak(data)
    if not args.num_of_peaks:
        n = infer_n(i_peaks)
        print(n)
    else:
        n = args.num_of_peaks
    # model = Model(n)
    model = Model(n, data)
    cdata = generate_cdata(data, i_peaks, model)
    visualize(data, cdata, angle, i_peaks, model)
    # infer_mu(data, cdata, angle, i_peaks, model)
    

    return

if __name__ == "__main__":
    main()
