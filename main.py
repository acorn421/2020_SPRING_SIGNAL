from __future__ import print_function
import argparse
import os
import sys
import math
import time

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
    def __init__(self, n, data, threshold):
        self.SIGMA = 1
        self.AMP = 50
        self.xrange = range(-256, 256)
        self.yrange = range(-256, 256)

        self.n = n
        self.data = data
        self.threshold = threshold

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
    th = []
    for data in npdata:
        threshold = np.average(data) * 5
        th.append(threshold)
        print("Threshold : %d" % threshold)
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
                    infer_peaks.append([i-1-256, data[i-1]])
        
        peaks.append(infer_peaks)

    return np.average(th), peaks

def infer_n(peaks):
    return max([len(p) for p in peaks])

def generate_cdata():
    global data, cdata, angle, i_peaks, model

    cdata = []

    SIGMA = 50

    x = np.arange(model.xrange[0], model.xrange[-1]+1, 0.1)
    # yrange = np.arange(model.yrange[0], model.yrange[-1], 0.1)

    for i, d in enumerate(data):
        peaks = i_peaks[i]
        cd = np.zeros(len(x))
        for j, p in enumerate(peaks):
            cd = cd + p[1]*np.exp(-(x-p[0])**2/(2*SIGMA))
        # cdata.append(np.append(x.reshape(len(x), 1), cd.reshape(len(cd), 1), axis=1))
        cdata.append(cd)

    return cdata

def cidx(xx):
    idx = int((xx+256)/0.1)
    if idx > 5109:
        return 5109
    elif idx < 0:
        return 0
    else:
        return int((xx+256)/0.1)

def gradient(xx, cd, acc):
    idx = cidx(xx)
    if idx > (5109 - acc) or idx < (0 + acc):
        return 0
    else:
        return (cd[idx+acc] - cd[idx-acc]) / (0.2 * acc)

def proj(a, angle):
    b = [math.cos(math.radians(angle)), math.sin(math.radians(angle))]
    return np.dot(a, b)

def r_proj(s, angle):
    return [s * math.cos(math.radians(angle)), s * math.sin(math.radians(angle))]

def infer_mu():
    global data, cdata, angle, i_peaks, model, finished

    ALPHA = 1
    MIN_ALPHA = 0.1
    MAX_ITERATION = 50
    MAX_CNT = 10000
    MIN_NORM = 10
    
    alpha = [ALPHA] * model.n
    gamma = 0.9
    conv_cond = 0.01
    finished = [False] * model.n
    iteration = [0] * model.n
    cnt = 0
    

    while True:
        if cnt >= MAX_CNT:
            break
        for j, p in enumerate(model.peaks):
            if cnt >= MAX_CNT:
                break
            if finished[j]:
                continue

            mu = p.mu

            grad = []
            step = [0, 0]
            for i, ang in enumerate(angle):
                cd = cdata[i]
                xx = proj(mu, ang)
                grad1 = gradient(xx, cd, 1)
                grad.append(grad1)
                s = r_proj(alpha[j] * grad1, ang)
                
                # print(alpha)
                # print(grad1)
                # print(finished)
                if cidx(mu[0] + s[0]) < 5110 or cidx(mu[1] + s[1]) < 5110:
                    step[0] += s[0]
                    step[1] += s[1]
            mu[0] += step[0]
            mu[1] += step[1]
                

            if alpha[j] > MIN_ALPHA:
                alpha[j] *= gamma
            p.mu = mu
            iteration[j] += 1
            cnt += 1
            
            if cnt % 100 == 0:
                update_vis()    
            
            # print(np.max(np.abs(grad)))
            # print(grad)
            if np.max(np.abs(grad)) < conv_cond or iteration[j] > MAX_ITERATION:
                print("[DBG] Check convergence condition of peak %d!" % j)
                flag = 0
                for i, ang in enumerate(angle):
                    cd = cdata[i]
                    xx = proj(mu, ang)
                    if cd[cidx(xx)] < model.threshold:
                        flag += 1

                if flag == 0:
                    print("[DBG] Peak %d is finished" % j)
                    print(gradient)
                    finished[j] = True
                    for k, p2 in enumerate(model.peaks):
                        if j==k or not finished[k]: continue

                        dx = p.mu[0] - p2.mu[0]
                        dy = p.mu[1] - p2.mu[1]
                        if dx**2 + dy**2 < MIN_NORM:
                            rand = np.random.rand(2)
                            r = rand[0] * 255
                            deg = rand[1] * 360
                            p.mu = [r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg))]
                            print("new mu is %s" % p.mu)
                            alpha[j] = ALPHA
                            iteration[j] = 0
                            finished[j] = False
                            break
                    


                    # for i, peaks in enumerate(i_peaks):
                    #     ang = angle[i]
                    #     xx = proj(p.mu, ang)
                    #     min = -1
                    #     minIdx = -1
                    #     for j, p2 in enumerate(peaks):
                    #         if abs(p2[0] - xx) < min:
                    #             min = p2[0] - xx
                    #             minIdx = j
                    #     del peaks[minIdx]
                    # cdata = generate_cdata()
                else:
                    rand = np.random.rand(2)
                    if flag >= (model.n-1):
                        r = rand[0] * 255
                        deg = rand[1] * 360
                        p.mu = [r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg))]
                    else:
                        r = rand[0] * 30
                        deg = rand[1] * 360
                        p.mu = [p.mu[0] + r * math.cos(math.radians(deg)), p.mu[1] + r * math.sin(math.radians(deg))]
                    print("new mu is %s" % p.mu)
                    alpha[j] = ALPHA
                    iteration[j] = 0
def generate_rdata():
    global data, cdata, angle, i_peaks, model, rdata

    rdata = []

    x = np.arange(model.xrange[0], model.xrange[-1]+1, 0.1)
    # yrange = np.arange(model.yrange[0], model.yrange[-1], 0.1)

    for i, d in enumerate(data):
        rd = np.zeros(len(x))
        ang = angle[i]
        for j, p in enumerate(model.peaks):
            mumu = proj(p.mu, ang)
            rd = rd + p.amp*np.exp(-(x-mumu)**2/(2*p.sigma))
        rdata.append(rd)

    return rdata

def calculate_pi():
    global data, cdata, rdata, angle, i_peaks, model, finished

    pi = 0
    for j, d in enumerate(data):
        rd = rdata[j]
        for i, ry in enumerate(d):
            if ry > model.threshold:
                # print(rd[i*10])
                # print(ry)
                pi += abs(rd[i*10]-ry)  
                # print(pi)

    return pi

def MCMC():
    global data, cdata, rdata, angle, i_peaks, model, finished

    MU_SIGMA1 = 1
    MU_SIGMA2 = 5
    AMP_SIGMA = 1
    SIGMA_SIGMA = 1

    cnt = 0

    generate_rdata()
    update_vis2()

    # generate_rdata()
    # update_vis2()

    # generate_rdata()
    # update_vis2()

    while True:
        generate_rdata()
        origin_pi = calculate_pi()

        for i, p in enumerate(model.peaks):
            if finished[i]:
                e1 = np.random.normal(0, MU_SIGMA1, 2)
            else:
                e1 = np.random.normal(0, MU_SIGMA2, 2)

            p.mu[0] += e1[0]
            p.mu[1] += e1[1]
            generate_rdata()
            pi_mu = calculate_pi()
            
            if pi_mu > origin_pi:
                origin_pi = pi_mu
            else:
                u = np.random.rand(1)
                if u[0] < pi_mu / origin_pi:
                    origin_pi = pi_mu
                else:
                    p.mu[0] -= e1[0]
                    p.mu[1] -= e1[1]
            
            e2 = np.random.normal(0, AMP_SIGMA, 1)
            
            p.amp += e2[0]

            if p.amp > 0:
                generate_rdata()
                pi_amp = calculate_pi()
                
                if pi_amp > origin_pi:
                    origin_pi = pi_amp
                else:
                    u = np.random.rand(1)
                    if u[0] < pi_amp / origin_pi:
                        origin_pi = pi_amp
                    else:
                        p.amp -= e2[0]
            else:
                p.amp -= e2[0]
            
            e3 = np.random.normal(0, SIGMA_SIGMA, 1)
            
            p.sigma += e3[0]

            if(p.sigma > 0):
                generate_rdata()
                pi_sigma = calculate_pi()
                
                if pi_sigma > origin_pi:
                    origin_pi = pi_sigma
                else:
                    u = np.random.rand(1)
                    if u[0] < pi_sigma / origin_pi:
                        origin_pi = pi_sigma
                    else:
                        p.sigma -= e3[0]
            else:
                p.sigma -= e3[0]

            
            cnt += 1
            if cnt % 1000 == 0:
                update_vis()
                update_vis2()
        

    # while True:
    #     for i, p in enumerate(model.peaks):


def update_vis2():
    global fig, axes
    global data, cdata, rdata, angle, i_peaks, model

    xn = 2
    yn = len(angle)/2

    for i, d in enumerate(data):
        axes[i/yn][i%yn].clear()
        axes[i/yn][i%yn].plot(model.xrange, d, 'k')

    # for i, peaks in enumerate(i_peaks):
    #     for p in peaks:
    #         axes[i/yn][i%yn].plot(p[0], p[1], 'rx')

    x = np.arange(model.xrange[0], model.xrange[-1]+1, 0.1)
    
    for i, rd in enumerate(rdata):
        # axes[i/yn][i%yn].clear()
        # print(rd)
        axes[i/yn][i%yn].plot(x, rd, 'b')
    # print(len(rdata))
    fig.canvas.draw()

    plt.pause(1)

            
def update_vis():
    global fig2, fig3, axes2, axes3
    global data, cdata, angle, i_peaks, model

    xn = 2
    yn = len(angle)/2
    colors = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'bs', 'gs', 'rs']

    if 'fig2' not in globals():
        print('init fig 2')
        fig2, axes2 = plt.subplots(xn, yn)
    x = np.arange(model.xrange[0], model.xrange[-1]+1, 0.1)
    for i, cd in enumerate(cdata):
        axes2[i/yn][i%yn].clear()
        axes2[i/yn][i%yn].plot(x, cd, 'k')
        for j, p in enumerate(model.peaks):
            xx = proj(p.mu, angle[i])
            axes2[i/yn][i%yn].plot(xx, cd[cidx(xx)], colors[j])
    for i, peaks in enumerate(i_peaks):
        for j, p in enumerate(peaks):
            axes2[i/yn][i%yn].plot(p[0], p[1], 'rx')
    fig2.canvas.draw()

    if 'fig3' not in globals():
        print('init fig 3')
        fig3, axes3 = plt.subplots(1, 1)
    axes3.clear()
    for i, p in enumerate(model.peaks):
        axes3.plot(p.mu[0], p.mu[1], colors[i]) 
    axes3.set_xlim([-256, 256])
    axes3.set_ylim([-256, 256])
    fig3.canvas.draw()

    plt.pause(0.1)

    return

def visualize():
    global fig, axes
    global data, cdata, angle, i_peaks, model


    plt.ion()

    xn = 2
    yn = len(angle)/2
    fig, axes = plt.subplots(xn, yn)
    for i, d in enumerate(data):
        axes[i/yn][i%yn].plot(model.xrange, d, 'k')
    for i, peaks in enumerate(i_peaks):
        for p in peaks:
            axes[i/yn][i%yn].plot(p[0], p[1], 'rx')

    update_vis()

    return

def main():
    global data, cdata, angle, i_peaks, model

    np.random.seed(int(round(time.time())))
    args = parse_arguments()
    (data, angle) = parse_inputs(args)
    threshold, i_peaks = infer_peak(data)
    if not args.num_of_peaks:
        n = infer_n(i_peaks)
    else:
        n = args.num_of_peaks
    # model = Model(n)
    model = Model(n, data, threshold)
    cdata = generate_cdata()
    visualize()
    infer_mu()
    MCMC()
    input("Press [enter] to continue.")

    return

if __name__ == "__main__":
    main()
