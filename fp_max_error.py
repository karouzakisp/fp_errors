from mpmath import mp 
import random
import math
import numpy as np

from bayes_optim import BO, RealSpace
from bayes_optim.surrogate import GaussianProcess, trend, RandomForest


mp.prec = 240



def blackbox_imprv_f(ax):
    sum_sins = 0
    sums_sin_mp = 0
    c = 0
    c_mp = 0
    freq = ax[-1]
    for x in ax[:-1]:
        x_mp = mp.mpf(x)
        f_mp = mp.mpf(freq)
        x1 = math.sin(x*freq*2*math.pi)
        x2 = mp.sin(x_mp * f_mp * mp.mpf(2) * math.pi)
        y = x1 - c
        y_mp = x2 - c_mp
        t = sum_sins + y
        t_mp = sums_sin_mp + y
        c = (t - sum_sins) - y
        c_mp = (t_mp - sums_sin_mp) - y_mp
        sum_sins = t
        sums_sin_mp = t_mp
    y = mp.absmin(sums_sin_mp - sum_sins)
    return -math.log2(getFPNum(sums_sin_mp, sum_sins))

     

def black_box_f(ax):
    sum_sins = 0
    sums_sin_mp = 0
    freq = ax[-1]
    for x in ax[:-1]:
        x_mp = mp.mpf(x)
        f_mp = mp.mpf(freq)
        sum_sins += math.sin(x*freq*2*math.pi)
        sums_sin_mp += mp.sin(x_mp * f_mp * mp.mpf(2) * math.pi)
    y = mp.absmin(sums_sin_mp - sum_sins)
    return -math.log2(getFPNum(sums_sin_mp, sum_sins))



import struct

def floatToRawLongBits(value):
	return struct.unpack('Q', struct.pack('d', value))[0]

def longBitsToFloat(bits):
	return struct.unpack('d', struct.pack('Q', bits))[0]



def getFPNum(a,b):
    ia = floatToRawLongBits(np.abs(a))
    ib = floatToRawLongBits(np.abs(b))
    zo = floatToRawLongBits(0)
    if mp.sign(a)!=mp.sign(b):
        res = abs(ib-zo)+abs(ia-zo)
    else:
        res = abs(ib-ia)
    return int(res+1)



ll = []
def run():
    ppi = math.pi
    dim = 81
    lt, ht = 0, 1000
    SS = RealSpace([0, 1000], var_name="time_series") * dim
    thetaL = 1e-10 * (ht - lt) * np.ones(dim)
    thetaU = 10 * (ht - lt) * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

    #autocorrelation paremeters of GPR
    model = GaussianProcess(
        corr="squared_exponential",
        theta0=theta0,
        thetaL=thetaL,
        thetaU=thetaU,
        noise_estim=True,
        nugget=1e-3,
        optimizer="BFGS",
        wait_iter=10,
        random_start = 5 * (dim),
        likelihood="concentrated",
        eval_budget = 30 * (dim)
    )

    bo = BO(search_space=SS, obj_fun=black_box_f,
            model=model,  max_FEs=70,
            verbose=False, n_point=50,
            acquisition_optimization={"optimizer" : "BFGS"});
    xopt, fopt, _ = bo.run()
    ll.append(-fopt[0])

import matplotlib.pyplot as plt
for i in range(0, 20):
    run()

plt.hist(ll, edgecolor='black')
plt.title('Histogram of Maximum Errors')
plt.xlabel('Maximum Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
