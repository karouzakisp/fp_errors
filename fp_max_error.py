from mpmath import mp 
import random
import math
import numpy as np

from bayes_optim import BO, RealSpace
from bayes_optim.surrogate import GaussianProcess, trend, RandomForest


mp.prec = 540

import struct

def floatToRawLongBits(value):
	return struct.unpack('Q', struct.pack('d', value))[0]

def longBitsToFloat(bits):
	return struct.unpack('d', struct.pack('Q', bits))[0]



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
        eval_budget = 20 * (dim)
    )

    bo = BO(search_space=SS, obj_fun=black_box_f,
            model=model,  max_FEs=100,
            verbose=False, n_point=50,
            acquisition_optimization={"optimizer" : "BFGS"});
    xopt, fopt, _ = bo.run()
    ll.append(fopt[0])

import matplotlib.pyplot as plt
for i in range(0, 20):
    run()

plt.hist(ll, edgecolor='black')
plt.title('Histogram of Maximum Errors')
plt.xlabel('Maximum Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()