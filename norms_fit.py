# <editor-fold desc="IMPORTS">
import sys
import numpy as np
import numpy.polynomial.polynomial as npp
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.stats as st
from scipy.optimize import minimize
import pandas as pd
import argparse
import pickle
sys.path.insert(0, '/dependencies/')
from dependencies import plotting_convention
from sklearn import preprocessing
# </editor-fold>


# LOAD DATA
L5_kerns = pickle.load(open('data/L5.p', 'rb'))
L23_kerns = pickle.load(open('data/L23.p', 'rb'))
LFP_signal = pickle.load(open('data/LFP_lines.p', 'rb'))

# <editor-fold desc="VARIABLES">
err_break = 0.03 #while till err below this value
max_iter=5
num_norms=1
t_min = 0.
t_max=40.
t_kern_min=0.
t_kern_max=50.
# kerns = np.vstack((L5_kerns,L23_kerns))
kerns = L5_kerns
kerns = np.asarray(kerns)
kerns = np.flip(kerns, axis=1)  # kerns was sorted inversely to signal!
num_tsteps = len(LFP_signal[1][1])

kerns0=preprocessing.scale(kerns[0])
kerns0=preprocessing.Normalizer().transform(kerns[0])
kerns1=preprocessing.scale(kerns[1])
kerns1=preprocessing.Normalizer().transform(kerns[1])

kerns=[kerns0,kerns1]
kerns = np.asarray(kerns)
data = LFP_signal[1]
args = [data]
num_kernels_to_run = kerns.shape[0]
kernel_num_tsteps = len(kerns[0][0])
num_channels = 16
dt = 40. / num_tsteps
dt_k = 1e-3
dt_k = dt_k / kernel_num_tsteps
t = np.arange(len(data)) * dt
err = 1e16#1e9
amp_min=0
amp_max=5
norm_scales=[2,4]
norm_scales_bounds=[1,5]
norm_loc_bounds=[0,40]
norm_amp_bounds=[0,10]
bound_big = 1e3 #huge value for bounds, temp thing
fit = np.zeros((num_channels, num_tsteps))
firing_rates_opt = np.zeros((num_kernels_to_run, num_tsteps))
teller = 0
# </editor-fold>


def reroll(kerns):
    x0 = []
    for j in range(kerns.shape[0]):
        loc_=np.linspace(0,40,num_norms)
        scale_ = 0.1*np.ones(num_norms)#np.random.randint(norm_scales[0], norm_scales[1], num_norms)
        amp_ = 0*np.ones(num_norms)#np.random.randint(1, 5, num_norms)

        for i in range(num_norms):
            x0 = np.hstack((x0, loc_[i]))
            x0 = np.hstack((x0, scale_[i]))
            x0 = np.hstack((x0, amp_[i]))
    return x0
bounds=[]
bounder = reroll(kerns)
for i in range(len(bounder)):
    if (i + 3) % 3 == 0:
        bounds.append(norm_loc_bounds)
    if (i + 5) % 3 == 0:
        bounds.append(norm_scales_bounds)
    if (i + 4) % 3 == 0:
        bounds.append(norm_amp_bounds)
cons=None


result_thing=0
locs = np.linspace(0,40,1)
# locs=np.random.randint(0,40,num_norms)
scales = np.linspace(1e-8,5,1)
amps = np.linspace(0,5,1)
errr = 1e8
vals=[]
for s1_ in scales:
    for a1_ in amps:
        for l1_ in locs:
            for s2_ in scales:
                for a2_ in amps:
                    for l2_ in locs:
                        fit_ = np.zeros(len(data[0]))
                        vals=[]
                        for idx in range(num_kernels_to_run):
                            firing_rate_ = np.zeros(num_tsteps)
                            for jj in range(num_norms):
                                if jj==0:
                                    loc_,scale_,amp_ = l1_,s1_,a1_
                                    v=[loc_,scale_,amp_]
                                if jj ==1:
                                    loc_,scale_,amp_ = l2_,s2_,a2_
                                rv = st.norm.pdf(np.linspace(t_min, t_max, num_tsteps), loc=loc_,
                                                 scale=scale_)
                                normie = amp_* rv
                                firing_rate_ += normie
                            # plt.figure()
                            # plt.plot(firing_rate_)
                            # plt.show()
                            fit_ += ss.convolve(firing_rate_, kerns[idx][0], mode="same")
                        result_thing += np.sum((data[0][0] - fit_) ** 2)
                        if result_thing<errr:
                            errr=result_thing
                            vals=[loc_,scale_,amp_]
                            vals=np.hstack((v,vals))
                            # vals=[scale_,amp_]
print(errr,vals)


for idx in range(num_kernels_to_run):
    firing_rate_ = np.zeros(num_tsteps)
    for jj in range(num_norms):
        rv = st.norm.pdf(np.linspace(t_min, t_max, num_tsteps), loc=vals[jj*3],
                         scale=vals[jj*3+1])
        amp_ = vals[jj*3+2]
        normie = amp_ * rv
        firing_rate_ += normie

    firing_rates_opt[idx] = firing_rate_
    fit[0] += ss.convolve(firing_rate_, kerns[idx][0], mode="same")  # ,method='direct')

# # pickle.dump(fit, open('data/fit_all_channels_test.p', 'wb'))

# <editor-fold desc="PLOT">
fig = plt.figure(figsize=[9, 4])
fig.subplots_adjust(hspace=0.5, top=0.75, bottom=0.2)
ax_k = fig.add_subplot(131, title="kernels", xlabel="time [ms]", ylabel='Voltage [mV]')
ax_fr = fig.add_subplot(132, title="Firing rates", xlabel="time [ms]")
ax_sig = fig.add_subplot(133, title="signal (convolution)", xlabel="time [ms]", ylabel='Voltage [mV]')
coll = ['r', 'b', 'm', 'g']
lines = []
line_names = []
# t_kern = np.linspace(t_kern_min, t_kern_max, len(L5_kerns[0][0]))
t_kern = np.linspace(t_kern_min, t_kern_max, len(kerns[0][0]))

for idx in range(num_kernels_to_run):
    # l_, = ax_fr.plot(np.linspace(t_min, t_max, len(firing_rates_opt[0])), firing_rates_opt[idx], c=coll[idx], alpha=1.0)
    l_, = ax_fr.plot(firing_rates_opt[idx], c=coll[idx], alpha=1.0)

    ax_k.plot(t_kern, kerns[idx][0], c=coll[idx])
    # lines.append(l_)
    line_names.append("firing rate fit {}".format(idx))

ax_sig.plot(np.linspace(t_min, t_max, len(data[0])), data[0], c='k')
ax_sig.plot(np.linspace(t_min, t_max, len(fit[0])), fit[0], c='gray', ls='--')

# fig.legend(lines, line_names, frameon=False, ncol=4)
plotting_convention.simplify_axes(fig.axes)
plt.savefig("plots/100_runs.png")
# </editor-fold>
