import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.stats as st
from scipy.optimize import minimize
import pandas
import argparse
import pickle

sys.path.insert(0, '/dependencies/')
from dependencies import plotting_convention

#HANDLE INPUT
parser = argparse.ArgumentParser()
parser.add_argument('-fr',help='firing_rate')
input_args=parser.parse_args()

# LOAD DATA
L5_kerns = pickle.load(open('data/L5.p', 'rb'))
L23_kerns = pickle.load(open('data/L23.p', 'rb'))
LFP_signal = pickle.load(open('data/LFP_lines.p', 'rb'))

# SET VARIABLES
err_break = 0.03 #while till err below this value
max_iter=5
p_degree=7
num_norms=10
t_min = 0.
t_max=40.
# kerns = np.vstack((L5_kerns,L23_kerns))
kerns = L5_kerns
kerns = np.asarray(kerns)
kerns = np.flip(kerns, axis=1)  # kerns was sorted reversely to signal!
num_tsteps = len(LFP_signal[1][1])
data = LFP_signal[1]
args = [data]
num_kernels_to_run = kerns.shape[0]
kernel_num_tsteps = len(kerns[0][0])
num_channels = 16
dt = 40. / num_tsteps
dt_k = 1e-3
dt_k = dt_k / kernel_num_tsteps
t = np.arange(len(data)) * dt
err = 1e9
amp_min=0
amp_max=5
bound_big = 1e3 #huge value for bounds, temp thing
#create initial values and boundaries
if input_args.fr == 'poly':
    def reroll(kerns):
        x0 = []
        for i in range(kerns.shape[0]):
            x = np.random.randint(0, 2, p_degree)
            z = np.random.normal(loc=0, scale=10)
            x0 = np.hstack((x0, x))
            x0 = np.hstack((x0, z))
        return x0
    bounds=[]
    bounder = reroll(kerns)
    for i in range(len(bounder)):
        bounds.append([0, 1e3])#CHANGE! ALSO ABS SOMEWHERE?
elif input_args.fr == 'norm':
    def reroll(kerns):
        x0 = []
        for i in range(kerns.shape[0]):
            loc_ = np.random.randint(1, 39, num_norms)
            scale_ = np.random.randint(2, 4, num_norms)
            amp_ = np.random.randint(1e3+1, 1e5-1, num_norms)
            for i in range(num_norms):
                x0 = np.hstack((x0, loc_[i]))
                x0 = np.hstack((x0, scale_[i]))
                x0 = np.hstack((x0, amp_[i]))
        return x0
    bounds=[]
    bounder = reroll(kerns)
    for i in range(len(bounder)):
        if (i + 3) % 3 == 0:
            bounds.append([0, 40])
        if (i + 5) % 3 == 0:
            bounds.append([1, 5])
        if (i + 4) % 3 == 0:
            bounds.append([1e3, 1e6])
else:
    def reroll(kerns):
        x0 = []
        for i in range(kerns.shape[0]):
            x = np.zeros(num_tsteps)  # time step value
            # x=np.random.randint(0,1000,num_tsteps)
            x0 = np.hstack((x0, x))
            x0 = np.hstack((x0, np.random.randint(amp_min, amp_max)))  # amp
        return x0
    bounds=[]
    bounder=reroll(kerns)
    for i in range(len(bounder)):
        bounds.append([0, bound_big])  # should maybe be the sqrt of this, as amp*timestep is the "real" max height


def minimize_firing_rates(x, *args):
    data = args[0]
    result_thing = 0
    # for i in range(num_channels):
    for i in range(1):
        fit_ = np.zeros(len(data[0][i]))
        for idx in range(num_kernels_to_run):
            if input_args.fr !='norm' and input_args.fr!='poly':
                amp_=x[num_tsteps*(idx+1)+idx]
                firing_rate_ = amp_*x[(idx)*num_tsteps+idx:(idx+1)*num_tsteps+idx]
            if input_args.fr == 'poly':
                poly_ = np.poly1d(x[(idx)*p_degree+idx:(idx+1)*p_degree+idx])
                f = lambda t: poly_(t - x[(idx+1)*p_degree])
                poly_=f(np.linspace(t_min,t_max,num_tsteps))
                firing_rate_= poly_
            if input_args.fr == 'norm':
                firing_rate_ = np.zeros(num_tsteps)
                for jj in range(num_norms):
                    rv = st.norm.pdf(np.linspace(t_min, t_max, num_tsteps), loc=x0[idx * 3 + jj * 3],
                                     scale=x0[idx * 3 + (jj * 3 + 1)])
                    amp_ = x0[idx * 3 + (jj * 3 + 2)]
                    normie = amp_ * rv
                    firing_rate_ += normie
            # plt.figure()
            # plt.plot(firing_rate_)
            # plt.show()
            fit_ += ss.convolve(firing_rate_, kerns[idx][i], mode="same")
        result_thing += np.sum((data[0][i] - fit_) ** 2)
    return result_thing

hist_bins = np.linspace(0, 50, num_tsteps + 1)

teller = 0
while err > err_break:
    teller += 1
    if teller == max_iter:
        break
    x0 = reroll(kerns)
    # res = minimize(minimize_firing_rates, x0, args=args, bounds=bounds, tol=1e-25, options={'eps': 1e-7})
    res = minimize(minimize_firing_rates, x0, args=args, bounds=bounds,method= None,
                   tol=1e-25,
                   options={'eps': 1e-7,})#'gtol':-8000,'maxcor':1000,'maxls':1000,})#'iprint':100},)
    # if teller==1:
    #     res_best=res
    if minimize_firing_rates(res['x'], args) < err:
        err = minimize_firing_rates(res['x'], args)
        res_best = res
    print('iteration:', teller, ', error:', minimize_firing_rates(res['x'], args), 'best:', err)

print('final error pred', minimize_firing_rates(res_best['x'], args))

print(res['message'])
# pickle.dump(res['x'],open('data/pdf_params.p','wb'))

##FILTER FIRING
dt = 20 / 74.
fs = 1 / dt * 1000
fc = 200
w = fc / (fs * 0.5)
b, a = ss.butter(2, w, 'lowpass')

fit = np.zeros((num_channels, num_tsteps))

# fit=np.zeros(kernel_num_tsteps)
# firing_rates_opt = []
# firing_rates_opt = [np.zeros(num_kernels_to_run),np.zeros(num_tsteps)]
firing_rates_opt = np.zeros((num_kernels_to_run, num_tsteps))
# firing_rates_opt = np.empty(num_kernels_to_run)
# firing_rates_opt = np.zeros(num_kernels_to_run)
filtfire = []
fit_lp = np.zeros(num_tsteps)

fit_s = [np.zeros(num_tsteps), np.zeros(num_tsteps)]
all_firing_rates = []

# fit_s = [np.zeros((kernel_num_tsteps)),np.zeros((kernel_num_tsteps))]
for i in range(num_channels):
    for idx in range(num_kernels_to_run):
        if input_args.fr != 'norm' and input_args.fr != 'poly':
            amp_ = res_best.x[num_tsteps * (idx + 1) + idx]
            firing_rate_ = amp_ * res_best.x[(idx) * num_tsteps + idx:(idx + 1) * num_tsteps + idx]

        if input_args.fr == 'poly':
            poly_ = np.poly1d(res_best.x[(idx)*p_degree+idx:(idx+1)*p_degree+idx])
            f = lambda t: poly_(t - res_best.x[(idx+1)*p_degree])
            poly_=f(np.linspace(0,40,num_tsteps))
            firing_rate_= poly_
        if input_args.fr == 'norm':
            firing_rate_ = np.zeros(num_tsteps)
            for jj in range(num_norms):  # num_norms
                rv = st.norm.pdf(np.linspace(0, 40, num_tsteps), loc=res_best.x[idx * 3 + jj * 3],
                                 scale=res_best.x[idx * 3 + (jj * 3 + 1)])
                amp_ = res_best.x[idx * 3 + (jj * 3 + 2)]
                normie = amp_ * rv
                firing_rate_ += normie

        firing_rates_opt[idx] = firing_rate_
        filtr = ss.filtfilt(b, a, firing_rate_)
        filtfire.append(filtr)
        # fit += ss.convolve(firing_rate_, L5_kerns[idx][0], mode="same")#,method='direct')
        fit[i] += ss.convolve(firing_rate_, kerns[idx][i], mode="same")  # ,method='direct')

        # fit_lp += ss.convolve(filtfire[idx],L5_kerns[idx][0],mode='same')#,method='direct')
        # fit_s[idx] = ss.convolve(firing_rate_,L5_kerns[idx][0],mode='same')

pickle.dump(fit, open('data/fit_all_channels_test.p', 'wb'))

# filtfire=np.where(np.asarray(filtfire)>0,np.asarray(filtfire),0)
#
# pickle.dump(firing_rates_opt,open('data/firing_inhibition_on.p','wb'))
# pickle.dump(filtfire,open('data/filtered_firing_inhibition_on.p','wb'))
# pickle.dump(fit_s,open('data/fits_inhibition_on.p','wb'))
# #
# # #PLOT
fig = plt.figure(figsize=[9, 4])
fig.subplots_adjust(hspace=0.5, top=0.75, bottom=0.2)
ax_k = fig.add_subplot(131, title="kernels", xlabel="time [ms]", ylabel='Voltage [mV]')
ax_fr = fig.add_subplot(132, title="Firing rates", xlabel="time [ms]")
ax_sig = fig.add_subplot(133, title="signal (convolution)", xlabel="time [ms]", ylabel='Voltage [mV]')
coll = ['r', 'b', 'm', 'g']
coll2 = ['orange', 'cyan', 'pink', 'lime']
lines = []
line_names = []
t_kern = np.linspace(0, 50, len(L5_kerns[0][0]))
# print(np.shape(t),np.shape(firing_rates_opt))
for idx in range(num_kernels_to_run):
    # l_, = ax_fr.plot(t, firing_rates_opt[idx],c=coll2[idx],alpha=0.5)
    l_, = ax_fr.plot(np.linspace(0, 40, len(firing_rates_opt[0])), firing_rates_opt[idx], c=coll2[idx], alpha=0.5)

    # l2_, = ax_fr.plot(t, filtfire[idx],c=coll[idx])
    # ax_k.plot(t_kern,L5_kerns[idx][0],c=coll[idx])
    # ax_k.plot(t_kern,kerns[idx][0],c=coll[idx])
    ax_k.plot(t_kern, kerns[idx][0], c=coll[idx])

    lines.append(l_)
    # lines.append(l2_)
    line_names.append("firing rate fit {}".format(idx))

ax_sig.plot(np.linspace(0, 40, len(data[0])), data[0], c='k')
ax_sig.plot(np.linspace(0, 40, len(fit[0])), fit[0], c='gray', ls='--')
# ax_sig.plot(t,fit_lp, c='red', ls='--')
# ax_sig.plot(t,fit_s[0], c='orange', ls='--')
# ax_sig.plot(t,fit_s[1], c='green', ls='--')
# ax_sig.plot(t,fit_s[0]+fit_s[1],c='cyan',ls='--')

fig.legend(lines, line_names, frameon=False, ncol=4)
plotting_convention.simplify_axes(fig.axes)
# plt.show()
plt.savefig("plots/100_runs.png")
#
#
# fig = plt.figure(figsize=[9, 4])
# fig.subplots_adjust(hspace=0.5, top=0.75, bottom=0.2)
# ax_1 = fig.add_subplot(121, title="kernels", xlabel="time [ms]",ylabel='Voltage [mV]')
# ax_2 = fig.add_subplot(122, title="Firing rates", xlabel="time [ms]")
# ax_3 = fig.add_subplot(221, title="signal (convolution)", xlabel="time [ms]",ylabel='Voltage [mV]')
# ax_4 = fig.add_subplot(222, title="signal (convolution)", xlabel="time [ms]",ylabel='Voltage [mV]')
#
# for idx in range(num_kernels_to_run):
#     k1, = ax_1.plot(t,)
# plt.plot()
# plt.savefig("plots/all_fits.png")
