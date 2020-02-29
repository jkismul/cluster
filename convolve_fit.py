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

#HANDLE INPUT
parser = argparse.ArgumentParser()
parser.add_argument('-fr',help='firing_rate')
input_args=parser.parse_args()

# LOAD DATA
L5_kerns = pickle.load(open('data/L5.p', 'rb'))
L23_kerns = pickle.load(open('data/L23.p', 'rb'))
LFP_signal = pickle.load(open('data/LFP_lines.p', 'rb'))

# <editor-fold desc="VARIABLES">
a=1 #for skewnorm
err_break = 0.03 #while till err below this value
max_iter=5
p_degree=7
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
#NORMALIZE KERNELS
# kerns0=preprocessing.Normalizer().transform(kerns[0])
# kerns1=preprocessing.Normalizer().transform(kerns[1])
# kerns=[kerns0,kerns1]

data = LFP_signal[1]

args = [data]
num_kernels_to_run = kerns.shape[0]
kernel_num_tsteps = len(kerns[0][0])
num_channels = 16
dt = 40. / num_tsteps
dt_k = 1e-3
dt_k = dt_k / kernel_num_tsteps
t = np.arange(len(data)) * dt #denne er 16 lang?
err = 1e16#1e9
amp_min=0
amp_max=5
norm_scales=[2,4]
norm_scales_bounds=[1,5]
norm_loc_bounds=[0,40]
norm_amp_bounds=[0,1e5]
bound_big = 1e3 #huge value for bounds, temp thing
fit = np.zeros((num_channels, num_tsteps))
firing_rates_opt = np.zeros((num_kernels_to_run, num_tsteps))
teller = 0
# print(np.shape(data))
t = np.arange(len(data[0])) * dt #denne er 16 lang?
t2 = np.arange(len(data[0])) * dt #denne er 16 lang?


# </editor-fold>

def confun(x): #polynomial constraints
    constr = []
    for idx in range(num_kernels_to_run):
        poly_ = np.poly1d(x[(idx) * p_degree + idx:(idx + 1) * p_degree + idx])
        f = lambda t: poly_(t / t[-1])
        poly_ = f(np.linspace(t_min, t_max, num_tsteps))
        firing_rate_ = np.abs(poly_)
        constr=np.hstack((constr,firing_rate_[0]))
        constr=np.hstack((constr,firing_rate_[-1]))
    return constr

#create initial values and boundaries
if input_args.fr == 'poly':
    def reroll(kerns):
        x0 = []
        for i in range(kerns.shape[0]):
            x = np.random.randint(-5, 5, p_degree)
            z = np.random.normal(loc=0, scale=10)
            x0 = np.hstack((x0, x))
            x0 = np.hstack((x0, z))
        return x0
    bounds=[]
    # bounder = reroll(kerns)
    for i in range(kerns.shape[0]):
        for j in range(p_degree):
            bounds.append([-1e6,1e6])
        bounds.append([-10,10])
    cons = ({'type': 'eq',
             'fun': confun})
    # cons=None
elif input_args.fr == 'norm':
    def reroll(kerns):
        x0 = []
        for j in range(kerns.shape[0]):
            #set loc = every N point along t-axis, then use only scale+amp to optimize
            loc_ = np.random.randint(t_min+0.5, t_max-0.5, num_norms)
            # loc_=np.linspace(0,40,num_norms)
            scale_ = np.random.randint(norm_scales[0], norm_scales[1], num_norms)
            # amp_ = np.random.randint(bound_big+1, bound_big**2-1, num_norms)
            amp_ = np.random.randint(1,2, num_norms)

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
    cons = None
gauss = lambda x, loc, scale: 1 / (np.sqrt(2 * np.pi) * scale) * np.exp(
    -np.power((x - loc) / scale, 2) / 2)




def minimize_firing_rates(x, *args):
    data = args[0]
    result_thing = 0
    # for i in range(num_channels):
    for i in range(1):
    # for i in [0,15]:
        fit_ = np.zeros(len(data[0][i]))
        for idx in range(num_kernels_to_run):
            if input_args.fr !='norm' and input_args.fr!='poly':
                amp_=x[num_tsteps*(idx+1)+idx]
                firing_rate_ = amp_*x[(idx)*num_tsteps+idx:(idx+1)*num_tsteps+idx]
            if input_args.fr == 'poly':
                poly_ = np.poly1d(x[(idx)*p_degree+idx:(idx+1)*p_degree+idx])
                # f = lambda t: poly_(t - x[(idx+1)*p_degree])
                # f = lambda t: poly_((t - x[(idx+1)*p_degree])/t_max)

                f = lambda t: poly_(t/t[-1])
                # f = lambda t: poly_(t/t_max)

                poly_=f(np.linspace(t_min,t_max,num_tsteps))
                firing_rate_=poly_
                # firing_rate_= np.abs(poly_)
                firing_rate_[firing_rate_<0]=0
            if input_args.fr == 'norm':
                firing_rate_ = np.zeros(num_tsteps)
                for jj in range(num_norms):
                    firing_rate_ +=x[idx * 3 + (jj * 3 + 2)]*st.skewnorm.pdf(np.linspace(t_min, t_max, num_tsteps),a,loc=x[idx * 3 + jj * 3],scale=x[idx * 3 + (jj * 3 + 1)])

                    # firing_rate_ += x0[idx * 3 + (jj * 3 + 2)]*gauss(np.linspace(t_min, t_max, num_tsteps),x0[idx * 3 + jj * 3],x0[idx * 3 + (jj * 3 + 1)])
                    # rv = st.norm.pdf(np.linspace(t_min, t_max, num_tsteps), loc=x0[idx * 3 + jj * 3],
                    #                  scale=x0[idx * 3 + (jj * 3 + 1)])

                    # rv = st.norm.pdf(np.linspace(t_min, t_max, num_tsteps), loc=x0[idx * 3 + jj * 3],
                    #                  scale=x0[idx * 3 + (jj * 3 + 1)])
                    #
                    #
                    # amp_ = x0[idx * 3 + (jj * 3 + 2)]
                    # normie = amp_ * rv
                    # firing_rate_ += normie

            fit_ += ss.convolve(firing_rate_, kerns[idx][i], mode="same")
            # fit_ += ss.convolve(firing_rate_, krnl[idx], mode="same")

        result_thing += np.sum((data[0][i] - fit_) ** 2)

    # print('Layer',i,'error:',np.sum((data[0][i] - fit_) ** 2))
    return result_thing




while err > err_break:
    teller += 1
    if teller == max_iter:
        break
    x0 = reroll(kerns)
    # print(x0)
    x0 = [5.,2.,0.5,15.,5.,1.5]
    # cons = ({'type': 'eq',
    #          'fun': confun})
    # cons=None
    res = minimize(minimize_firing_rates, x0, args=args, bounds=bounds, method="L-BFGS-B",
                   options={'maxfun': 1000000, 'disp': False, 'gtol': 1e-29, 'ftol': 1e-100})  # ,'iprint':0})

    # res = minimize(minimize_firing_rates, x0, args=args, bounds=bounds, tol=1e-25, options={'eps': 1e-7})
    # res = minimize(minimize_firing_rates, x0, args=args,constraints=cons, bounds=bounds,method=None,
    #                tol=1e-25,
    #                options={'eps': 1e-7,'maxiter':1000,'maxfun':1000000,'ftol':1e-100,'gtol':1e-8,'maxcor':1000,'maxls':1000,'alpha':1e-4})#'iprint':100},)
    if minimize_firing_rates(res['x'], args) < err:
        err = minimize_firing_rates(res['x'], args)
        res_best = res
    print('iteration:', teller, ', error:', minimize_firing_rates(res['x'], args), 'best:', err)
    print(res['message'],'nit',res['nit'])
    # print(res['nit'])
print('final error pred', minimize_firing_rates(res_best['x'], args))
print(res['message'])
print('iters done',res['nit'])
# print(res_best)
# for i in range(num_channels):
for i in range(1):
# for i in [0,15]:
    for idx in range(num_kernels_to_run):
        if input_args.fr != 'norm' and input_args.fr != 'poly':
            amp_ = res_best.x[num_tsteps * (idx + 1) + idx]
            firing_rate_ = amp_ * res_best.x[(idx) * num_tsteps + idx:(idx + 1) * num_tsteps + idx]
        if input_args.fr == 'poly':
            poly_ = np.poly1d(res_best.x[(idx)*p_degree+idx:(idx+1)*p_degree+idx])
            # f = lambda t: poly_(t - res_best.x[(idx+1)*p_degree])
            # f = lambda t: poly_((t - res_best.x[(idx+1)*p_degree])/t[-1])
            f = lambda t: poly_(t / t[-1])

            poly_=f(np.linspace(t_min,t_max,num_tsteps))
            # firing_rate_= np.abs(poly_)
            firing_rate_=poly_
            firing_rate_[firing_rate_ < 0] = 0

        if input_args.fr == 'norm':
            firing_rate_ = np.zeros(num_tsteps)
            for jj in range(num_norms):  # num_norms

                firing_rate_ += res.x[idx * 3 + (jj * 3 + 2)] * st.skewnorm.pdf(np.linspace(t_min, t_max, num_tsteps), a,
                                                                             loc=res.x[idx * 3 + jj * 3],
                                                                             scale=res.x[idx * 3 + (jj * 3 + 1)])

                # firing_rate_ += x0[idx * 3 + (jj * 3 + 2)] * gauss(np.linspace(t_min, t_max, num_tsteps),
                #                                                    x0[idx * 3 + jj * 3], x0[idx * 3 + (jj * 3 + 1)])

                # rv = st.norm.pdf(np.linspace(t_min, t_max, num_tsteps), loc=res_best.x[idx * 3 + jj * 3],
                #                  scale=res_best.x[idx * 3 + (jj * 3 + 1)])
                # amp_ = res_best.x[idx * 3 + (jj * 3 + 2)]
                # normie = amp_ * rv
                # firing_rate_ += normie
        firing_rates_opt[idx] = firing_rate_
        fit[i] += ss.convolve(firing_rate_, kerns[idx][i], mode="same")  # ,method='direct')
        # fit[i] += ss.convolve(firing_rate_, krnl[idx], mode="same")  # ,method='direct')

pickle.dump(fit, open('data/fit_all_channels_test.p', 'wb'))

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
    # ax_k.plot(t_kern, krnl[idx], c=coll[idx])

    # lines.append(l_)
    line_names.append("firing rate fit {}".format(idx))

ax_sig.plot(np.linspace(t_min, t_max, len(data[0])), data[0], c='k')
# ax_sig.plot(np.linspace(t_min, t_max, len(doto[0])), doto[0], c='k')

ax_sig.plot(np.linspace(t_min, t_max, len(fit[0])), fit[0], c='gray', ls='--')

# fig.legend(lines, line_names, frameon=False, ncol=4)
plotting_convention.simplify_axes(fig.axes)
plt.savefig("plots/100_runs.png")
# </editor-fold>
print("BØR JEG HA MED SHIFT I POLY? VIL VEL STARTE I 0 UANSETT")
print("KERNELENE VÅRE SER GANSKE FORMLIKE UT, "
      "SÅ DET INNEBÆRER AT MULIG INFO FRA KUN EEG MÅLING ER BEGRENSET OM MAN TILLATER"
      "ALL SLAGS FYRING? MEN MED CONSTRAINTS OM EN GLATT FORM "
      "PÅ FYRINGSRATER KAN MAN KANKSJE SI NOE OM POPULAJSONENE?")
print("gauss-metoden slik den ser ut kan tillate flere gausser med forskjellig amp og scale å være sentrert i samme punkt."
      "Det virker litt ufysisk, så kanskje noe slikt som 1 gaus per N tids-steg?")