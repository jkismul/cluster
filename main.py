# <editor-fold desc="import packages">
import os
import argparse
import numpy as np
import LFPy
import pickle
import matplotlib.pyplot as plt
# </editor-fold>

# <editor-fold desc="Parse input">
parser = argparse.ArgumentParser()
parser.add_argument('-f',help='filtered:y/n')
parser.add_argument('-b',help='blocked:y/n')
args = parser.parse_args()
# </editor-fold>

# <editor-fold desc="Read data">
morphology = 'morphologies/L5_Mainen96_LFPy.hoc'
if args.f == 'y' and args.b!='y':
    dist_E = pickle.load(open('data/firing_signal_basal_inhibition_off_filter_on.p', 'rb'))
    dist_I = pickle.load(open('data/firing_signal_apical_inhibition_off_filter_on.p', 'rb'))
    suffix = 'fy_bn'

if args.f != 'y' and args.b!='y':
    dist_E = pickle.load(open('data/firing_signal_basal_inhibition_off_filter_off.p', 'rb'))
    dist_I = pickle.load(open('data/firing_signal_apical_inhibition_off_filter_off.p', 'rb'))
    suffix= 'fn_bn'

if args.f == 'y' and args.b=='y':
    dist_E = pickle.load(open('data/firing_signal_basal_inhibition_on_filter_on.p', 'rb'))
    dist_I = pickle.load(open('data/firing_signal_apical_inhibition_on_filter_on.p', 'rb'))
    suffix = 'fy_by'

if args.f != 'y' and args.b=='y':
    dist_E = pickle.load(open('data/firing_signal_basal_inhibition_on_filter_off.p', 'rb'))
    dist_I = pickle.load(open('data/firing_signal_apical_inhibition_on_filter_off.p', 'rb'))
    suffix='fn_by'

if args.b == 'y':
    fitses = pickle.load(open('data/fits_inhibition_off.p', 'rb'))
    lines=pickle.load(open('data/lines_inhibition_off.p','rb'))
else:
    fitses = pickle.load(open('data/fits_inhibition_on.p', 'rb'))
    lines = pickle.load(open('data/lines_inhibition_on.p', 'rb'))

# dist_I = dist_I[dist_I>2.5]
# dist_E = dist_E[dist_E>2.]
# dist_I = dist_I[dist_I<11]
# test uten det over
# </editor-fold>

# <editor-fold desc="make dir">
try:
    os.makedirs(os.path.join('plots', suffix))
except OSError:
    pass
# </editor-fold>

# <editor-fold desc="Define functions">
def insert_synapses(synparams, section, n, delay,heights):
    if section == 'dend':
        maxim = heights['basal_max']
        minim = heights['min']
        dist = dist_E
        delay =delay['E']
        # delay=np.random.normal(loc=delay['E'],scale=delay['std_E'])
        firings_per_synapse = firings_per_synapse_E
    if section == 'apic':
        maxim = heights['max']
        minim = heights['apical_min']
        dist = dist_I
        delay = delay['I']
        # delay=np.random.normal(loc=delay['I'],scale=delay['std_I'])
        firings_per_synapse = firings_per_synapse_I
    if section == 'allsec':
        maxim = heights['max']
        minim = heights['min']
        delay = delay['T']
    '''find n compartments to insert synapses onto'''
    idx = cell.get_rand_idx_area_norm(section=section, nidx=n, z_min=minim, z_max=maxim)
    # Insert synapses in an iterative fashion
    for i in idx:
        ############################################
        #hva om cell i velger en tid fra distr, og synapsene fyrer normalfordelt rundt den tiden
        ############################################
        synparams.update({'idx': int(i)})
        # Create synapse(s) and setting times using the Synapse class in LFPy
        s = LFPy.Synapse(cell, **synparams)
        s.set_spike_times(np.random.choice(dist, firings_per_synapse) + delay)
# </editor-fold>

# <editor-fold desc="Define parameters & variables">
parameters{
    'num_cells':300,
    'divi':2,
    'n_syn':10,

}
#PUTT ALLE DISSE I EN DICTIONARY, SÅ DET ER LETT Å LAGRE PARAM SOM SKAPER ET PLOTT
# Define all parameters and variables used
num_cells = 300
divi = 2 #first 1/divi cells get basal input, rest apical
cutoff = int(num_cells / divi)
# n_syn = 200
n_syn = 10
# delay = {
#     'E':0.,
#     'I':0.,
#     'T':0.,
# }
# std_factor = {
#     'E':0.1,
#     'I':0.1,
#     'T':0.1,
# }
for i in ['E','I','T']:
    delay.update({'std_{}'.format(i):std_factor[i]*delay[i]})
heights = {
    'max':1000.,
    'min':-1000.,
    'basal_max':-50.,
    'apical_min':500.,
}
firings_per_synapse = 1
firings_per_synapse_E = 1
firings_per_synapse_I = 1

tstart=-300.
tstop = 20.  # bryuns-H has 20 and 25 for eeg, 40 for lfp and icsd
p = [] #dipole moment
v = [] #vmem
x_line = np.linspace(0, tstop, len(fitses[0]))

# Define cell parameters
cell_parameters = {
    'morphology': morphology,
    'cm': 1.0,  # membrane capacitance
    'Ra': 150.,  # axial resistance
    'v_init': -65.,  # initial crossmembrane potential
    'passive': True,  # turn on NEURONs passive mechanism for all sections
    # 'passive': False,  # turn on NEURONs passive mechanism for all sections
    'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
    'nsegs_method': 'lambda_f',  # spatial discretization method
    'lambda_f': 100.,  # frequency where length constants are computed
    'dt': 2. ** -3,  # simulation time step size
    'tstart': tstart,  # start time of simulation, recorders start at t=0
    'tstop': tstop  # 50.,  # stop simulation at 100 ms.
}

# Define synapse parameters
synapse_parameters = {
    'idx': [],
    'syntype': 'Exp2Syn',
    'tau1': 1.,
    'tau2': 3.,
    'weight': .001,  # synaptic weight
    'record_current': True,  # record synapse current
}

# </editor-fold>

# <editor-fold desc="Main simulation">
for i in range(num_cells):
    print('cell {} of {}.'.format(i + 1, num_cells), end='\r')
    cell = LFPy.Cell(**cell_parameters)
    cell.set_rotation(x=4.99, y=-4.33)  # let rotate around z-axis

    if i <= cutoff:
        insert_synapses(synapse_parameters, 'dend', n_syn, delay,heights)
    else:
        insert_synapses(synapse_parameters, 'apic', n_syn, delay,heights)

    cell.simulate(rec_imem=True, rec_vmem=True, rec_current_dipole_moment=True)
    P = cell.current_dipole_moment
    p.append(np.asarray(P))
    v.append(cell.vmem[0])
print("                                 ", end='\r')
print("done")
# </editor-fold>

# <editor-fold desc="Mean or Sum">
bas = np.mean(p[0:cutoff], axis=0)[:, 0]
api = np.mean(p[cutoff:], axis=0)[:, 0]
signal_mean = np.mean(p, axis=0)[:, 0]

# bas = np.sum(p[0:cutoff], axis=0)[:, 0]
# api = np.sum(p[cutoff:], axis=0)[:, 0]
# signal_mean = np.sum(p, axis=0)[:, 0]
# </editor-fold>

# <editor-fold desc="Scaling">
a=np.max(abs(signal_mean))
b=np.max(abs(lines[1]))
c=a/b
d=np.max(abs(bas))
e=np.max(abs(fitses[0]))
f=d/e
g=np.max(abs(api))
h=np.max(abs(fitses[1]))
i=g/h
factors = [c, f, i]
# </editor-fold>

# <editor-fold desc="Main plot">
plt.figure()
plt.plot(cell.tvec, signal_mean, color='gray', ls='--', label='total')
plt.plot(lines[0], lines[1] * factors[0], 'k')

plt.plot(cell.tvec, bas, label='dend', color='orange', ls='--')
plt.plot(x_line, fitses[0] * factors[1], color='red', ls='-')

plt.plot(cell.tvec, api, label='apic', color='cyan', ls='--')
plt.plot(x_line, fitses[1] * factors[2], color='blue', ls='-')

plt.xlabel('time [ms]')
plt.ylabel('Voltage [some unit]')
plt.title('{}'.format(suffix))
plt.legend()
plt.savefig('plots/{}/signal_{}.pdf'.format(suffix,suffix))
# </editor-fold>

# <editor-fold desc = "Firing rate plots">
binny = np.linspace(0, 20, 200)
plt.figure()
plt.hist(dist_E, bins=binny, color='red')
plt.title('{}'.format(suffix))
plt.savefig('plots/{}/dend_{}.pdf'.format(suffix,suffix))
plt.figure()
plt.hist(dist_I, bins=binny, color='blue')
plt.title('{}'.format(suffix))
plt.savefig('plots/{}/api_{}.pdf'.format(suffix,suffix))
# </editor-fold>

# <editor-fold desc="Vmem plot">
fig,axs = plt.subplots(2)
for i in range(num_cells):
    if i<=cutoff:
        axs[0].plot(cell.tvec, v[i])
    else:
        axs[1].plot(cell.tvec,v[i])
axs[0].set_title('basal')
axs[1].set_title('apical')
fig.suptitle('{}'.format(suffix))
fig.savefig('plots/{}/vmems_{}.pdf'.format(suffix,suffix))
# </editor-fold>