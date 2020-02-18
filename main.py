# <editor-fold desc="import packages">
import os
import argparse
import numpy as np
import LFPy
import pickle
import matplotlib.pyplot as plt
import time
import scipy.signal as ss
# </editor-fold>
print("I'm not allowing individual neurons to receive both apical and basal inputs here, should I?")
print('er dette en svakhet ved kernelmetoden, eller regner man med at en kjøring av mange nevroner med hver fyringsrate kan tilsvare ett nevron med påde api og basal?')
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
    fitses = pickle.load(open('data/fits_inhibition_on.p', 'rb'))
    lines=pickle.load(open('data/lines_inhibition_on.p','rb'))
else:
    fitses = pickle.load(open('data/fits_inhibition_off.p', 'rb'))
    lines = pickle.load(open('data/lines_inhibition_off.p', 'rb'))
# </editor-fold>

# <editor-fold desc="make dir">
folder_time = time.strftime("%m%d-%H%M")
try:
    os.makedirs(os.path.join('plots', '{}_{}'.format(suffix,folder_time)))
except OSError:
    pass
FOLDER = os.path.join('plots', '{}_{}'.format(suffix,folder_time))
# </editor-fold>

# <editor-fold desc="Define functions">
def insert_synapses(synparams, section, n,heights,firings):
    if section == 'dend':
        maxim = heights['basal_max']
        minim = heights['min']
        dist = dist_E
        firings_per_synapse = firings['firings_per_E_synapse']
    if section == 'apic':
        maxim = heights['max']
        minim = heights['apical_min']
        dist = dist_I
        firings_per_synapse = firings['firings_per_I_synapse']
    if section == 'allsec':
        maxim = heights['max']
        minim = heights['min']
        delay = delay['T']
    '''find n compartments to insert synapses onto'''
    idx = cell.get_rand_idx_area_norm(section=section, nidx=n, z_min=minim, z_max=maxim)
    # Insert synapses in an iterative fashion
    for i in idx:
        synparams.update({'idx': int(i)})
        # Create synapse(s) and setting times using the Synapse class in LFPy
        s = LFPy.Synapse(cell, **synparams)
        s.set_spike_times(np.random.choice(dist, firings_per_synapse))
# </editor-fold>

# <editor-fold desc="Define parameters & variables">

# Define electrode geometry corresponding to a laminar electrode, where contact
# points have a radius r, surface normal vectors N, and LFP calculated as the
# average LFP in n random points on each contact:
N = np.empty((16, 3))
for i in range(N.shape[0]): N[i,] = [1, 0, 0] #normal unit vec. to contacts
# put parameters in dictionary
electrodeParameters = {
    'sigma' : 0.3,              # Extracellular potential
    'x' : np.zeros(16),# + 25,      # x,y,z-coordinates of electrode contacts
    'y' : np.zeros(16),
    'z' : np.linspace(-1600, -100, 16),
    'n' : 20,
    'r' : 10,
    'N' : N,
}



parameters={
    'num_cells':300,
    'divi':2,
    'n_syn':200,
    'firings':{
        'firings_per_E_synapse':1,
        'firings_per_I_synapse':1,
    },
    'heights':{
        'max':1000.,
        'min':-1000.,
        'basal_max':-50.,
        'apical_min':500.,
    },
    'cell_parameters':{
        'morphology': morphology,
        'cm': 1.0,  # membrane capacitance
        'Ra': 150.,  # axial resistance
        'v_init': -65.,  # initial crossmembrane potential
        # 'passive': True,  # turn on NEURONs passive mechanism for all sections
        'passive': False,  # turn on NEURONs passive mechanism for all sections
        'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
        'nsegs_method': 'lambda_f',  # spatial discretization method
        'lambda_f': 100.,  # frequency where length constants are computed
        'dt': 2. ** -3,  # simulation time step size
        'tstart': -300.,  # start time of simulation, recorders start at t=0
        'tstop': 20.  # 50.,  # stop simulation at 100 ms.
    },
    'synapse_parameters':{
        'idx': [],
        # 'syntype': 'Exp2Syn',
        'syntype': 'ExpSynI',
        'tau1': 1.,
        'tau2': 3.,
        'tau':.1,
        'weight': .1001,  # synaptic weight
        'record_current': True,  # record synapse current
    },
}
cutoff = int(parameters['num_cells'] / parameters['divi'])
print('cutoff at',cutoff,'of',parameters['num_cells'])
p = [] #dipole moment
v = [] #vmem
x_line = np.linspace(0, parameters['cell_parameters']['tstop'], len(fitses[0]))
# </editor-fold>

# <editor-fold desc="Main simulation">
for i in range(parameters['num_cells']):
    print('cell {} of {}.'.format(i + 1, parameters['num_cells']), end='\r')
    cell = LFPy.Cell(**parameters['cell_parameters'])#**cell_parameters)
    cell.set_rotation(x=4.99, y=-4.33)  # let rotate around z-axis

    if i <= cutoff:
        insert_synapses(parameters['synapse_parameters'], 'dend', parameters['n_syn']
                        ,parameters['heights'],parameters['firings'])
    else:
        insert_synapses(parameters['synapse_parameters'], 'apic', parameters['n_syn']
                        ,parameters['heights'],parameters['firings'])

    cell.simulate(rec_imem=True, rec_vmem=True, rec_current_dipole_moment=True)
    P = cell.current_dipole_moment
    p.append(np.asarray(P))
    v.append(cell.vmem[0])
print("                                 ", end='\r')
print("done")
# </editor-fold>

# <editor-fold desc="Mean">
bas = np.mean(p[0:cutoff], axis=0)[:, 0]
api = np.mean(p[cutoff:], axis=0)[:, 0]
signal_mean = np.mean(p, axis=0)[:, 0]
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
print(factors)
# factors=[1,1,1]
# print(factors)
# </editor-fold>
opp = []
ned=[]
for i in range(parameters['num_cells']):
    if i<=cutoff:
         opp.append(v[i])
    else:
        ned.append(v[i])

# <editor-fold desc="Main plot">
plt.figure()
plt.plot(cell.tvec, signal_mean, color='gray', ls='--', label='total')
plt.plot(lines[0], lines[1] * factors[0], 'k')

plt.plot(cell.tvec, bas, label='dend', color='orange', ls='--')
plt.plot(x_line, fitses[0] * factors[1], color='red', ls='-')
vmemline = 65+np.mean(opp,axis=0)
vl = np.max(vmemline)
fi = np.max(fitses[0]*factors[1])
plt.plot(cell.tvec,(fi/vl)*vmemline,'g')

plt.plot(cell.tvec, api, label='apic', color='cyan', ls='--')
plt.plot(x_line, fitses[1] * factors[2], color='blue', ls='-')
vmemline=-65-np.mean(ned,axis=0)
vl = np.min(vmemline)
fi = np.min(fitses[1]*factors[2])
plt.plot(cell.tvec,(fi/vl)*vmemline,'g')

plt.xlabel('time [ms]')
plt.ylabel('Voltage [some unit]')
plt.title('{}'.format(suffix))
plt.legend()
plt.savefig('{}/signal_{}.pdf'.format(FOLDER,suffix))
# </editor-fold>

# <editor-fold desc="Main plot smoothed">
smooth_sig = ss.savgol_filter(signal_mean,5,3)
smooth_bas = ss.savgol_filter(bas,5,3)
smooth_api = ss.savgol_filter(api,5,3)
plt.figure()
plt.plot(cell.tvec, smooth_sig, color='gray', ls='--', label='total')
plt.plot(lines[0], lines[1] * factors[0], 'k')

plt.plot(cell.tvec, smooth_bas, label='dend', color='orange', ls='--')
plt.plot(x_line, fitses[0] * factors[1], color='red', ls='-')

plt.plot(cell.tvec, smooth_api, label='apic', color='cyan', ls='--')
plt.plot(x_line, fitses[1] * factors[2], color='blue', ls='-')

plt.xlabel('time [ms]')
plt.ylabel('Voltage [some unit]')
plt.title('{}'.format(suffix))
plt.legend()
plt.savefig('{}/signal_smoothed_{}.pdf'.format(FOLDER,suffix))
# </editor-fold>

# <editor-fold desc = "Firing rate plots">
binny = np.linspace(0, 20, 200)
plt.figure()
plt.hist(dist_E, bins=binny, color='red')
plt.title('{}'.format(suffix))
plt.savefig('{}/dend_{}.pdf'.format(FOLDER,suffix))
plt.figure()
plt.hist(dist_I, bins=binny, color='blue')
plt.title('{}'.format(suffix))
plt.savefig('{}/api_{}.pdf'.format(FOLDER,suffix))
# </editor-fold>

# <editor-fold desc="Vmem plot">
fig,axs = plt.subplots(2)
for i in range(parameters['num_cells']):
    if i<=cutoff:
        axs[0].plot(cell.tvec, v[i])
    else:
        axs[1].plot(cell.tvec,v[i])
axs[0].set_title('basal')
axs[1].set_title('apical')
fig.suptitle('{}'.format(suffix))
fig.savefig('{}/vmems_{}.pdf'.format(FOLDER,suffix))
# </editor-fold>

# <editor-fold desc="dump param">
with open('{}/parameters.txt'.format(FOLDER),'w') as f:
    print(parameters,file=f)
# </editor-fold>