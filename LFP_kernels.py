# <editor-fold desc="import packages">
import os
import argparse
import numpy as np
import LFPy
import pickle
import matplotlib.pyplot as plt
import time
import scipy.signal as ss
from matplotlib.collections import PolyCollection
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
    fitses = pickle.load(open('data/fits_inhibition_on.p', 'rb'))
    lines=pickle.load(open('data/lines_inhibition_on.p','rb'))
else:
    fitses = pickle.load(open('data/fits_inhibition_off.p', 'rb'))
    lines = pickle.load(open('data/lines_inhibition_off.p', 'rb'))
# </editor-fold>
"""
# <editor-fold desc="make dir">
folder_time = time.strftime("%m%d-%H%M")
try:
    os.makedirs(os.path.join('plots', '{}_{}'.format(suffix,folder_time)))
except OSError:
    pass
FOLDER = os.path.join('plots', '{}_{}'.format(suffix,folder_time))
# </editor-fold>
"""
# <editor-fold desc="Define functions">
def insert_synapses(synparams, section, n,heights,firings):
    soma_h = heights['soma']
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
    '''find n compartments to insert synapses onto'''
    idx = cell.get_rand_idx_area_norm(section=section, nidx=n, z_min=soma_h+minim, z_max=soma_h+maxim)
    # idx = cell.get_rand_idx_area_norm(section=section, nidx=n, z_min=minim, z_max=maxim)
    # Insert synapses in an iterative fashion
    for i in idx:
        synparams.update({'idx': int(i)})
    #     # Create synapse(s) and setting times using the Synapse class in LFPy
        s = LFPy.Synapse(cell, **synparams)
        # s.set_spike_times(np.random.choice(dist, firings_per_synapse))
        s.set_spike_times(np.array([25.]))
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
    # 'z' : np.linspace(-500,1000,16),
    'z' : np.linspace(-1600, -100, 16),
    'n' : 20,
    'r' : 10,
    'N' : N,
}



parameters={
    'num_cells':10,
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
        'soma': -1250.,
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
        'tstop':  50.,  # stop simulation at 100 ms.
    },
    'synapse_parameters':{
        'idx': [],
        'e':0.,
        'syntype': 'ExpSynI',
        'tau':.1,
        'weight': .1001,  # synaptic weight
        'record_current': True,  # record synapse current
    },
}

p = [] #dipole moment
v = [] #vmem
im = []
x_line = np.linspace(0, parameters['cell_parameters']['tstop'], len(fitses[0]))
# </editor-fold>
elleffpe = [9]
syni=[]
# <editor-fold desc="Main simulation">
for i in range(parameters['num_cells']):
    print('cell {} of {}.'.format(i + 1, parameters['num_cells']), end='\r')
    cell = LFPy.Cell(**parameters['cell_parameters'])#**cell_parameters)
    cell.set_rotation(x=4.99, y=-4.33)  # let rotate around z-axis
    cell.set_pos(x=np.random.normal(loc=0,scale=100),y=np.random.normal(loc=0,scale=100),z=np.random.normal(loc=parameters['heights']['soma'],scale=10))
    insert_synapses(parameters['synapse_parameters'], 'apic', parameters['n_syn']
                    ,parameters['heights'],parameters['firings'])


    cell.simulate(rec_imem=True, rec_vmem=True, rec_current_dipole_moment=True)
    P = cell.current_dipole_moment
    p.append(np.asarray(P))
    v.append(cell.vmem[0])
    im.append(cell.imem)
    elec = LFPy.RecExtElectrode(cell,**electrodeParameters)
    elec.calc_lfp()
    elleffpe.append(elec.LFP)
print("                                 ", end='\r')
print("done")
syni.append(cell.synidx)
syni = cell.synidx
ehh = np.mean(elleffpe,axis=0)
# plt.figure()
mupp = []
for i in range(16):
    mupp.append(ehh[i]-ehh[i][0])

# for i in range(16):
#     plt.plot(cell.tvec,mupp[i]/np.max(np.abs(mupp))+(i))
#
# plt.yticks(np.arange(0,16),labels=np.arange(1,17))
# plt.gca().invert_yaxis()

p = [] #dipole moment
v = [] #vmem
im = []
x_line = np.linspace(0, parameters['cell_parameters']['tstop'], len(fitses[0]))
# </editor-fold>
elleffpe = [9]
# <editor-fold desc="Main simulation">
for i in range(parameters['num_cells']):
    print('cell {} of {}.'.format(i + 1, parameters['num_cells']), end='\r')
    cell = LFPy.Cell(**parameters['cell_parameters'])#**cell_parameters)
    cell.set_rotation(x=4.99, y=-4.33)  # let rotate around z-axis
    cell.set_pos(x=np.random.normal(loc=0,scale=100),y=np.random.normal(loc=0,scale=100),z=np.random.normal(loc=parameters['heights']['soma'],scale=10))
    insert_synapses(parameters['synapse_parameters'], 'dend', parameters['n_syn']
                    ,parameters['heights'],parameters['firings'])


    cell.simulate(rec_imem=True, rec_vmem=True, rec_current_dipole_moment=True)
    P = cell.current_dipole_moment
    p.append(np.asarray(P))
    v.append(cell.vmem[0])
    im.append(cell.imem)
    elec = LFPy.RecExtElectrode(cell,**electrodeParameters)
    elec.calc_lfp()
    elleffpe.append(elec.LFP)
print("                                 ", end='\r')
print("done")

ehh = np.mean(elleffpe,axis=0)
# plt.figure()
mupp2 = []
for i in range(16):
    mupp2.append(ehh[i]-ehh[i][0])
pickle.dump([mupp,mupp2],open('data/L5.p','wb'))
#
# # plt.show()
fig = plt.figure()
ax = fig.add_axes([.4,.1,.55,.8], aspect='equal', frameon=False)
# #plot morphology
zips = []
for x, z in cell.get_idx_polygons():
    zips.append(list(zip(x, z)))
polycol = PolyCollection(zips,
                         edgecolors='none',
                         facecolors='k',
                         alpha=0.5)
ax.add_collection(polycol)

# ax.plot([100, 200], [-400+parameters['heights']['soma'], -400+parameters['heights']['soma']], 'k', lw=1, clip_on=False)
# ax.text(150, -470+parameters['heights']['soma'], r'100$\mu$m', va='center', ha='center')
# ax.plot([-550, -550], [100+parameters['heights']['soma'], 200+parameters['heights']['soma']], 'k', lw=1, clip_on=False)
# ax.text(-670, 150+parameters['heights']['soma'], r'100$\mu$m', va='center', ha='center')

# ax.axis('off')

ax.plot([-450,-450],[-1600,-1600],'y*')
ax.plot(cell.xmid[cell.synidx],cell.zmid[cell.synidx], 'o', ms=1,
        markeredgecolor='cyan',
        markerfacecolor='cyan',
        alpha=0.5)


ax.plot(cell.xmid[syni],cell.zmid[syni], 'o', ms=1,
        markeredgecolor='orange',
        markerfacecolor='orange',
        alpha=0.5)

for i in range(16):
    ax.plot(cell.tvec*20-550,65*mupp[i]/np.max(np.abs(mupp))+(i*100)-300+parameters['heights']['soma'],'r',lw=0.5)
    # ax.plot(cell.tvec*20-550,30*mupp[i]/np.max(np.abs(mupp[i]))+(i*100)-300+parameters['heights']['soma'],'r')

for i in range(16):
    ax.plot(cell.tvec*20-550,65*mupp2[i]/np.max(np.abs(mupp2))+(i*100)-300+parameters['heights']['soma'],'b',lw=0.5)
    # ax.plot(cell.tvec*20-550,30*mupp2[i]/np.max(np.abs(mupp2[i]))+(i*100)-300+parameters['heights']['soma'],'b')
ax.set_yticks(np.linspace(-50,-1550,16))
ax.set_yticklabels(np.linspace(100,1600,16).astype(int))
ax.set_xticks(np.linspace(-550,450,5))
ax.set_xticklabels(np.linspace(0,50,5))
ax.set_xlabel('time [ms]')
ax.set_ylabel('depth [um]')
plt.savefig('plots/L5_kernels.jpg')
# plt.show()

# print("for å lage l2/3, samle imem over inervaller som 100-200,200-300 etc, summere og halvere rangen, kanksje? "
#       "bør i såfall kjøre dybder ned til 3200 for å få l23 lfp helt ned?")