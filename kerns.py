import pickle
import matplotlib.pyplot as plt
import numpy as np

#read kernel data
L5 = pickle.load(open('data/L5.p','rb'))
L23 = pickle.load(open('data/L23.p','rb'))

# Create plot
fig,(ax1,ax2) = plt.subplots(1,2,sharey='row',gridspec_kw={'wspace':0})

# L5 kernel
for i in range(16):
    ax1.plot(65*L5[0][i]/np.max(np.abs(L5[0]))+(i*100),'r',lw=0.5)
for i in range(16):
    ax1.plot(65*L5[1][i]/np.max(np.abs(L5[1]))+(i*100),'b',lw=0.5)

ax1.set_xticks(np.linspace(0,400,6))
ax1.set_xticklabels(np.linspace(0,50,6).astype(int))
ax1.set_yticks(np.linspace(0,1500,16))
ax1.set_yticklabels(np.linspace(1600,100,16).astype(int))
ax1.set_xlabel('time [ms]')
ax1.set_ylabel('depth [um]')
ax1.set_title('L5 kernel')

# L2/3 kernel
for i in range(16):
    ax2.plot(65*L23[0][i]/np.max(np.abs(L23[0]))+(i*100),'r',lw=0.5)
for i in range(16):
    ax2.plot(65*L23[1][i]/np.max(np.abs(L23[1]))+(i*100),'b',lw=0.5)

ax2.set_xticks(np.linspace(0,400,6))
ax2.set_xticklabels(np.linspace(0,50,6).astype(int))
ax2.set_xlabel('time [ms]')
ax2.set_title('L2/3 kernel')

plt.savefig('plots/kernels.jpg')
# plt.show()