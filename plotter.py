import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.optimize import minimize
import pandas
import pickle
sys.path.insert(0, '/dependencies/')
from dependencies import plotting_convention

#LOAD DATA
fits = pickle.load(open('data/fit_all_channels_test.p','rb'))
data = pickle.load(open('data/LFP_lines.p','rb'))
data = data[1] #inhib

num_channels = 16
# num_channels=1
plt.figure()
for i in range(num_channels):
    #normalized together, not sure if needed
    plt.plot(np.linspace(0,40,len(fits[0])),fits[i]/np.max(np.abs(data[i]))-i,'r')
    plt.plot(np.linspace(0,40,len(data[0])),data[i]/np.max(np.abs(data[i]))-i,'k')
plt.savefig('plots/Laminar_failure.jpg')
# plt.show()