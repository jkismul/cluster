import matplotlib
matplotlib.rc('pdf', fonttype=42)
import pylab as plt
from matplotlib.colors import LogNorm

# THIS IS NOW MY PAPER FIGURE DEFAULT
plt.rcParams.update({
    'xtick.labelsize': 13,
    'xtick.major.size': 5,
    'ytick.labelsize': 13,
    'ytick.major.size': 5,
    'font.size': 17,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 16,
    'figure.subplot.wspace': 0.4,
    'figure.subplot.hspace': 0.4,
    'figure.subplot.left': 0.1,
})

def mark_subplots(axes, letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ', xpos=-0.12, ypos=1.15):

    if not type(axes) is list:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.text(xpos, ypos, letters[idx].capitalize(),
                horizontalalignment='center',
                verticalalignment='center',
                fontweight='demibold',
                fontsize=12,
                transform=ax.transAxes)

def simplify_axes(axes):

    if not type(axes) is list:
        axes = [axes]

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

def color_axes(axes, clr):
    if not type(axes) is list:
        axes = [axes]
    for ax in axes:
        ax.tick_params(axis='x', colors=clr)
        ax.tick_params(axis='y', colors=clr)
        for spine in ax.spines.values():
            spine.set_edgecolor(clr)