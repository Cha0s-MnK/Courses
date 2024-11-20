"""
Function: Configuration file of Python scripts used by Cha0s_MnK
Usage:    ~/*.py: from config import *
Version:  Last edited by Cha0s_MnK on 2024-11-15 (UTC+08:00).
"""

#########################
# CONFIGURE ENVIRONMENT #
#########################

import glob
import h5py
import imageio.v2 as imageio
import matplotlib
import matplotlib.colors as mcolours
from matplotlib.gridspec import GridSpec as gs
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Times New Roman",
                     'mathtext.default': 'regular',
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'text.usetex': True})
import numpy as np
#np.set_printoptions(threshold=np.inf, precision=2, suppress=True, linewidth=128)
import os
from pathlib import Path
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import minimize_scalar
import sys
from typing import List, Tuple
import warnings

###################
# SET ARGUMENT(S) #
###################

# physical constants (reference: https://en.wikipedia.org/wiki/List_of_physical_constants)
c         = 2.99792458e8        # speed of light [m·s⁻¹]
e         = 1.602176634e-19     # elementary charge [C]
epsilon_0 = 8.854187817e-12     # vacuum permittivity [F·m⁻¹]
G         = 4.301220369e-3      # gravitational constant [pc·(km/s)²·M☉⁻¹]
h         = 6.62607015e-34      # Planck constant [J·s]
k_B       = 1.380649e-23        # Boltzmann constant [J·K⁻¹]
m_e       = 9.109383713928e-31  # electron mass [kg]
m_p       = 1.6726219259552e-27 # proton mass [kg]

BOX_SIZE = np.float64(1.0)
DPI_MIN  = np.int64(256)
FLOAT    = np.float64
FPS      = 24
INT      = np.int64

######################
# HELPER FUNCTION(S) #
######################

def set_fig(ax: matplotlib.axes.Axes, equal: bool = False, grid: bool = True, legend: bool = True, 
            title: str = None, xlabel: str = None, ylabel: str = None, xlim: list = None, ylim: list = None,
            xlog: bool = False, ylog: bool = False, xticks: list = None, yticks: list = None):
    """
    Set various properties for a given axis.

    Parameter(s):
    - ax     (matplotlib.axes.Axes): the axis to modify
    - equal  (bool):                 whether to set equal scaling for the x and y axes
    - grid   (bool):                 whether to show the grid
    - legend (bool):                 whether to show the legend
    - title  (str):                  title for the axis
    - xlabel (str):                  label for the x-axis
    - ylabel (str):                  label for the y-axis
    - xlim   (list):                 limits for the x-axis [xmin, xmax]
    - ylim   (list):                 limits for the y-axis [ymin, ymax]
    - xlog   (bool):                 whether to use a logarithmic scale for the x-axis
    - ylog   (bool):                 whether to use a logarithmic scale for the y-axis
    - xticks (list):                 custom tick locations for the x-axis
    - yticks (list):                 custom tick locations for the y-axis
    """
    if equal:
        ax.axis('equal')
    if grid:
        ax.grid(True, which='both', linestyle=':')
    if legend:
        ax.legend()
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)

def save_fig(fig: matplotlib.figure.Figure, name: str, plots_dir: str = os.getcwd(), suptitle: str = None):
    """
    Save a Matplotlib figure as a PNG file with optional supertitle.

    Parameter(s):
    - fig (matplotlib.figure.Figure): the figure object to save
    - name (str):                     the base name of the file (without extension) to save the figure as
    - plots_dir (str):                the directory where the figure should be saved; defaults to the current
                                      working directory.
    - suptitle (str):                 a supertitle to add to the figure; defaults to None
    """
    png_path = Path(plots_dir) / f'{name}.png'
    if suptitle:                                     # set the supertitle if provided
        fig.suptitle(suptitle)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning) # ignore UserWarnings
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])  # adjust layout to prevent overlap: left, bottom, right, top
    fig.savefig(png_path)                            # save the figure
    print(f'Saved: {png_path}')                      # print confirmation message
    plt.close(fig)                                   # close the figure to free memory