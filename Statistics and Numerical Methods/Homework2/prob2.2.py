"""
Function: Solution to Problem 2.2 in Statistics & Numerical Methods.
Usage:    python3.9 prob2.2.py
Version:  Last edited by Cha0s_MnK on 2024-10-22 (UTC+08:00).
"""

#########################
# CONFIGURE ENVIRONMENT #
#########################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Times New Roman",
                     'mathtext.default': 'regular',
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'text.usetex': True})
import os
from pathlib import Path
from scipy.integrate import quad
from scipy.optimize import fsolve
import warnings

# physical constants (reference: https://en.wikipedia.org/wiki/List_of_physical_constants)
c         = 2.99792458e8        # speed of light [m·s⁻¹]
e         = 1.602176634e-19     # elementary charge [C]
epsilon_0 = 8.854187817e-12     # vacuum permittivity [F·m⁻¹]
h         = 6.62607015e-34      # Planck constant [J·s]
k_B       = 1.380649e-23        # Boltzmann constant [J·K⁻¹]
m_e       = 9.109383713928e-31  # electron mass [kg]
m_p       = 1.6726219259552e-27 # proton mass [kg]

DPI = np.int64(512)

###################
# SET ARGUMENT(S) #
###################

# Load the data from the file and Extract the columns: position (x), brightness (y), and noise (dy)
data   = np.loadtxt('stars.dat')
xs     = data[:, 0]
ys     = data[:, 1]
sigmas = data[:, 2]

N            = 5
NumMaxIterat = 2000
varepsilon   = 1e-6

Xlim  = [np.float64(-41), np.float64(41)]
Xsize = np.int64(15)
Ysize = np.int64(5)

######################
# HELPER FUNCTION(S) #
######################

def set_fig(ax: matplotlib.axes.Axes, xlim: list = None, ylim: list = None, xticks: list = None,
            yticks: list = None, xlabel: str = None, ylabel: str = None, title: str = None,
            legend: bool = True, grid: bool = True):
    """
    Set various properties for a given axis.

    Parameter(s):
    - ax (matplotlib.axes.Axes): the axis to modify
    - xlim (list):               limits for the x-axis [xmin, xmax]
    - ylim (list):               limits for the y-axis [ymin, ymax]
    - xticks (list):             custom tick locations for the x-axis
    - yticks (list):             custom tick locations for the y-axis
    - xlabel (str):              label for the x-axis
    - ylabel (str):              label for the y-axis
    - title (str):               title for the axis
    - legend (bool):             whether to show the legend
    - grid (bool):               whether to show the grid
    """
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend()
    if grid:
        ax.grid(True, which='both', linestyle=':')

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

def PSF(x):
    """point spread function (PSF)"""
    return np.exp(- x**2 / 2) / np.sqrt(2 * np.pi)

def chi_squared(As, xs__s):
    """chi-squared function"""
    chi_squared = 0
    ys_hat      = np.zeros_like(ys)
    for i in range(len(xs)):
        for j in range(N):
            ys_hat[i] += As[j] * PSF(xs[i] - xs__s[j])
        chi_squared += ((ys[i] - ys_hat[i]) / sigmas[i])**2
    return chi_squared

def calc_grad(As, xs__s):
    """gradient of chi-squared with respect to A_j and x_s_j"""
    grads_A    = np.zeros_like(As)
    grads_x__s = np.zeros_like(xs__s)
    ys_hat     = np.zeros_like(ys)
    for j in range(N):
        for i in range(len(xs)):
            for k in range(N):
                ys_hat[i] += As[k] * PSF(x = xs[i] - xs__s[k])
            
            # compute gradient with respect to A_j
            grad_A_j_i  = - 2 * PSF(x = xs[i] - xs__s[j]) * (ys[i] - ys_hat[i]) / sigmas[i]**2
            grads_A[j] += grad_A_j_i
    
            # compute gradient with respect to x_j__s
            grads_x__s[j] += As[j] * (xs[i] - xs__s[j]) * grad_A_j_i

    return grads_A, grads_x__s

def optimize_gradient_descent(As_init, xs_init__s, learning_rate = 0.01):
    """custom optimization function using gradient descent"""
    As    = As_init
    xs__s = xs_init__s
    for iteration in range(NumMaxIterat):
        # compute gradients
        grads_A, grads_x__s = calc_grad(As = As, xs__s = xs__s)

        # update parameters
        As    -= learning_rate * grads_A
        xs__s -= learning_rate * grads_x__s

        # check for convergence
        if chi_squared(As = As, xs__s = xs__s) < 1:
            print(f"Converged in {iteration} iterations.")
            break

    return As, xs__s

#################
# MAIN FUNCTION #
#################

def main():
    # problem 2.2.1

    # create a figure and plot the brightness with 1sigma error bars
    fig, ax = plt.subplots(1, 1, figsize=(Xsize, Ysize), dpi = DPI)
    ax.errorbar(xs, ys, yerr = sigmas, fmt = 'o', markersize = 2, color = '#1f3b4d', ecolor = 'gray',
                elinewidth = 1, capsize = 2, capthick = 1, alpha = 0.8, label = 'observation')

    # plot settings and save
    set_fig(ax = ax, xlim = Xlim, xlabel = r'position $x$', ylabel = r'brightness $y$',
            title = r'1D Image of Stars with 1$\sigma$ Error Bars')
    save_fig(fig = fig, name = f'prob2.2.1')

    # infer by observation
    N = 5

    # problem 2.2.3

    # run the optimization
    As_init    = [0.75, 0.75, 0.75, 0.55, 0.85]  # initial guess for amplitudes
    xs_init__s = [-26.0, -4.8, -2.6, 6.3, 14.4]  # initial guess for star positions
    As, xs__s = optimize_gradient_descent(As_init = As_init, xs_init__s = xs_init__s)

    # create a figure and plot the brightness with 1sigma error bars
    fig, ax = plt.subplots(1, 1, figsize=(Xsize, Ysize), dpi = DPI)
    ax.errorbar(xs, ys, yerr = sigmas, fmt = 'o', markersize = 2, color = '#1f3b4d', ecolor = 'gray',
                elinewidth = 1, capsize = 2, capthick = 1, alpha = 0.8, label = 'observation')

    # Generate the fitted model with optimized parameters
    ys_hat = np.zeros_like(ys)
    for j in range(N):
        ys_hat += As[j] * PSF(x = xs - xs__s[j])
    ax.plot(xs, ys_hat, 'r-', label='predicted model')

    # plot settings and save
    set_fig(ax = ax, xlim = Xlim, xlabel = r'position $x$', ylabel = r'brightness $y$',
            title = r'Fitted Model with Custom Optimization')
    save_fig(fig = fig, name = f'prob2.2.2')

    # Print optimized parameters
    print("Optimized Amplitudes (A):", As)
    print("Optimized Star Positions (x_s):", xs__s)

if __name__ == "__main__":
    main()
