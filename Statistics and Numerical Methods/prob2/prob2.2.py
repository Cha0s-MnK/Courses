"""
Function: Solution to Problem 2.2 in Statistics & Numerical Methods.
Usage:    python3.9 prob2.2.py
Version:  Last edited by Cha0s_MnK on 2024-10-29 (UTC+08:00).
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
from scipy.optimize import minimize_scalar
import warnings

# physical constants (reference: https://en.wikipedia.org/wiki/List_of_physical_constants)
c         = 2.99792458e8        # speed of light [m·s⁻¹]
e         = 1.602176634e-19     # elementary charge [C]
epsilon_0 = 8.854187817e-12     # vacuum permittivity [F·m⁻¹]
h         = 6.62607015e-34      # Planck constant [J·s]
k_B       = 1.380649e-23        # Boltzmann constant [J·K⁻¹]
m_e       = 9.109383713928e-31  # electron mass [kg]
m_p       = 1.6726219259552e-27 # proton mass [kg]

DPI = np.int64(1024)

###################
# SET ARGUMENT(S) #
###################

# load the data
data   = np.loadtxt('stars.dat')
xs     = data[:, 0]
ys     = data[:, 1]
sigmas = data[:, 2]

Delta_x__s   = 1e-6 # small perturbation for numerical gradients
N            = 5    # by observation
NumMaxIterat = 2000
varepsilon   = 1e-2

# plot arguments
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

# abolished
def calc_grad(As, xs__s):
    """gradient of chi-squared with respect to A_j and x_s_j"""
    grads_A    = np.zeros_like(As)
    grads_x__s = np.zeros_like(xs__s)
    ys_hat     = np.zeros_like(ys)

    # compute predicted ys
    for j in range(N):
        ys_hat += As[j] * PSF(x = xs - xs__s[j])

    for j in range(N):
        # compute gradient with respect to A_j
        grad_A_j_i     = - 2 * PSF(x = xs - xs__s[j]) * (ys - ys_hat) / sigmas**2
        grads_A[j]    += np.sum(grad_A_j_i)
        # compute gradient with respect to x_j__s
        grads_x__s[j] += np.sum(As[j] * (xs - xs__s[j]) * grad_A_j_i)

    return grads_A, grads_x__s
# abolished
def optimize_gradient_descent(As_init, xs_init__s):
    """custom optimization function using gradient descent"""
    As    = As_init
    xs__s = xs_init__s
    for NumIterat in range(1, NumMaxIterat):
        # compute gradients
        grads_A, grads_x__s = calc_grad(As = As, xs__s = xs__s)

        # update parameters
        As    -= alpha * grads_A
        xs__s -= alpha * grads_x__s

        # Compute chi-squared for convergence check
        chi_sq = chi_squared(As=As, xs__s=xs__s)

        # Print progress
        if NumIterat % 10 == 0:
            print(f"After {NumIterat} iterations, chi-squared = {chi_sq}.")

        # Check for convergence
        if np.linalg.norm(np.concatenate([grads_A, grads_x__s])) < varepsilon:
            print(f"Converged in {NumIterat} iterations.")
            break
    else:
        print("Maximum iterations reached without convergence.")

    return As, xs__s
# abolished
def optimize_steepest_gradient(As_init, xs_init__s):
    """optimize using steepest gradient descent (with line search minimization)"""
    As     = As_init.copy()
    xs__s  = xs_init__s.copy()
    params = np.concatenate([As, xs__s])

    for NumIterat in range(1, NumMaxIterat):
        # Compute gradients
        grads_A, grads_x__s = calc_grad(As = As, xs__s = xs__s)
        grads = np.concatenate([grads_A, grads_x__s])

        # Define the line search function
        def line_func(alpha):
            params_new = params - alpha * grads
            return chi_squared(As = params_new[:N], xs__s = params_new[N:])

        # perform line search to find optimal alpha
        res = minimize_scalar(line_func, bounds=(0, 1), method='bounded')
        optimal_alpha = res.x

        # Update parameters
        params -= optimal_alpha * grads
        As      = params[:N]
        xs__s   = params[N:]

        # Compute chi-squared for convergence check
        chi_sq = chi_squared(As = As, xs__s = xs__s)

        # Print progress
        if NumIterat % 10 == 0:
            print(f"Iteration {NumIterat}: chi-squared = {chi_sq}, alpha = {optimal_alpha}")

        # Check for convergence
        if np.linalg.norm(grads) < varepsilon:
            print(f"Converged in {NumIterat} iterations.")
            break
    else:
        print("Maximum iterations reached without convergence.")

    return As, xs__s
# abolished
def optimize_conjugate_gradient(As_init, xs_init__s):
    """optimize using the conjugate gradient descent method"""
    As = As_init.copy()
    xs__s = xs_init__s.copy()

    params = np.concatenate([As, xs__s])

    # Compute initial gradients
    grads_A, grads_x__s = calc_grad(As=As, xs__s=xs__s)
    grads = np.concatenate([grads_A, grads_x__s])
    direction = -grads  # Initial search direction

    for NumIterat in range(1, NumMaxIterat + 1):
        # Define the line search function along the conjugate direction
        def line_func(alpha):
            new_params = params + alpha * direction
            new_As = new_params[:N]
            new_xs__s = new_params[N:]
            return chi_squared(As=new_As, xs__s=new_xs__s)

        # Perform line search to find optimal alpha
        res = minimize_scalar(line_func, bounds=(0, 1), method='bounded')
        optimal_alpha = res.x

        # Update parameters
        params += optimal_alpha * direction
        As = params[:N]
        xs__s = params[N:]

        # Compute new gradients
        grads_A_new, grads_x__s_new = calc_grad(As=As, xs__s=xs__s)
        grads_new = np.concatenate([grads_A_new, grads_x__s_new])

        # Compute beta using the Polak-Ribiere formula
        beta_num = np.dot(grads_new - grads, grads_new)
        beta_den = np.dot(grads, grads)
        beta = beta_num / beta_den

        # Update direction
        direction = -grads_new + beta * direction

        # Update gradients
        grads = grads_new.copy()

        # Compute chi-squared for convergence check
        chi_sq = chi_squared(As=As, xs__s=xs__s)

        # Print progress
        if NumIterat % 10 == 0:
            print(f"Iteration {NumIterat}: chi-squared = {chi_sq}, alpha = {optimal_alpha}")

        # Check for convergence
        if np.linalg.norm(grads) < varepsilon:
            print(f"Converged in {NumIterat} iterations.")
            break
    else:
        print("Maximum iterations reached without convergence.")

    return As, xs__s

def calc_design_matrix(xs__s):
    """compute the design matrix varphi_matrix based on positions xs and source positions xs__s"""
    varphi_matrix = np.zeros((len(xs), N))
    for j in range(N):
        varphi_matrix[:, j] = PSF(xs - xs__s[j])
    return varphi_matrix

def solveAs(xs__s):
    """solve for As using weighted linear least squares given source positions xs__s"""
    varphi_matrix = calc_design_matrix(xs__s = xs__s)
    varphi_W      = varphi_matrix.T @ np.diag(1 / sigmas**2) # varphi^T * weight matrix
    # solve the weighted normal equations
    return np.linalg.solve(varphi_W @ varphi_matrix, varphi_W @ ys)

def chi_sq(As, xs__s):
    """chi-squared function"""
    varphi_matrix = calc_design_matrix(xs__s = xs__s)
    return np.sum(((ys - (varphi_matrix @ As)) / sigmas)**2)

def calc_grads_x__s(As, xs__s):
    """compute the gradient of chi-squared function with respect to xs__s numerically"""
    grads_x__s = np.zeros_like(xs__s)
    chi_sq0    = chi_sq(As = As, xs__s = xs__s)
    Delta_x__s = 1e-5  # Small perturbation for numerical derivative
    for j in range(len(xs__s)):
        xs_perturb__s     = xs__s.copy()
        xs_perturb__s[j] += Delta_x__s
        # Compute the numerical derivative
        grads_x__s[j] = (chi_sq(As = As, xs__s = xs_perturb__s) - chi_sq0) / Delta_x__s
    return grads_x__s

def optimize_alter(As_init, xs_init__s):
    """optimize using alternating optimization algorithm"""
    xs__s = xs_init__s.copy()
    for NumIterat in range(NumMaxIterat):
        # step 1: solve for As given xs__s using weighted linear least squares
        As = solveAs(xs__s = xs__s)

        # step 2: optimize xs__s given As using steepest gradient descent with line minimization search
        # compute the gradient of chi-squared function with respect to xs__s numerically
        grads_x__s = calc_grads_x__s(As = As, xs__s = xs__s)

        # line search along the gradient direction to minimize chi^2
        def chi_sq_alpha(alpha):
            return chi_sq(As = As, xs__s = xs__s - alpha * grads_x__s)

        res     = minimize_scalar(chi_sq_alpha)
        alpha   = minimize_scalar(chi_sq_alpha).x
        xs__s  -= alpha * grads_x__s

        # check for convergence
        if np.linalg.norm(grads_x__s) < varepsilon:
            print(f"Method converges in {NumIterat} iterations.")
            break

        # compute chi-squared and print progress
        if NumIterat % 10 == 0:
            print(f"after {NumIterat} iterations, chi-squared = {chi_sq(As = As, xs__s = xs__s)}")
    else:
        print(f"Method not converges after {NumMaxIterat} iterations.")

    # solve for As with optimized xs__s
    return  solveAs(xs__s), xs__s

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

    # problem 2.2.3

    # run the optimization
    As_init    = [0.75, 0.75, 0.75, 0.55, 0.85]  # initial guess for amplitudes
    xs_init__s = [-26.0, -4.8, -2.6, 6.3, 14.4]  # initial guess for star positions
    As, xs__s = optimize_alter(As_init = As_init, xs_init__s = xs_init__s)

    # create a figure and plot the brightness with 1sigma error bars
    fig, ax = plt.subplots(1, 1, figsize=(Xsize, Ysize), dpi = DPI)
    ax.errorbar(xs, ys, yerr = sigmas, fmt = 'o', markersize = 2, color = '#1f3b4d', ecolor = 'gray',
                elinewidth = 1, capsize = 2, capthick = 1, alpha = 0.8, label = 'observation')

    # Generate the fitted model with optimized parameters
    ys_hat = np.zeros_like(ys)
    for j in range(N):
        ys_hat += As[j] * PSF(x = xs - xs__s[j])
    ax.plot(xs, ys_hat, 'r-', label='fitted model')

    # plot settings and save
    set_fig(ax = ax, xlim = Xlim, xlabel = r'position $x$', ylabel = r'brightness $y$',
            title = r'Fitted Model with Alternating Optimization')
    save_fig(fig = fig, name = f'prob2.2.2')

    # print optimized parameters
    print("Optimized amplitudes:", As)
    print("Optimized positions: ", xs__s)

if __name__ == "__main__":
    main()