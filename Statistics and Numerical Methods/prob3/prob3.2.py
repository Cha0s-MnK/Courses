"""
Function: Solution to Problem 3.2 in Statistics & Numerical Methods.
Usage:    python3.11 prob3.2.py
Version:  Last edited by Cha0s_MnK on 2024-11-11 (UTC+08:00).
"""

###########################################
# CONFIGURE ENVIRONMENT & SET ARGUMENT(S) #
###########################################

from config import *

Delta_t     = 0.1                                               # timestep
M           = 10                                                # number of dust species
T           = 10.0                                              # total integration time
t_s         = 1.0                                               # stopping time
times       = np.arange(0, T + Delta_t, Delta_t)
ts_s        = np.array([10**(-6 + i) for i in range(1, M + 1)])
varepsilons = np.full(M, 0.1)

######################
# HELPER FUNCTION(S) #
######################

def backwardEuler1D(t_s, times, v_g):
    v_ds    = np.zeros_like(times) # dust velocities
    v_ds[0] = 0.0                  # initial dust velocity
    for n in range(len(times) - 1):
        v_ds[n+1] = (v_ds[n] * t_s + v_g(times[n+1]) * Delta_t) / (t_s + Delta_t)

    return v_ds

def backwardEuler(times):
    v_ds       = np.zeros((M, len(times))) # dust velocities
    v_ds[:, 0] = np.zeros(M)               # initial dust velocities
    v_gs       = np.zeros(len(times))      # gas velocities
    v_gs[0]    = 1.0                       # initial gas velocity

    # LHS of Equation 3.2.5
    N = M + 1
    A = np.zeros((N, N))
    b = np.zeros(N)
    alpha    = Delta_t / ts_s
    beta     = varepsilons * Delta_t / ts_s
    A[0, 0]  = 1.0 + np.sum(beta)
    A[0, 1:] = - beta
    for i in range(M):
        A[i+1, 0]   = - alpha[i]
        A[i+1, i+1] = 1.0 + alpha[i]

    for n in range(len(times) - 1):
        # RHS of Equation 3.2.5
        b[0] = v_gs[n]
        for i in range(M):
            b[i+1] = v_ds[i, n]

        # solve the linear system
        x = np.linalg.solve(A, b)

        # update velocities
        v_gs[n+1]    = x[0]
        v_ds[:, n+1] = x[1:]

    return v_gs, v_ds

#################
# MAIN FUNCTION #
#################

def main():
    global Delta_t, M, T, t_s, times
    # problem 3.2.4

    # numerical solution
    v_ds = backwardEuler1D(t_s = t_s, times = times, v_g = lambda t: 1.0)

    # plot 1
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi = 2 * DPI_MIN)
    ax.plot(times, v_ds, label=r'backward Euler ($\Delta t$ = 0.1)')
    ax.plot(times, np.ones_like(times), linestyle='--', label=r'gas velocity $v_\mathrm{g}$')
    ax.plot(times, 1 - np.exp(- times), label='analytical')
    set_fig(ax = ax, title = r'Time evolution of $v_\mathrm{d}$', xlabel = r'time $t$',
            ylabel = r'dust velocity $v_\mathrm{d}$', ylim = [-0.1, 1.1])
    save_fig(fig = fig, name = f'prob3.2.1')

    # numerical solution
    v_ds = backwardEuler1D(t_s = t_s, times = times, v_g=lambda t: np.sin(t))

    # plot 2
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi = 2 * DPI_MIN)
    ax.plot(times, v_ds, label=r'backward Euler ($\Delta t$ = 0.1)')
    ax.plot(times, np.sin(times), linestyle='--', label=r'gas velocity $v_\mathrm{g}$')
    set_fig(ax = ax, title = r'Time evolution of $v_\mathrm{d}$', xlabel = r'time $t$',
            ylabel = r'dust velocity $v_\mathrm{d}$')
    save_fig(fig = fig, name = f'prob3.2.2')

    # numerical solution
    t_s  = 1e-5
    v_ds = backwardEuler1D(t_s = t_s, times = times, v_g=lambda t: np.sin(t))

    # plot 3
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi = 2 * DPI_MIN)
    ax.plot(times, v_ds, label=r'backward Euler ($\Delta t$ = 0.1)')
    ax.plot(times, np.sin(times), linestyle='--', label=r'gas velocity $v_\mathrm{g}$')
    set_fig(ax = ax, title = r'Time evolution of $v_\mathrm{d}$', xlabel = r'time $t$',
            ylabel = r'dust velocity $v_\mathrm{d}$')
    save_fig(fig = fig, name = f'prob3.2.3')

    # problem 3.2.5
    Delta_t = 1.0
    T       = 1e5
    times   = np.arange(0, T + Delta_t, Delta_t)

    # numerical solution using backward Euler method
    v_gs, v_ds = backwardEuler(times = times)

    # plot 1: gas velocity and first 5 dust species velocities (first 10 steps)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi = 2 * DPI_MIN)
    ax.plot(times[:11], v_gs[:11], label=r'gas velocity $v_\mathrm{g}$')
    for i in range(5):
        ax.plot(times[:11], v_ds[i, :11], label=f'dust species {i+1} velocity')
    set_fig(ax = ax, title = 'Gas and 1st 5 dust species velocities (1st 10 steps)', xlabel = r'time $t$',
            ylabel = r'velocity $v$', ylim = [-0.1, 1.1])
    save_fig(fig = fig, name = f'prob3.2.4')

    # plot 2: gas velocity and remaining 5 dust species velocities (logarithmic time scale)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi = 2 * DPI_MIN)
    ax.plot(times[1:], v_gs[1:], label=r'gas velocity $v_\mathrm{g}$')
    for i in range(5, M):
        ax.plot(times[1:], v_ds[i, 1:], label=f'dust species {i+1} velocity')
    set_fig(ax = ax, title = 'Gas and remaining 5 dust species velocities', xlabel = r'time $t$',
            xlog=True, ylabel = r'velocity $v$', ylim = [-0.1, 1.1])
    save_fig(fig = fig, name = f'prob3.2.5')

if __name__ == "__main__":
    main()