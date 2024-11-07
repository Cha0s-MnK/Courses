"""
Function: Solution to Problem 3.1 in Statistics & Numerical Methods.
Usage:    python3.11 prob3.1.py
Version:  Last edited by Cha0s_MnK on 2024-11-06 (UTC+08:00).
"""

###########################################
# CONFIGURE ENVIRONMENT & SET ARGUMENT(S) #
###########################################

from config import *

######################
# HELPER FUNCTION(S) #
######################

def createIC(t_init: FLOAT = 0.0, x_init: FLOAT = 1.0, y_init: FLOAT = 0.0, vx_init: FLOAT = 0.0,
             vy_init: FLOAT = 1.0) -> Tuple[List[FLOAT], List[FLOAT], List[FLOAT], List[FLOAT], List[FLOAT]]:
    """
    Initialize and return the ICs for time, position and velocity.

    Parameter(s):
    - t_init  (FLOAT, optional, default = 0.0): initial time value
    - x_init  (FLOAT, optional, default = 1.0): initial X-coordinate
    - y_init  (FLOAT, optional, default = 0.0): initial Y-coordinate
    - vx_init (FLOAT, optional, default = 0.0): initial X-component of the velocity
    - vy_init (FLOAT, optional, default = 1.0): initial Y-component of the velocity

    Return(s):
    - tuple: a tuple of lists containing the ICs:
        - ts  (List[FLOAT]): time values (initialized with t_init)
        - xs  (List[FLOAT]): X-coordinates (initialized with x_init)
        - ys  (List[FLOAT]): Y-coordinates (initialized with y_init)
        - vxs (List[FLOAT]): X-components of velocity (initialized with vx_init)
        - vys (List[FLOAT]): Y-components of velocity (initialized with vy_init)
    """

    # initialize lists with the ICs
    return [t_init], [x_init], [y_init], [vx_init], [vy_init]

def calcEs_Ls(xs: List[FLOAT], ys: List[FLOAT], vxs: List[FLOAT], vys: List[FLOAT]) -> \
    Tuple[List[FLOAT], List[FLOAT]]:
    """
    Calculate the total energy (Es) and angular momentum (Ls) for each timestep.

    Parameter(s):
    - xs  (List[FLOAT]): X-coordinates at each timestep
    - ys  (List[FLOAT]): Y-coordinates at each timestep
    - vxs (List[FLOAT]): X-components of velocity at each timestep
    - vys (List[FLOAT]): Y-components of velocity at each timestep

    Return(s):
    - tuple: a tuple containing
        - Es (List[FLOAT]): total energy values at each timestep
        - Ls (List[FLOAT]): angular momentum values at each timestep
    """
    Es = []
    Ls = []
    for x, y, vx, vy in zip(xs, ys, vxs, vys):
        Es.append(0.5 * (vx**2 + vy**2) - 1 / np.sqrt(x**2 + y**2))
        Ls.append(x * vy - y * vx)
    return Es, Ls

def calc_errors(Delta_ts: List[FLOAT], t_f: FLOAT, method) -> Tuple[List[FLOAT], List[FLOAT], List[FLOAT]]:
    """
    Calculate the final errors in position, total energy and angular momentum for different timesteps using 
    the specified method for numerical integration.

    Parameter(s):
    - Delta_ts (List[FLOAT]): a list of timesteps to use in the the specified numerical integration.
    - t_f      (FLOAT)      : the final time for the integration
    - method   (function)   : the method to use for numerical integration

    Return(s):
    - errs_f_pos (List[FLOAT]): a list of position errors at the final time for each timestep
    - errs_f_E   (List[FLOAT]): a list of energy errors at the final time for each timestep
    - errs_f_L   (List[FLOAT]): a list of angular momentum errors at the final time for each timestep
    """
    # lists to store errors for each timestep
    errs_f_pos = []
    errs_f_E   = []
    errs_f_L   = []

    # loop over each timestep
    for Delta_t in Delta_ts:
        ts_numeric, xs_numeric, ys_numeric, vxs_numeric, vys_numeric = method(Delta_t = Delta_t, t_f = t_f)
        Es_numeric, Ls_numeric = calcEs_Ls(xs = xs_numeric, ys = ys_numeric, vxs = vxs_numeric, vys = vys_numeric)

        errs_f_pos.append(np.sqrt((xs_numeric[-1] - np.cos(ts_numeric)[-1])**2 + (ys_numeric[-1] - np.sin(ts_numeric)[-1])**2))
        errs_f_E.append(np.abs(Es_numeric[-1] - Es_numeric[0]))
        errs_f_L.append(np.abs(Ls_numeric[-1] - Ls_numeric[0]))

    return errs_f_pos, errs_f_E, errs_f_L


def forwardEuler(Delta_t: FLOAT, t_f: FLOAT) -> \
    Tuple[List[FLOAT], List[FLOAT], List[FLOAT], List[FLOAT], List[FLOAT]]:
    """
    Perform forward Euler method to integrate positions and velocities over time.

    Parameter(s):
    - Delta_t (FLOAT): timestep size for the forward Euler method
    - t_f     (FLOAT): final time value to stop the integration

    Return(s):
    - tuple: a tuple of lists containing the time, positions and velocities
        - ts  (List[FLOAT]): time values
        - xs  (List[FLOAT]): X-coordinates
        - ys  (List[FLOAT]): Y-coordinates
        - vxs (List[FLOAT]): X-components of velocity
        - vys (List[FLOAT]): Y-components of velocity
    """
    # initialize the ICs
    ts, xs, ys, vxs, vys = createIC()

    # loop until the final time is reached
    while ts[-1] < t_f:
        # get the current state
        t, x, y, vx, vy = ts[-1], xs[-1], ys[-1], vxs[-1], vys[-1]

        # update time, positions and velocities
        t_next = t + Delta_t

        x_next = x + vx * Delta_t
        y_next = y + vy * Delta_t

        r3_inv = 1 / (np.sqrt(x**2 + y**2)**3)
        vx_next = vx - x * r3_inv * Delta_t
        vy_next = vy - y * r3_inv * Delta_t

        # append the updated values to the lists
        ts.append(t_next)
        xs.append(x_next)
        ys.append(y_next)
        vxs.append(vx_next)
        vys.append(vy_next)

    return ts, xs, ys, vxs, vys

def RK4(Delta_t: FLOAT, t_f: FLOAT) -> \
    Tuple[List[FLOAT], List[FLOAT], List[FLOAT], List[FLOAT], List[FLOAT]]:
    """
    Perform 4th-order Runge-Kutta method to integrate positions and velocities over time.

    Parameter(s):
    - Delta_t (FLOAT): timestep size for the forward Euler method
    - t_f     (FLOAT): final time value to stop the integration

    Return(s):
    - tuple: a tuple of lists containing the time, positions and velocities
        - ts   (List[FLOAT]): time values
        - xs   (List[FLOAT]): X-coordinates
        - ys   (List[FLOAT]): Y-coordinates
        - vs_x (List[FLOAT]): X-components of velocity
        - vs_y (List[FLOAT]): Y-components of velocity
    """
    # initialize the ICs
    ts, xs, ys, vxs, vys = createIC()

    # loop until the final time is reached
    while ts[-1] < t_f:
        # get the current state
        t, x, y, vx, vy = ts[-1], xs[-1], ys[-1], vxs[-1], vys[-1]

        # calculate k_1
        r3inv = 1 / (np.sqrt(x**2 + y**2)**3)
        k1_x  = vx
        k1_y  = vy
        k1_vx = - x * r3inv
        k1_vy = - y * r3inv

        # calculate k_2
        x_k2     = x + 0.5 * k1_x * Delta_t
        y_k2     = y + 0.5 * k1_y * Delta_t
        r3inv_k2 = 1 / (np.sqrt(x_k2**2 + y_k2**2)**3)
        k2_x     = vx + 0.5 * k1_vx * Delta_t
        k2_y     = vy + 0.5 * k1_vy * Delta_t
        k2_vx    = - x_k2 * r3inv_k2
        k2_vy    = - y_k2 * r3inv_k2

        # calculate k_3
        x_k3     = x + 0.5 * k2_x * Delta_t
        y_k3     = y + 0.5 * k2_y * Delta_t
        r3inv_k3 = 1 / (np.sqrt(x_k3**2 + y_k3**2)**3)
        k3_x     = vx + 0.5 * k2_vx * Delta_t
        k3_y     = vy + 0.5 * k2_vy * Delta_t
        k3_vx    = - x_k3 * r3inv_k3
        k3_vy    = - y_k3 * r3inv_k3

        # calculate k_4
        x_k4     = x + k3_x * Delta_t
        y_k4     = y + k3_y * Delta_t
        r3inv_k4 = 1 / (np.sqrt(x_k4**2 + y_k4**2)**3)
        k4_x     = vx + k3_vx * Delta_t
        k4_y     = vy + k3_vy * Delta_t
        k4_vx    = - x_k4 * r3inv_k4
        k4_vy    = - y_k4 * r3inv_k4
        
        # update time, positions and velocities
        inv6    = 1 / 6
        t_next  = t + Delta_t
        x_next  = x + inv6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) * Delta_t
        y_next  = y + inv6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y) * Delta_t
        vx_next = vx + inv6 * (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx) * Delta_t
        vy_next = vy + inv6 * (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy) * Delta_t

        # append the updated values to the lists
        ts.append(t_next)
        xs.append(x_next)
        ys.append(y_next)
        vxs.append(vx_next)
        vys.append(vy_next)

    return ts, xs, ys, vxs, vys

def leapfrog(Delta_t: FLOAT, t_f: FLOAT) -> \
    Tuple[List[FLOAT], List[FLOAT], List[FLOAT], List[FLOAT], List[FLOAT]]:
    """
    Perform leapfrog (kick-drift-kick, KDK) method to integrate positions and velocities over time.

    Parameter(s):
    - Delta_t (FLOAT): timestep size for the forward Euler method
    - t_f     (FLOAT): final time value to stop the integration

    Return(s):
    - tuple: a tuple of lists containing the time, positions and velocities
        - ts  (List[FLOAT]): time values
        - xs  (List[FLOAT]): X-coordinates
        - ys  (List[FLOAT]): Y-coordinates
        - vxs (List[FLOAT]): X-components of velocity
        - vys (List[FLOAT]): Y-components of velocity
    """
    # initialize the ICs
    ts, xs, ys, vxs, vys = createIC()

    # 1st kick (K) for velocity
    r3inv   = 1 / (np.sqrt(xs[0]**2 + ys[0]**2)**3)
    vx_half = vxs[0] - 0.5 * xs[0] * r3inv * Delta_t
    vy_half = vys[0] - 0.5 * ys[0] * r3inv * Delta_t

    # loop until the final time is reached
    while ts[-1] < t_f:
        # get the current state
        t, x, y, vx, vy = ts[-1], xs[-1], ys[-1], vxs[-1], vys[-1]
    
        # update position / drift (D) for position
        x_next = x + vx_half * Delta_t
        y_next = y + vy_half * Delta_t

        # update velocity / kick (K) for velocity
        r3inv    = 1 / (np.sqrt(x_next**2 + y_next**2)**3)
        ax       = - x_next * r3inv
        ay       = - y_next * r3inv
        vx_half += ax * Delta_t
        vy_half += ay * Delta_t

        # append the updated values to the lists
        ts.append(t + Delta_t)
        xs.append(x_next)
        ys.append(y_next)
        vxs.append(vx_half - 0.5 * ax * Delta_t) # record velocities at integer steps
        vys.append(vy_half - 0.5 * ay * Delta_t)

    return ts, xs, ys, vxs, vys

#################
# MAIN FUNCTION #
#################

def main():
    # problem 3.1.2

    # numerical and analytical solution
    t_f = 4 * np.pi # 2 orbital periods
    ts_numeric, xs_numeric, ys_numeric, vs_x_numeric, vs_y_numeric = forwardEuler(Delta_t = 0.01, t_f = t_f)
    ts_analy = np.linspace(0, 2 * np.pi, 1000)
    xs_analy = np.cos(ts_analy)
    ys_analy = np.sin(ts_analy)

    # plot and save
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi = 4 * DPI_MIN)
    ax.plot(xs_numeric, ys_numeric, linestyle='--', label=r'forward Euler ($\Delta t$ = 0.01)')
    ax.plot(xs_analy, ys_analy, label='analytical')
    set_fig(ax = ax, equal = True, title = r'Particle trajectory over 2 orbital periods',
            xlabel = r'$x$', ylabel = r'$y$')
    save_fig(fig = fig, name = f'prob3.1.1')

    Delta_ts = [5e-4, 1e-3, 2e-3, 4e-3]
    errs_f_pos, errs_f_E, errs_f_L = calc_errors(Delta_ts = Delta_ts, t_f = t_f, method = forwardEuler)

    # plot and save
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi = 4 * DPI_MIN)
    ax.plot(Delta_ts, errs_f_pos, 'o-', label = r'final position error $\delta_\mathrm{f} (r)$')
    ax.plot(Delta_ts, errs_f_E, '^-', label = r'final total energy error $\delta_\mathrm{f} (E)$')
    ax.plot(Delta_ts, errs_f_L, 's-', label = r'final angular momentum error $\delta_\mathrm{f} (L)$')
    set_fig(ax = ax, title = 'Final errors vs. timesteps',
            xlabel = r'timestep $\Delta t$', ylabel = r'final error $\delta_\mathrm{f}$',
            xlim = [4e-4, 5e-3], xlog = True, ylog = True)
    save_fig(fig = fig, name = f'prob3.1.2')

    # problem 3.1.3

    # numerical solution
    ts_numeric, xs_numeric, ys_numeric, vs_x_numeric, vs_y_numeric = RK4(Delta_t = 0.5, t_f = t_f)

    # plot and save
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi = 4 * DPI_MIN)
    ax.plot(xs_numeric, ys_numeric, linestyle='--', label=r'RK4 ($\Delta t$ = 0.5)')
    ax.plot(xs_analy, ys_analy, label='analytical')
    set_fig(ax = ax, equal = True,
            title = r'Particle trajectory over 2 orbital periods',
            xlabel = r'$x$', ylabel = r'$y$')
    save_fig(fig = fig, name = f'prob3.1.3')

    #
    Delta_ts = [0.05, 0.1, 0.2, 0.4]
    errs_f_pos, errs_f_E, errs_f_L = calc_errors(Delta_ts = Delta_ts, t_f = t_f, method = forwardEuler)

    # plot and save
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi = 4 * DPI_MIN)
    ax.plot(Delta_ts, errs_f_pos, 'o-', label = r'final position error $\delta_\mathrm{f} (r)$')
    ax.plot(Delta_ts, errs_f_E, '^-', label = r'final total energy error $\delta_\mathrm{f} (E)$')
    ax.plot(Delta_ts, errs_f_L, 's-', label = r'final angular momentum error $\delta_\mathrm{f} (L)$')
    set_fig(ax = ax, title = 'Final errors vs. timesteps',
            xlabel = r'timestep $\Delta t$', ylabel = r'final error $\delta_\mathrm{f}$',
            xlim = [0.04, 0.5], xlog = True, ylog = True)
    save_fig(fig = fig, name = f'prob3.1.4')

    # numerical solution
    ts_numeric, xs_numeric, ys_numeric, vs_x_numeric, vs_y_numeric = RK4(Delta_t = 1.0, t_f = t_f)

    # plot and save
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi = 4 * DPI_MIN)
    ax.plot(xs_numeric, ys_numeric, linestyle='--', label=r'RK4 ($\Delta t$ = 1.0)')
    ax.plot(xs_analy, ys_analy, label='analytical')
    set_fig(ax = ax, equal = True,
            title = r'Particle trajectory over 2 orbital periods',
            xlabel = r'$x$', ylabel = r'$y$')
    save_fig(fig = fig, name = f'prob3.1.5')

    # problem 3.1.4

    # numerical solution
    t_f = 10 * np.pi # 10 orbital periods
    ts_numeric, xs_numeric, ys_numeric, vxs_numeric, vys_numeric = leapfrog(Delta_t = 0.3, t_f = t_f)

    # plot and save
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi = 4 * DPI_MIN)
    ax.plot(xs_numeric, ys_numeric, linestyle='--', label=r'leapfrog ($\Delta t$ = 0.3)')
    ax.plot(xs_analy, ys_analy, label='analytical')
    set_fig(ax = ax, equal = True,
            title = r'Particle trajectory over 10 orbital periods',
            xlabel = r'$x$', ylabel = r'$y$')
    save_fig(fig = fig, name = f'prob3.1.6')

    # errors
    Es_numeric, Ls_numeric = calcEs_Ls(xs = xs_numeric, ys = ys_numeric, vxs = vxs_numeric, vys = vys_numeric)

    # plot and save
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi = 2 * DPI_MIN)
    ax.plot(ts_numeric, np.sqrt((xs_numeric - np.cos(ts_numeric))**2 + (ys_numeric - np.sin(ts_numeric))**2), 'o-', label = r'position error $\delta (r)$')
    ax.plot(ts_numeric, np.abs(np.array(Es_numeric) - Es_numeric[0]), '^-', label = r'total energy error $\delta (E)$')
    ax.plot(ts_numeric, np.abs(np.array(Ls_numeric) - Ls_numeric[0]), 's-', label = r'angular momentum error $\delta (L)$')
    set_fig(ax = ax, title = 'Different errors over time',
            xlabel = r'time $t$', ylabel = r'error $\delta$',
            ylog = True)
    save_fig(fig = fig, name = f'prob3.1.7')

if __name__ == "__main__":
    main()