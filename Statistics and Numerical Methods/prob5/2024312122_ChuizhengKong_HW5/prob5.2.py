"""
Function: Solution to Problem 5.2 in Statistics & Numerical Methods.
Usage:    python3.11 prob5.2.py
Version:  Last edited by Cha0s_MnK on 2024-11-27 (UTC+08:00).
"""

#########################################
# CONFIGURE ENVIRONMENT & SET ARGUMENTS #
#########################################

from config import *

Delta_x = FLOAT(0.02)
N       = INT(10**6)

######################
# HELPER FUNCTION(S) #
######################

def p(x):
    return np.exp(-(2 * x + 3 * np.cos(x)**2)**2)

#################
# MAIN FUNCTION #
#################

def main():
    # initialize the chain
    chain = np.zeros(N)

    # start with a random guess x0
    chain[0] = np.random.uniform(-10, 10)

    # Metropolis-Hastings algorithm
    for i in range(1, N):
        # propose a new point by adding a random number from Uniform[-1, 1]
        x_new = chain[i - 1] + np.random.uniform(-1, 1)

        # calculate the acceptance probability
        r = min(1, p(x_new) / p(chain[i - 1]))

        # accept or reject the new point
        if np.random.uniform(0, 1) < r:
            chain[i] = x_new
        else: # keep the current point
            chain[i] = chain[i - 1]

    # create a histogram of the samples with bin size Δx
    bins = np.arange(min(chain), max(chain) + Delta_x, Delta_x)
    hist_density, bin_edges = np.histogram(chain, bins=bins, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2 # bin centres for plotting

    # plot the histogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi = 2 * DPI_MIN)
    ax.bar(bin_centres, hist_density, width=Delta_x, alpha=0.7, label='Sampled distribution')

    # overplot the given distribution
    xs      = np.linspace(min(chain), max(chain), 9999)
    ps      = p(xs)
    ps_norm = ps / np.trapz(ps, xs) # normalize the given distribution
    ax.plot(xs, ps_norm, 'r-', label='Analytic distribution')

    set_fig(ax = ax, title = r'Metropolis-Hastings Sampling of the Given Distribution',
            xlabel = r'$x$', ylabel = r'Given distribution $p(x)$', xlim=[-2.9, 1.4])
    save_fig(fig = fig, name = f'fig5.2')

    print(f"Number of unique points in the chain: {len(np.unique(chain))}")

if __name__ == "__main__":
    main()