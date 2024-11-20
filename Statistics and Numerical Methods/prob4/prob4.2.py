"""
Function: Solution to Problem 4.2 in Statistics & Numerical Methods.
Usage:    python3.11 prob4.2.py
Version:  Last edited by Cha0s_MnK on 2024-11-20 (UTC+08:00).
"""

###########################################
# CONFIGURE ENVIRONMENT & SET ARGUMENT(S) #
###########################################

from config import *

# data
Ds  = np.array([0.032, 0.034, 0.214, 0.263, 0.275, 0.275, 0.45, 0.5, 0.5, 0.63, 0.8, 0.9, 0.9, 0.9, 0.9, 1.0,
1.1, 1.1, 1.4, 1.7, 2.0, 2.0, 2.0, 2.0]) # distance [Mpc]
vrs = np.array([170, 290, -130, -70, -185, -220, 200, 290, 270, 200, 300, -30, 650, 150, 500, 920, 450, 500,
500, 960, 500, 850, 800, 1000]) # radial velocity [km·s⁻¹]

NumBootstrap = 1000000 # number of bootstrap samples
NumData      = len(Ds) # number of data points

######################
# HELPER FUNCTION(S) #
######################

#################
# MAIN FUNCTION #
#################

def main():
    # problem 4.2.1

    # calculate the optimal Hubble constant analytically
    H0_hat = np.sum(Ds * vrs) / np.sum(Ds**2)
    Ds_fit = np.linspace(min(Ds), max(Ds), 1000)

    # plot and save
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi = 2 * DPI_MIN)
    ax.scatter(Ds, vrs, color='blue', label='Data points')
    ax.plot(Ds_fit, H0_hat * Ds_fit, color='red', label=r'Best fit: $v_\mathrm{r}$ = ' + f'{H0_hat:.2f}' + r' $D$')
    set_fig(ax = ax, title = 'Distance vs. Radial Velocity of Galaxies',
            xlabel = r'Distance $D$ (Mpc)', ylabel = r'Radial velocity $v_\mathrm{r}$ (km/s)')
    save_fig(fig = fig, name = f'fig4.2.1')

    # problem 4.2.2 & 4.2.3

    # initialize array to store bootstrap estimates of H_0 and r
    H0s_hat = np.empty(NumBootstrap)
    rs_hat  = np.empty(NumBootstrap)

    # calculate the sample Pearson correlation coefficient r
    Ds_dev  = Ds - np.mean(Ds)
    vrs_dev = vrs - np.mean(vrs)
    r_hat   = np.sum(Ds_dev * vrs_dev) / np.sqrt(np.sum(Ds_dev**2) * np.sum(vrs_dev**2))
    print(f"Sample Pearson correlation coefficient r = {r_hat:.5f}")

    # perform bootstrap sampling
    for i in range(NumBootstrap):
        # resample with replacement
        ids     = np.random.randint(0, NumData, NumData)
        Ds_hat  = Ds[ids]
        vrs_hat = vrs[ids]

        # calculate H_0 and r
        H0s_hat[i] = np.sum(Ds_hat * vrs_hat) / np.sum(Ds_hat ** 2)
        Ds_dev     = Ds_hat - np.mean(Ds_hat)
        vrs_dev    = vrs_hat - np.mean(vrs_hat)
        rs_hat[i]  = np.sum(Ds_dev * vrs_dev) / np.sqrt(np.sum(Ds_dev**2) * np.sum(vrs_dev**2))

    # calculate 95% and 99% confidence intervals
    lower95_H0 = np.percentile(H0s_hat, 2.5)
    lower99_H0 = np.percentile(H0s_hat, 0.5)
    upper95_H0 = np.percentile(H0s_hat, 97.5)
    upper99_H0 = np.percentile(H0s_hat, 99.5)

    lower95_r = np.percentile(rs_hat, 2.5)
    lower99_r = np.percentile(rs_hat, 0.5)
    upper95_r = np.percentile(rs_hat, 97.5)
    upper99_r = np.percentile(rs_hat, 99.5)

    # print the confidence intervals
    print(f"Confidence interval for Hubble constant H_0:")
    print(f"95%: [{lower95_H0:.2f}, {upper95_H0:.2f}] km/s/Mpc")
    print(f"99%: [{lower99_H0:.2f}, {upper99_H0:.2f}] km/s/Mpc")
    print(f"Confidence interval for Pearson correlation coefficient r:")
    print(f"95%: [{lower95_r:.5f}, {upper95_r:.5f}]")
    print(f"99%: [{lower99_r:.5f}, {upper99_r:.5f}]")

    # plot the bootstrap distribution and save
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi = 2 * DPI_MIN)
    ax.hist(H0s_hat, bins=128, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    ax.axvline(lower95_H0, color='red', linestyle='--', label='95\% confidence interval')
    ax.axvline(upper95_H0, color='red', linestyle='--')
    ax.axvline(lower99_H0, color='green', linestyle='--', label='99\% confidence interval')
    ax.axvline(upper99_H0, color='green', linestyle='--')
    set_fig(ax = ax, title = r'Bootstrap Distribution of Hubble Constant $H_0$',
            xlabel = r'Hubble constant $H_0$ (km/s/Mpc)', ylabel = r'Probability density')
    save_fig(fig = fig, name = f'fig4.2.2')

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi = 2 * DPI_MIN)
    ax.hist(rs_hat, bins=128, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    ax.axvline(lower95_r, color='red', linestyle='--', label='95\% confidence interval')
    ax.axvline(upper95_r, color='red', linestyle='--')
    ax.axvline(lower99_r, color='green', linestyle='--', label='99\% confidence interval')
    ax.axvline(upper99_r, color='green', linestyle='--')
    set_fig(ax = ax, title = r'Bootstrap Distribution of Pearson correlation coefficient $r$',
            xlabel = r'Pearson correlation coefficient $r$', ylabel = r'Probability density')
    save_fig(fig = fig, name = f'fig4.2.3')

    # problem 4.2.4
    F = np.sum((vrs - np.mean(vrs))**2) / np.sum((vrs - H0_hat * Ds)**2)
    print(f"F = {F:.5f}")

if __name__ == "__main__":
    main()