"""
Function: Solution to Problem 6.1 in Statistics & Numerical Methods.
Usage:    python3.11 prob6.1.py
Version:  Last edited by Cha0s_MnK on 2024-12-17 (UTC+08:00).
"""

#########################################
# CONFIGURE ENVIRONMENT & SET ARGUMENTS #
#########################################

from config import *

degrees  = range(INT(1), INT(10))
N        = INT(30)
sigma1   = FLOAT(0.1)
sigma2   = FLOAT(1.0)
xs       = np.random.uniform(0, 2, N)
xs_analy = np.linspace(INT(0), INT(2), INT(999))

######################
# HELPER FUNCTION(S) #
######################

def f(x):
    """polynomial function"""
    return 7 + 2 * (x - 0.2) - 3 * (x - 0.5)**2 - 6 * (x - 0.8)**3

import numpy as np

def calcMSE(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameter(s):
    - y_true (array-like): True target values. Shape (n_samples,) or (n_samples, n_outputs).
    - y_pred (array-like): Predicted target values. Shape (n_samples,) or (n_samples, n_outputs).

    Return(s):
    - MSE (float or ndarray): The MSE loss. If `y_true` is 1-dimensional, returns a single float.
        If `y_true` is multi-dimensional, returns an array of MSE for each output.
    """
    # Convert inputs to NumPy arrays for efficient computation
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check if shapes of y_true and y_pred match
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true shape {y_true.shape} and y_pred shape {y_pred.shape} must be the same.")
    
    # Compute the squared differences
    squared_diff = (y_true - y_pred) ** 2
    
    # Compute the mean of the squared differences
    mse = np.mean(squared_diff)

    return mse

def calcAIC(MSE, k):
    return N * np.log(MSE) + 2 * k

def calcBIC(MSE, k):
    return N * np.log(MSE) + k * np.log(N)

def print_model_params(xs, ys, deg, method):
    """fit and print polynomial regression coefficients"""
    print(f"\nFitting polynomial of degree {INT(deg)} to the data...")
    xs_polyn   = PolynomialFeatures(INT(deg)).fit_transform(xs.reshape(-1, 1))
    model_best = LinearRegression()
    model_best.fit(xs_polyn, ys)
    print(f"\nBest model parameters according to {method}:")
    print(f"Intercept: {model_best.intercept_}")
    print(f"Coefficients: {model_best.coef_}")

#################
# MAIN FUNCTION #
#################

def main():
    # problem 6.1.1
    ys_hat   = f(xs)
    ys_analy = f(xs_analy)

    # plot the polynomial and sampled points
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi = 2 * DPI_MIN)
    ax.scatter(xs, ys_hat, color='red', label='MC sampled points')
    ax.plot(xs_analy, ys_analy, color='blue', label='polynomial function')
    set_fig(ax=ax, title='MC Sampling of the Polynomial Function', xlabel=r'$x$', ylabel=r'$y$')
    save_fig(fig=fig, name='fig6.1.1')

    # problem 6.1.2
    ys_noisy = ys_hat + np.random.normal(0, sigma1, N)

    # plot the polynomial, expected y, and noisy y
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi = 2 * DPI_MIN)
    ax.plot(xs_analy, ys_analy, label='polynomial function', color='blue')
    ax.scatter(xs, ys_hat, color='red', label='expected MC sampled points')
    ax.scatter(xs, ys_noisy, color='green', marker='x', label='noisy MC sampled points')
    set_fig(ax=ax, title='MC Sampling with Gaussian Noise of the Polynomial Function', xlabel=r'$x$',
            ylabel=r'$y$')
    save_fig(fig=fig, name='fig6.1.2')

    # problem 6.1.3
    AICs = []
    BICs = []
    MSEs = []

    for deg in degrees:
        xs_polyn = PolynomialFeatures(deg).fit_transform(xs.reshape(-1, 1))
        model    = LinearRegression()
        model.fit(xs_polyn, ys_noisy)
        MSE      = calcMSE(ys_noisy, model.predict(xs_polyn))
        AICs.append(calcAIC(N, MSE, xs_polyn.shape[1]))
        BICs.append(calcBIC(N, MSE, xs_polyn.shape[1]))

        # cross-validation
        k_fold   = KFold(n_splits=5, shuffle=True)
        MSEs_deg = []
        for id_train, id_test in k_fold.split(xs_polyn):
            xs_train, xs_test = xs_polyn[id_train], xs_polyn[id_test]
            ys_train, ys_test = ys_noisy[id_train], ys_noisy[id_test]
            model = LinearRegression()
            model.fit(xs_train, ys_train)
            MSEs_deg.append(calcMSE(ys_test, model.predict(xs_test)))
        MSEs.append(np.mean(MSEs_deg))

    results = pd.DataFrame({'Degree': degrees,
                            'AIC': AICs,
                            'BIC': BICs,
                            'MSE': MSEs})
    degAIC = results.loc[results['AIC'].idxmin()]['Degree']
    degBIC = results.loc[results['BIC'].idxmin()]['Degree']
    degMSE = results.loc[results['MSE'].idxmin()]['Degree']
    print(f"Best model according to AIC: {degAIC}")
    print(f"Best model according to BIC: {degBIC}")
    print(f"Best model according to cross-validation MSE : {degMSE}")
    # refit the best model to print the parameters
    print_model_params(xs=xs, ys=ys_noisy, deg=degAIC, method="AIC")
    print_model_params(xs=xs, ys=ys_noisy, deg=degBIC, method="BIC")
    print_model_params(xs=xs, ys=ys_noisy, deg=degMSE, method="MSE")

    # plot
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6), dpi=2 * DPI_MIN)

    line1, = ax1.plot(results['Degree'], results['AIC'], marker='o', color='red', label='AIC')
    line2, = ax1.plot(results['Degree'], results['BIC'], marker='s', color='orange', label='BIC')
    set_fig(ax=ax1, title=r'AIC, BIC and cross-validation MSE vs Polynomial Degree ($\sigma = 0.1$)',
            xlabel=r'Polynomial degree $n$', ylabel='Information criterion', legend=False)
    ax2 = ax1.twinx()
    line3, = ax2.plot(results['Degree'], results['MSE'], marker='^', color='green', label='MSE')
    set_fig(ax=ax2, ylabel='MSE', legend=False)

    # combine legends from ax1 and ax2
    lines  = [line1, line2, line3]                # collect all line objects
    labels = [line.get_label() for line in lines] # collect all labels
    ax1.legend(lines, labels, loc='upper center')
    save_fig(fig=fig, name='fig6.1.3')

    # problem 6.1.4
    ys_noisy = ys_hat + np.random.normal(0, sigma2, N)
    AICs = []
    BICs = []
    MSEs = []

    for deg in degrees:
        xs_polyn = PolynomialFeatures(deg).fit_transform(xs.reshape(-1, 1))
        model    = LinearRegression()
        model.fit(xs_polyn, ys_noisy)
        MSE      = calcMSE(ys_noisy, model.predict(xs_polyn))
        AICs.append(calcAIC(N, MSE, xs_polyn.shape[1]))
        BICs.append(calcBIC(N, MSE, xs_polyn.shape[1]))

        # cross-validation
        k_fold   = KFold(n_splits=5, shuffle=True)
        MSEs_deg = []
        for id_train, id_test in k_fold.split(xs_polyn):
            xs_train, xs_test = xs_polyn[id_train], xs_polyn[id_test]
            ys_train, ys_test = ys_noisy[id_train], ys_noisy[id_test]
            model = LinearRegression()
            model.fit(xs_train, ys_train)
            MSEs_deg.append(calcMSE(ys_test, model.predict(xs_test)))
        MSEs.append(np.mean(MSEs_deg))

    results = pd.DataFrame({'Degree': degrees,
                            'AIC': AICs,
                            'BIC': BICs,
                            'MSE': MSEs})
    degAIC = results.loc[results['AIC'].idxmin()]['Degree']
    degBIC = results.loc[results['BIC'].idxmin()]['Degree']
    degMSE = results.loc[results['MSE'].idxmin()]['Degree']
    print(f"Best model according to AIC: {degAIC}")
    print(f"Best model according to BIC: {degBIC}")
    print(f"Best model according to cross-validation MSE : {degMSE}")
    # refit the best model to print the parameters
    print_model_params(xs=xs, ys=ys_noisy, deg=degAIC, method="AIC")
    print_model_params(xs=xs, ys=ys_noisy, deg=degBIC, method="BIC")
    print_model_params(xs=xs, ys=ys_noisy, deg=degMSE, method="MSE")

    # plot
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6), dpi=2 * DPI_MIN)

    line1, = ax1.plot(results['Degree'], results['AIC'], marker='o', color='red', label='AIC')
    line2, = ax1.plot(results['Degree'], results['BIC'], marker='s', color='orange', label='BIC')
    set_fig(ax=ax1, title=r'AIC, BIC and cross-validation MSE vs Polynomial Degree ($\sigma = 1.0$)',
            xlabel=r'Polynomial degree $n$', ylabel='Information criterion', legend=False)
    ax2 = ax1.twinx()
    line3, = ax2.plot(results['Degree'], results['MSE'], marker='^', color='green', label='MSE')
    set_fig(ax=ax2, ylabel='MSE', legend=False)

    # combine legends from ax1 and ax2
    lines  = [line1, line2, line3]                # collect all line objects
    labels = [line.get_label() for line in lines] # collect all labels
    ax1.legend(lines, labels, loc='upper center')
    save_fig(fig=fig, name='fig6.1.4')

if __name__ == "__main__":
    main()

"""
# Section 6.5: Local Regression with Gaussian Kernel
def gaussian_kernel_regression(x_train, y_train, x_test, h):
    weights = np.exp(-((x_test - x_train)**2) / (2 * h**2))
    if np.sum(weights) == 0:
        return 0
    return np.sum(weights * y_train) / np.sum(weights)

# Define a range of bandwidths to test
h_values = np.linspace(0.05, 1.0, 20)
mse_h = []

for h in h_values:
    predictions = []
    loo = LeaveOneOut()
    for id_train, id_test in loo.split(X):
        X_train, X_test = X[id_train], X[id_test]
        y_train, y_test = y_new[id_train], y_new[id_test]
        y_pred = gaussian_kernel_regression(X_train.flatten(), y_train, X_test.flatten()[0], h)
        predictions.append(y_pred)
    mse = mean_squared_error(y_new, predictions)
    mse_h.append(mse)

# Find the h with minimum MSE
best_h_index = np.argmin(mse_h)
best_h = h_values[best_h_index]
print(f"Optimal bandwidth h: {best_h}")

# Plotting MSE vs h
plt.figure(figsize=(8, 5))
plt.plot(h_values, mse_h, marker='o')
plt.axvline(best_h, color='red', linestyle='--', label=f'Optimal h = {best_h:.2f}')
plt.xlabel('Bandwidth h')
plt.ylabel('LOOCV MSE')
plt.title('Bandwidth Selection via LOOCV')
plt.legend()
plt.grid(True)
plt.savefig('bandwidth_selection.png')
plt.show()

# Define function to perform local regression with given h
def local_regression(x_train, y_train, x_plot, h):
    y_pred = []
    for x_j in x_plot:
        y_j = gaussian_kernel_regression(x_train, y_train, x_j, h)
        y_pred.append(y_j)
    return np.array(y_pred)

# Compute predictions with best h, h/3, and 3h
h_best = best_h
h_low = h_best / 3
h_high = h_best * 3

y_pred_best = local_regression(x_samples, y_noisy, x_plot, h_best)
y_pred_low = local_regression(x_samples, y_noisy, x_plot, h_low)
y_pred_high = local_regression(x_samples, y_noisy, x_plot, h_high)

# Plot all models
plt.figure(figsize=(12, 8))

# (a) Original Polynomial Function
plt.plot(x_plot, y_plot, label='Original Polynomial', color='blue', linewidth=2)

# (b) Sampled Data Points
plt.scatter(x_sorted, y_noisy, color='green', label='Noisy Data Points', marker='x')

# (c) Best Regression Model (Degree from Section 6.3)
poly_best_degree = int(best_cv['Degree'])
poly_best = PolynomialFeatures(poly_best_degree)
X_poly_best = poly_best.fit_transform(X)
model_best = LinearRegression()
model_best.fit(X_poly_best, y_new)
y_best_pred = model_best.predict(poly_best.fit_transform(x_plot.reshape(-1, 1)))
plt.plot(x_plot, y_best_pred, label=f'Best Polynomial Degree {poly_best_degree}', color='red', linewidth=2)

# (d) Local Regression with Gaussian Kernel
plt.plot(x_plot, y_pred_best, label=f'Local Regression (h={h_best:.2f})', color='purple', linewidth=2)
plt.plot(x_plot, y_pred_low, label=f'Local Regression (h={h_low:.2f})', color='orange', linestyle='--')
plt.plot(x_plot, y_pred_high, label=f'Local Regression (h={h_high:.2f})', color='cyan', linestyle='--')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Regression Models Comparison')
plt.legend()
plt.grid(True)
plt.savefig('regression_models.png')
plt.show()
"""