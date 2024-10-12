#include <stdio.h>
#include <math.h>

// constants
#define H0 67.0          // Hubble constant in km/s/Mpc
#define OMEGA_M 0.3      // matter density
#define OMEGA_LAMBDA 0.7 // dark energy density
#define C 299792.458     // speed of light in km/s

// function to calculate H0 / H(z)
double H0overHz(double z) {
    return 1/(sqrt(OMEGA_M * pow(1 + z, 3) + OMEGA_LAMBDA));
}

// composite Simpson's rule integration
double simpsons_rule(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double integral = func(a) + func(b);
    
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        integral += (i % 2 == 0) ? 2 * func(x) : 4 * func(x);
    }
    
    integral *= h / 3.0;

    return integral;
}

// adaptive Simpson's rule with step increment of 2
double adaptive_simpson(double (*func)(double), double a, double b, double epsilon, double old_integral, int *steps) {
    int n = 4; 
    *steps = n; // store the number of steps
    double integral = simpsons_rule(func, a, b, n);
    
    // loop until desired precision is achieved
    while (fabs((integral - old_integral)/integral) > epsilon) {
        n += 2;  // increment by 2
        *steps = n; // store the number of steps
        old_integral = integral; // update full integral to the last estimate
        integral = simpsons_rule(func, a, b, n); 
    }
    
    return integral;
}

int main() {
    double z_values[] = {1.0, 3.0, 8.2};
    int num_z = sizeof(z_values) / sizeof(z_values[0]);
    
    for (int i = 0; i < num_z; i++) {
        double z = z_values[i];
        double comoving_distance;
        double epsilon = 1e-4; // relative precision
        int steps = 2; // initial step count
        
        // perform adaptive integration
        double old_integral = simpsons_rule(H0overHz, 0, z, 2); // start with 2 intervals
        comoving_distance = (C / H0) * adaptive_simpson(H0overHz, 0, z, epsilon, old_integral, &steps);
        
        printf("Comoving distance for z = %.1f: %.6f Mpc, steps(intervals) needed: %d\n", z, comoving_distance, steps);
    }
    
    return 0;
}
