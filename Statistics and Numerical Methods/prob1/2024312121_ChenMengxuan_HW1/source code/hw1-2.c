#include <stdio.h>

// function to find the smallest positive single
float min_single() {
    float f_min = 1.0f;
    while (f_min /2.0f != 0.0f) {
        f_min /= 2.0f;
    }
    return f_min;
}

// function to find the smallest positive double
double min_double() {
    double d_min = 1.0;
    while (d_min /2.0 != 0.0) {
        d_min /= 2.0;
    }
    return d_min;
}

int main() {
    float f_min = min_single();
    double d_min = min_double();

    printf("Smallest Positive Number for Single Precision(float): %.10e\n", f_min);
    printf("Smallest Positive Number for Double Precision(double): %.10e\n", d_min);

    return 0;
}