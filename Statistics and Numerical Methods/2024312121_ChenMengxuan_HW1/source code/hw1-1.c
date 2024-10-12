#include <stdio.h>

// function to find the machine precision for single precision
float machine_precision_single() {
    float m = 1.0f;
    while (1.0f + m != 1.0f) {
        m /= 2.0f;
    }
    return m * 2.0f;  
}

// function to find the machine precision for double precision
double machine_precision_double() {
    double m = 1.0;
    while (1.0 + m != 1.0) {
        m /= 2.0;
    }
    return m * 2.0;  
}

int main() {
    float single_precision = machine_precision_single();
    double double_precision = machine_precision_double();

    printf("Machine Precision for Single Precision(float): %.10e\n", single_precision);
    printf("Machine Precision for Double Precision(double): %.10e\n", double_precision);

    return 0;
}
