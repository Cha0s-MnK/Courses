#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// define whether to use float or double
// uncomment the following line to use single precision (float)
// #define USE_FLOAT

#ifdef USE_FLOAT
    typedef float real;
    #define REAL_FORMAT "%.20f"
    #define DATATYPE "FLOAT"
#else
    typedef double real;
    #define REAL_FORMAT "%.20lf"
    #define DATATYPE "DOUBLE"
#endif

// function to create the Hilbert matrix
real** create_hilbert(int n) {
    real** H = (real**)malloc(n * sizeof(real*));
    for (int i = 0; i < n; i++) {
        H[i] = (real*)malloc(n * sizeof(real));
        for (int j = 0; j < n; j++) {
            H[i][j] = 1.0 / (i + j + 1.0); // H(i,j) = 1/(i+j+1)
        }
    }
    return H;
}

// function for Cholesky decomposition
void cholesky_decomposition(real** A, real** L, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            real sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                L[i][j] = sqrt(A[i][i] - sum); // diagonal elements
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j]; // non-diagonal elements
            }
        }
    }
}

// function to calculate the b vector
void calculate_b(int n, real* b) {
    for (int i = 0; i < n; i++) {
        b[i] = 0.0;
        for (int j = 1; j <= n; j++) {
            b[i] += 1.0 / (i + j); 
        }
    }
}

// function to solve Ax = b using Cholesky decomposition
void solve_cholesky(real** L, real* b, real* x, int n) {
    real* y = (real*)malloc(n * sizeof(real));

    // solve Ly = b
    for (int i = 0; i < n; i++) {
        real sum = 0.0;
        for (int j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }

    // solve L^Tx = y
    for (int i = n - 1; i >= 0; i--) {
        real sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += L[j][i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i][i];
    }

    free(y);
}

// function to solve the Hilbert matrix equation Ax = b
void solve_hilbert(int n, real* x) {
    real** H = create_hilbert(n);
    real* b = (real*)malloc(n * sizeof(real));
    calculate_b(n, b);

    real** L = (real**)malloc(n * sizeof(real*));
    for (int i = 0; i < n; i++) {
        L[i] = (real*)malloc(n * sizeof(real));
    }

    cholesky_decomposition(H, L, n);
    solve_cholesky(L, b, x, n);

    // free allocated memory
    free(b);
    for (int i = 0; i < n; i++) {
        free(L[i]);
    }
    free(L);
    for (int i = 0; i < n; i++) {
        free(H[i]);
    }
    free(H);
}

// function to calculate the infinity norm
real infinity_norm(real** A, int n) {
    real max_row_sum = 0.0;
    for (int i = 0; i < n; i++) {
        real row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            row_sum += fabs(A[i][j]);
        }
        if (row_sum > max_row_sum) {
            max_row_sum = row_sum; // update maximum row sum
        }
    }
    return max_row_sum; // return the infinity norm
}

// function to compute the inverse of A using Cholesky decomposition
void cholesky_inverse(real** L, real** A_inv, int n) {
    // solve L * Y = I for Y
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                A_inv[i][j] = 1.0 / L[i][i]; // diagonal
            } 
            else if (j < i) {
                real sum = 0.0;
                for (int k = j; k < i; k++) {
                    sum += L[i][k] * A_inv[k][j];
                }
                A_inv[i][j] = -sum / L[i][i]; // lower triangle
            }
        }
    }
    
    // solve L^T * A_inv = Y for A_inv
    for (int i = n - 1; i >= 0; i--) {
        for (int j = 0; j < n; j++) {
            real sum = 0.0;
            for (int k = i + 1; k < n; k++) {
                sum += L[k][i] * A_inv[k][j];
            }
            A_inv[i][j] = (A_inv[i][j] - sum) / L[i][i]; // upper triangle
        }
    }
}

// function to calculate the condition number using infinity norm
real condition_number(real** A, int n) {
    real norm_A = infinity_norm(A, n); // compute the infinity norm of A
    
    real** A_inv = (real**)malloc(n * sizeof(real*));
    for (int i = 0; i < n; i++) {
        A_inv[i] = (real*)malloc(n * sizeof(real));
    }
    
    // compute the Cholesky decomposition
    real** L = (real**)malloc(n * sizeof(real*));
    for (int i = 0; i < n; i++) {
        L[i] = (real*)malloc(n * sizeof(real));
    }
    cholesky_decomposition(A, L, n); // perform Cholesky decomposition

    cholesky_inverse(L, A_inv, n); // compute the inverse of A

    real norm_A_inv = infinity_norm(A_inv, n); // compute the infinity norm of the inverse of A
    
    real cond_number = norm_A * norm_A_inv; // compute the condition number
    
    // free allocated memory
    for (int i = 0; i < n; i++) {
        free(A_inv[i]);
        free(L[i]);
    }
    free(A_inv);
    free(L);

    return cond_number; // return the condition number
}

// function to compute the condition number for specific n values
void compute_condition_numbers() {
    int ns[] = {3, 6, 9, 12};
    for (int i = 0; i < 4; i++) {
        int n = ns[i];
        real** H = create_hilbert(n);
        real cond_num = condition_number(H, n);
        printf("Condition number for n = %d: %.2e\n", n, cond_num);
        
        // free the Hilbert matrix
        for (int j = 0; j < n; j++) {
            free(H[j]);
        }
        free(H);
    }
}

// function to calculate the mean relative error
double calculate_mean_relative_error(real* x, int n) {
    double mean_relative_error = 0.0;
    for (int i = 0; i < n; i++) {
        mean_relative_error += fabs(x[i] - 1.0); // compare with expected solution x = (1, 1, ..., 1)
    }
    return (mean_relative_error/n);
}

// function to calculate the vector norm (2-norm)
real calculate_vector_norm(real* x, int n) {
    real vector_norm = 0.0;
    for (int i = 0; i < n; i++) {
        vector_norm += pow(x[i],2); 
    }
    vector_norm = sqrt(vector_norm);
    return vector_norm;
}

// function to find instability n (use 2-norm)
void find_instability_n() {
    int n = 1; 
    real last_error = 0.0;

    while (1) {
        n++; // start from n = 2
        real* x = (real*)malloc(n * sizeof(real));
        
        // solve the Hilbert matrix equation
        solve_hilbert(n, x);

        // calculate the relative error
        real vector_norm = calculate_vector_norm(x, n);
        real relative_error = (vector_norm - sqrt(n)) / sqrt(n); // relative error (use 2-norm)

        // check for instability
        if (relative_error > 0.5) {
            printf("Instability(use 2-norm) found at n = %d with relative error: %.2f%%\n", n, relative_error * 100);
            // printf("Last stable n = %d with relative error: %.2f%%\n", n - 1, last_error * 100);
            free(x);
            break;
        }

        last_error = relative_error; // store the last error for comparison

        free(x); 
    }
}

// main function
int main() {
    int n = 5; // set n for the second problem
    real* x = (real*)malloc(n * sizeof(real));
    
    // solve the Hilbert matrix equation for n = 5
    solve_hilbert(n, x);
    
    // print results
    printf("In "DATATYPE":\n");
    printf("Solution x for n = %d:\n", n);
    for (int i = 0; i < n; i++) {
        printf("x[%d] = " REAL_FORMAT "\n", i, x[i]);
    }
    
    // calculate the mean relative error
    real mean_relative_error = calculate_mean_relative_error(x, n);
    printf("Mean relative error compared to expected solution: " REAL_FORMAT "%%\n", mean_relative_error * 100);
    
    free(x);

    // find the instability n (solve the third problem)
    find_instability_n();
    
    // compute condition numbers for n = 3, 6, 9, 12 (solve fourth problem)
    compute_condition_numbers();

    return 0;
}
