
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>

void kernel_c(const float* A, const float* X, float* Y, size_t m, size_t n)
{
    for (size_t i = 0; i < m; i++) {
        float sum = 0.0f;

        for (size_t j = 0; j < n; j++) {
            sum += A[i * n + j] * X[j];
        }

        Y[i] = sum;
    }
}

void kernel_reference(const float* A, const float* X, double* Y, size_t m, size_t n)
{
    for (size_t i = 0; i < m; i++) {
        double sum = 0.0;

        for (size_t j = 0; j < n; j++) {
            sum += (double)A[i * n + j] * (double)X[j];
        }

        Y[i] = sum;
    }
}

void kernel_scalar(const float* A, const float* X, float* Y, size_t m, size_t n);

void kernel_xmm(const float* A, const float* X, float* Y, size_t m, size_t n);

void kernel_ymm(const float* A, const float* X, float* Y, size_t m, size_t n);

int correctness_check(const float* Y1, const float* Y2, size_t m)
{
	const float tolerance = 0.005f; // 0.5% tolerance
    for (size_t i = 0; i < m; i++) {
        float difference = fabsf(Y1[i] - Y2[i]);

        float allowed_error =
            tolerance * fmaxf(1.0f, fabsf(Y1[i]));

        if (difference > allowed_error) {
            printf(
                "Mismatch at Y[%zu]: C=%.7f, test=%.7f, "
                "difference=%.7f, tolerance=%.7f\n",
                i, Y1[i], Y2[i],
                difference, allowed_error
            );
            return 0;
        }
    }
	return 1; 
}

int reference_check(const double* Y_ref,
    const float* Y_test, size_t m)
{
    const double tolerance = 0.005; // 0.5%

    for (size_t i = 0; i < m; i++) {
        double difference = fabs(Y_ref[i] - (double)Y_test[i]);
        double allowed_error =
            tolerance * fmax(1.0, fabs(Y_ref[i]));

        if (difference > allowed_error) {
            printf(
                "Mismatch at Y[%zu]: reference=%.7f, "
                "test=%.7f, difference=%.7f, tolerance=%.7f\n",
                i, Y_ref[i], Y_test[i],
                difference, allowed_error
            );
            return 0;
        }
    }

    return 1;
}

typedef void (*kernel_func)(const float* A, const float* X, float* Y, size_t m, size_t n);
// Benchmarking function returns the average time in milliseconds for the kernel function to execute over a number of repetitions
double benchmark_kernel(kernel_func kernel, const float* A, const float* X, float* Y, size_t m, size_t n, int reps)
{
	LARGE_INTEGER start, end, freq;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
	for (int i = 0; i < reps; i++) {
		kernel(A, X, Y, m, n);
	}
	QueryPerformanceCounter(&end);
	double elapsed_time = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart;
    return (elapsed_time * 1000.0) / reps;
}


int main(void)
{
    const size_t m = 10;
    const size_t n = 1ULL << 25;
    const int reps = 30;
    
    // Allocate arrays dynamically
    float* A = malloc(m * n * sizeof(float));
    float* X = malloc(n * sizeof(float));
    float* Y_c = malloc(m * sizeof(float));
    float* Y_scalar = malloc(m * sizeof(float));
    float* Y_xmm = malloc(m * sizeof(float));
    float* Y_ymm = malloc(m * sizeof(float));
	double* Y_ref = malloc(m * sizeof(double));

    if (!A || !X || !Y_c || !Y_scalar ||
        !Y_xmm || !Y_ymm || !Y_ref) {
        printf("Memory allocation failed.\n");
        free(A);
        free(X);
        free(Y_c);
        free(Y_scalar);
        free(Y_xmm);
        free(Y_ymm);
		free(Y_ref);
        return 1;
    }

	// Intialize matrix and vector
	for (size_t i = 0; i < m * n; i++) {
		A[i] = (float)((i % 17) + 1) * 0.1f;
    }

	for (size_t i = 0; i < n; i++) {
		X[i] = (float)((i % 13) + 1) * 0.1f;
	}

	// Run each kernel once to populate the output arrays
    kernel_c(A, X, Y_c, m, n);
    kernel_scalar(A, X, Y_scalar, m, n);
    kernel_xmm(A, X, Y_xmm, m, n);
    kernel_ymm(A, X, Y_ymm, m, n);
	kernel_reference(A, X, Y_ref, m, n);
   
    // Benchmarking
    double time_c = benchmark_kernel(
        kernel_c, A, X, Y_c, m, n, reps);

    double time_scalar = benchmark_kernel(
        kernel_scalar, A, X, Y_scalar, m, n, reps);

    double time_xmm = benchmark_kernel(
        kernel_xmm, A, X, Y_xmm, m, n, reps);

    double time_ymm = benchmark_kernel(
        kernel_ymm, A, X, Y_ymm, m, n, reps);

    // Output
    printf("\n========== Matrix-Vector Product ==========\n");
    printf("Matrix dimensions: %zu x %zu\n", m, n);
    printf("Repetitions: %d\n", reps);

    size_t count = (m < 5) ? m : 5;

    printf("\n--- C ---\n");
    printf("Correctness: %s\n",
        reference_check(Y_ref, Y_c, m) ? "PASS" : "FAIL");
    printf("Average execution time: %.6f ms\n", time_c);

    printf("First %zu Y values: ", count);
    for (size_t i = 0; i < count; i++)
        printf("%.4f ", Y_c[i]);

    printf("\nLast %zu Y values:  ", count);
    for (size_t i = m - count; i < m; i++)
        printf("%.4f ", Y_c[i]);


    printf("\n\n--- Scalar x86-64 ---\n");
    printf("Correctness: %s\n",
        reference_check(Y_ref, Y_scalar, m) ? "PASS" : "FAIL");
    printf("Average execution time: %.6f ms\n", time_scalar);

    printf("First %zu Y values: ", count);
    for (size_t i = 0; i < count; i++)
        printf("%.4f ", Y_scalar[i]);

    printf("\nLast %zu Y values:  ", count);
    for (size_t i = m - count; i < m; i++)
        printf("%.4f ", Y_scalar[i]);


    printf("\n\n--- SIMD XMM ---\n");
    printf("Correctness: %s\n",
        reference_check(Y_ref, Y_xmm, m) ? "PASS" : "FAIL");
    printf("Average execution time: %.6f ms\n", time_xmm);

    printf("First %zu Y values: ", count);
    for (size_t i = 0; i < count; i++)
        printf("%.4f ", Y_xmm[i]);

    printf("\nLast %zu Y values:  ", count);
    for (size_t i = m - count; i < m; i++)
        printf("%.4f ", Y_xmm[i]);


    printf("\n\n--- SIMD YMM ---\n");
    printf("Correctness: %s\n",
        reference_check(Y_ref, Y_ymm, m) ? "PASS" : "FAIL");
    printf("Average execution time: %.6f ms\n", time_ymm);

    printf("First %zu Y values: ", count);
    for (size_t i = 0; i < count; i++)
        printf("%.4f ", Y_ymm[i]);

    printf("\nLast %zu Y values:  ", count);
    for (size_t i = m - count; i < m; i++)
        printf("%.4f ", Y_ymm[i]);

    printf("\n");


    // Release allocated memory
	free(A);
	free(X);
    free(Y_c);
    free(Y_scalar);
    free(Y_xmm);
    free(Y_ymm);
    free(Y_ref);

    return 0;
}