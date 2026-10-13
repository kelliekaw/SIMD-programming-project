
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <intrin.h>
#include <windows.h>

// Every kernel sums each row in blocks of BLOCK_SIZE columns: a fresh float
// partial sum per block, then the block's partial is added to the row total.
// This keeps all arithmetic in single precision but stops the running sum
// from growing so large that small products are rounded away.
// MUST match BLOCK_SIZE in kernel_scalar.asm, kernel_xmm.asm, kernel_ymm.asm,
// and must be a multiple of 8 (the YMM vector width).
#define BLOCK_SIZE 4096

#define TOLERANCE 0.005 // 0.5% relative tolerance

void kernel_c(const float* A, const float* X, float* Y, size_t m, size_t n)
{
    for (size_t i = 0; i < m; i++) {
        const float* row = A + i * n;
        float total = 0.0f;

        for (size_t start = 0; start < n; start += BLOCK_SIZE) {
            size_t end = (n - start > BLOCK_SIZE) ? start + BLOCK_SIZE : n;
            float partial = 0.0f;

            for (size_t j = start; j < end; j++) {
                partial += row[j] * X[j];
            }

            total += partial;
        }

        Y[i] = total;
    }
}

// Independent diagnostic: plain (unblocked) double-precision accumulation.
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

typedef struct {
    double max_abs_diff;
    double max_rel_diff;
    size_t outside;     // elements outside the tolerance (NaN counts as outside)
} check_result;

// Required assignment check: compare a kernel's output against the C output.
check_result correctness_check(const float* Y_c, const float* Y_test, size_t m)
{
    check_result r = { 0.0, 0.0, 0 };
    size_t printed = 0;

    for (size_t i = 0; i < m; i++) {
        double difference = fabs((double)Y_c[i] - (double)Y_test[i]);
        double allowed_error = TOLERANCE * fmax(1.0, fabs((double)Y_c[i]));
        double relative = difference / fmax(1.0, fabs((double)Y_c[i]));

        if (difference > r.max_abs_diff || isnan(difference)) r.max_abs_diff = difference;
        if (relative > r.max_rel_diff || isnan(relative)) r.max_rel_diff = relative;

        // Written as !(<=) so that a NaN result is treated as a failure.
        if (!(difference <= allowed_error)) {
            if (printed++ < 5)
                printf("  Mismatch at Y[%zu]: C=%.4f, test=%.4f, "
                    "difference=%.4f, allowed=%.4f\n",
                    i, Y_c[i], Y_test[i], difference, allowed_error);
            r.outside++;
        }
    }

    return r;
}

// Additional numerical diagnostic: compare against the double-precision reference.
check_result reference_check(const double* Y_ref, const float* Y_test, size_t m)
{
    check_result r = { 0.0, 0.0, 0 };

    for (size_t i = 0; i < m; i++) {
        double difference = fabs(Y_ref[i] - (double)Y_test[i]);
        double allowed_error = TOLERANCE * fmax(1.0, fabs(Y_ref[i]));
        double relative = difference / fmax(1.0, fabs(Y_ref[i]));

        if (difference > r.max_abs_diff || isnan(difference)) r.max_abs_diff = difference;
        if (relative > r.max_rel_diff || isnan(relative)) r.max_rel_diff = relative;

        if (!(difference <= allowed_error))
            r.outside++;
    }

    return r;
}

typedef void (*kernel_func)(const float* A, const float* X, float* Y, size_t m, size_t n);
// Benchmarking function returns the average time in milliseconds for the kernel function to execute over a number of repetitions.
// Only the kernel calls are inside the timed region.
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

// Exact-integer self-test. With small whole-number inputs every partial sum is
// exactly representable as a float, so all four kernels must return the exact
// answer regardless of summation order. A missed, duplicated or out-of-bounds
// element, or a wrong row offset, changes the result. The arrays are surrounded
// by poison values, and two sentinels after Y[m-1] detect out-of-bounds writes.
static int exact_selftest(void)
{
    const size_t ms[] = { 1, 3, 10 };
    const size_t extra[] = { 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
        4095, 4096, 4097, 4103, 4104, 4105, 8191, 8192, 8193, 12291 };
    const size_t n_extra = sizeof(extra) / sizeof(extra[0]);
    const size_t pad = 64;
    kernel_func kernels[4] = { kernel_c, kernel_scalar, kernel_xmm, kernel_ymm };
    const char* names[4] = { "C", "Scalar x86-64", "SIMD XMM", "SIMD YMM" };
    size_t cases = 0, failures = 0;

    for (size_t a = 0; a < 3; a++) {
        for (size_t t = 0; t < 40 + n_extra; t++) {
            size_t m = ms[a];
            size_t n = (t < 40) ? t + 1 : extra[t - 40];
            float* Abuf = malloc((m * n + 2 * pad) * sizeof(float));
            float* Xbuf = malloc((n + 2 * pad) * sizeof(float));
            float* Y = malloc((m + 2) * sizeof(float));
            double* exact = malloc(m * sizeof(double));
            if (!Abuf || !Xbuf || !Y || !exact) {
                printf("Self-test allocation failed.\n");
                free(Abuf); free(Xbuf); free(Y); free(exact);
                return 0;
            }
            float* A = Abuf + pad;
            float* X = Xbuf + pad;

            for (size_t k = 0; k < m * n + 2 * pad; k++) Abuf[k] = 1000.0f;
            for (size_t k = 0; k < n + 2 * pad; k++) Xbuf[k] = 1000.0f;
            for (size_t i = 0; i < m; i++)
                for (size_t j = 0; j < n; j++)
                    A[i * n + j] = (float)((i + 1) * 10 + (j % 7));
            for (size_t j = 0; j < n; j++)
                X[j] = (float)((j % 5) + 1);

            for (size_t i = 0; i < m; i++) {
                exact[i] = 0.0;
                for (size_t j = 0; j < n; j++)
                    exact[i] += (double)A[i * n + j] * (double)X[j];
            }

            for (int k = 0; k < 4; k++) {
                for (size_t i = 0; i < m; i++) Y[i] = NAN;
                Y[m] = Y[m + 1] = -7.0f;

                kernels[k](A, X, Y, m, n);
                cases++;

                int ok = (Y[m] == -7.0f && Y[m + 1] == -7.0f);
                for (size_t i = 0; i < m; i++)
                    if ((double)Y[i] != exact[i]) ok = 0;

                if (!ok) {
                    if (failures < 10)
                        printf("  Self-test FAIL: %s, m=%zu, n=%zu\n", names[k], m, n);
                    failures++;
                }
            }

            free(Abuf); free(Xbuf); free(Y); free(exact);
        }
    }

    printf("Exact-integer self-test: %zu kernel runs (m = 1, 3, 10; n = 1..40 and "
        "boundary sizes up to 12291): %s (%zu failures)\n",
        cases, failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0;
}

// Returns 0 on overflow.
static int checked_mul(size_t a, size_t b, size_t* out)
{
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

// Accepts a decimal number ("33554432") or a power of two ("2^25").
static int parse_size(const char* s, size_t* out)
{
    char* end;
    if (s[0] == '2' && s[1] == '^') {
        unsigned long e = strtoul(s + 2, &end, 10);
        if (*end != '\0' || e > 62) return 0;
        *out = (size_t)1 << e;
        return 1;
    }
    unsigned long long v = strtoull(s, &end, 10);
    if (*end != '\0' || v == 0) return 0;
    *out = (size_t)v;
    return 1;
}

// The XMM and YMM kernels use VEX-encoded (AVX) instructions.
static int cpu_supports_avx(void)
{
    int info[4];
    __cpuid(info, 1);
    int osxsave = (info[2] >> 27) & 1;
    int avx = (info[2] >> 28) & 1;
    if (!osxsave || !avx) return 0;
    return (_xgetbv(0) & 0x6) == 0x6; // OS saves XMM and YMM state
}

static int cpu_supports_avx2(void)
{
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] >> 5) & 1;
}

static void print_kernel(const char* name, const float* Y, size_t m,
    double time_ms, double time_c, const check_result* vs_c,
    const check_result* vs_ref, int is_baseline)
{
    size_t count = (m < 5) ? m : 5;

    printf("\n--- %s ---\n", name);
    printf("Average execution time: %.6f ms", time_ms);
    printf("   Speedup vs C: %.2fx\n", time_c / time_ms);

    printf("First %zu Y values: ", count);
    for (size_t i = 0; i < count; i++)
        printf("%.4f ", Y[i]);

    printf("\nLast %zu Y values:  ", count);
    for (size_t i = m - count; i < m; i++)
        printf("%.4f ", Y[i]);
    printf("\n");

    if (is_baseline)
        printf("Check vs C (required):   BASELINE (answer key)\n");
    else
        printf("Check vs C (required):   %s  max abs diff=%.4f, max rel diff=%.6f%%, outside tolerance=%zu/%zu\n",
            vs_c->outside == 0 ? "PASS" : "FAIL",
            vs_c->max_abs_diff, vs_c->max_rel_diff * 100.0, vs_c->outside, m);

    printf("Check vs double ref:     %s  max abs err=%.4f, max rel err=%.6f%%, outside tolerance=%zu/%zu\n",
        vs_ref->outside == 0 ? "PASS" : "FAIL",
        vs_ref->max_abs_diff, vs_ref->max_rel_diff * 100.0, vs_ref->outside, m);
}

// Usage: SIMD-programming-project.exe [n] [m] [reps] [--force]
//        SIMD-programming-project.exe --selftest
//   n    : number of columns / length of X   (default 2^25)
//   m    : number of rows / length of Y      (default 10)
//   reps : timed repetitions per kernel      (default 30)
//   --force    : skip the physical-memory safety check
//   --selftest : run only the exact-integer indexing/tail test
int main(int argc, char** argv)
{
    size_t m = 10;
    size_t n = (size_t)1 << 25;
    int reps = 30;
    int force = 0;

    int positional = 0;
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--force") == 0) { force = 1; continue; }
        if (strcmp(argv[a], "--selftest") == 0) {
            if (!cpu_supports_avx()) {
                printf("This CPU/OS does not support AVX; the XMM/YMM kernels cannot run.\n");
                return 1;
            }
            return exact_selftest() ? 0 : 3;
        }
        size_t v;
        if (!parse_size(argv[a], &v)) {
            printf("Invalid argument: %s\n", argv[a]);
            return 1;
        }
        if (positional == 0) n = v;
        else if (positional == 1) m = v;
        else if (positional == 2) reps = (int)v;
        positional++;
    }
    if (reps < 1) reps = 1;

#ifdef _DEBUG
    const char* config = "Debug";
#else
    const char* config = "Release";
#endif

    printf("\n========== Matrix-Vector Product ==========\n");
    printf("Build configuration: %s\n", config);
    printf("Matrix dimensions: %zu x %zu (m x n), X length %zu, Y length %zu\n", m, n, n, m);
    printf("Repetitions: %d\n", reps);
    printf("Block size: %d columns\n", BLOCK_SIZE);

    if (!cpu_supports_avx()) {
        printf("This CPU/OS does not support AVX; the XMM/YMM kernels cannot run.\n");
        return 1;
    }
    printf("CPU: AVX supported, AVX2 %s\n", cpu_supports_avx2() ? "supported" : "NOT supported");

    // Overflow-checked buffer sizes
    size_t elems_A, bytes_A, bytes_X, bytes_Y, bytes_ref;
    if (!checked_mul(m, n, &elems_A) ||
        !checked_mul(elems_A, sizeof(float), &bytes_A) ||
        !checked_mul(n, sizeof(float), &bytes_X) ||
        !checked_mul(m, sizeof(float), &bytes_Y) ||
        !checked_mul(m, sizeof(double), &bytes_ref)) {
        printf("Requested size overflows size_t.\n");
        return 1;
    }
    double total_bytes = (double)bytes_A + bytes_X + 4.0 * bytes_Y + bytes_ref;

    MEMORYSTATUSEX mem = { sizeof(mem) };
    GlobalMemoryStatusEx(&mem);
    printf("Memory required: %.3f GiB (A %.3f GiB, X %.3f GiB); "
        "physical RAM %.2f GiB total, %.2f GiB available\n",
        total_bytes / 1073741824.0, bytes_A / 1073741824.0, bytes_X / 1073741824.0,
        mem.ullTotalPhys / 1073741824.0, mem.ullAvailPhys / 1073741824.0);

    // Refuse sizes that cannot fit in physical memory: they would either fail
    // to allocate or thrash the page file and make the timings meaningless.
    if (!force && total_bytes > 0.8 * (double)mem.ullTotalPhys) {
        printf("NOT RUN: required memory exceeds 80%% of physical RAM. "
            "Use a smaller n or m (or --force at your own risk).\n");
        return 2;
    }
    if (total_bytes > (double)mem.ullAvailPhys)
        printf("WARNING: required memory exceeds currently available RAM; "
            "paging may distort timings. Close other programs and re-run.\n");

    // Allocate arrays dynamically
    float* A = malloc(bytes_A);
    float* X = malloc(bytes_X);
    float* Y_c = malloc(bytes_Y);
    float* Y_scalar = malloc(bytes_Y);
    float* Y_xmm = malloc(bytes_Y);
    float* Y_ymm = malloc(bytes_Y);
	double* Y_ref = malloc(bytes_ref);

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

	// Initialize matrix and vector
	for (size_t i = 0; i < elems_A; i++) {
		A[i] = (float)((i % 17) + 1) * 0.1f;
    }

	for (size_t i = 0; i < n; i++) {
		X[i] = (float)((i % 13) + 1) * 0.1f;
	}

    // Fill outputs with NaN so a missed or stale element fails the checks
    for (size_t i = 0; i < m; i++)
        Y_c[i] = Y_scalar[i] = Y_xmm[i] = Y_ymm[i] = NAN;

	// Run each kernel once to populate the output arrays (also warms caches)
    kernel_c(A, X, Y_c, m, n);
    kernel_scalar(A, X, Y_scalar, m, n);
    kernel_xmm(A, X, Y_xmm, m, n);
    kernel_ymm(A, X, Y_ymm, m, n);
	kernel_reference(A, X, Y_ref, m, n);

    // Correctness (outside the timed region)
    check_result c_vs_ref = reference_check(Y_ref, Y_c, m);

    printf("\nChecking Scalar x86-64 against C...\n");
    check_result scalar_vs_c = correctness_check(Y_c, Y_scalar, m);
    check_result scalar_vs_ref = reference_check(Y_ref, Y_scalar, m);
    printf("Checking SIMD XMM against C...\n");
    check_result xmm_vs_c = correctness_check(Y_c, Y_xmm, m);
    check_result xmm_vs_ref = reference_check(Y_ref, Y_xmm, m);
    printf("Checking SIMD YMM against C...\n");
    check_result ymm_vs_c = correctness_check(Y_c, Y_ymm, m);
    check_result ymm_vs_ref = reference_check(Y_ref, Y_ymm, m);

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
    print_kernel("C", Y_c, m, time_c, time_c, NULL, &c_vs_ref, 1);
    print_kernel("Scalar x86-64", Y_scalar, m, time_scalar, time_c, &scalar_vs_c, &scalar_vs_ref, 0);
    print_kernel("SIMD XMM", Y_xmm, m, time_xmm, time_c, &xmm_vs_c, &xmm_vs_ref, 0);
    print_kernel("SIMD YMM", Y_ymm, m, time_ymm, time_c, &ymm_vs_c, &ymm_vs_ref, 0);

    printf("\nDouble reference: first Y = %.4f, last Y = %.4f\n", Y_ref[0], Y_ref[m - 1]);
    printf("Tolerance: %.1f%% relative (allowed = %.3f * max(1, |expected|))\n",
        TOLERANCE * 100.0, TOLERANCE);

    int all_pass = scalar_vs_c.outside == 0 && xmm_vs_c.outside == 0 && ymm_vs_c.outside == 0;
    printf("Overall check vs C: %s\n", all_pass ? "ALL PASS" : "FAILURES");

    // Release allocated memory
	free(A);
	free(X);
    free(Y_c);
    free(Y_scalar);
    free(Y_xmm);
    free(Y_ymm);
    free(Y_ref);

    return all_pass ? 0 : 3;
}
