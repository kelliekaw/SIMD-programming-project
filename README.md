# SIMD Programming Project: Matrix-Vector Product

## 1. Project Overview

This project implements and compares matrix-vector multiplication using four approaches:

1. **C implementation** — the baseline and the answer key for the correctness checks.
2. **Scalar x86-64 assembly** — processes one element at a time with scalar SSE instructions (`movss`, `mulss`, `addss`).
3. **SIMD XMM assembly** — uses 128-bit XMM registers to process four single-precision values per vector instruction.
4. **SIMD YMM assembly** — uses 256-bit YMM registers to process eight single-precision values per vector instruction.

The operation computes the product of a matrix $A$ and a vector $X$, producing an output vector $Y$ of length $m$:

$$
Y_i = \sum_{j=0}^{n-1} A_{ij}X_j
$$

The implementations are evaluated on execution time, speedup, agreement with the C baseline, numerical accuracy against a double-precision reference, and their handling of input sizes that are not multiples of the SIMD vector width.

## 2. Experimental Setup

* **Programming languages:** C and x86-64 assembly (NASM syntax)
* **SIMD instructions:** VEX-encoded AVX floating-point instructions on XMM (128-bit) and YMM (256-bit) registers. The kernels use `vmovups`, `vmulps`, `vaddps`, `vhaddps` and `vextractf128`, which are part of the AVX instruction set that every AVX2 processor supports. The program checks for AVX support at startup and reports whether AVX2 is present.
* **Data type:** single-precision `float` for the matrix, the vector, all arithmetic and the output of all four kernels
* **Diagnostic reference:** a separate, unblocked double-precision (`double`) accumulation
* **Timing:** Windows `QueryPerformanceCounter`, kernel calls only
* **Repetitions:** 30 timed repetitions per kernel, after one untimed warm-up call
* **Matrix storage:** one contiguous array in row-major order; element $(i, j)$ is `A[i * n + j]`

**Test machine:**

| Item | Value |
| --- | --- |
| CPU | Intel Core i5-11300H (4 cores / 8 threads, 3.1 GHz base); AVX and AVX2 supported |
| RAM | 7.70 GiB visible to Windows (only about 2 GiB was free during testing) |
| OS | Windows 11 Home (10.0.26200) |
| C compiler used for the measurements | MSVC 14.29 (Visual Studio 2019 Build Tools, toolset v142), via `build.cmd` |
| Assembler | NASM 2.15.05 |

The project file targets toolset v145 (Visual Studio 2026). The Visual Studio 2026 installation on the test machine does not have the C++ workload installed, so the measurements were built with `build.cmd`, which calls `cl.exe` with the same compiler options as the project's Debug and Release configurations. The `.vcxproj` was also built successfully with MSBuild for both configurations using `/p:PlatformToolset=v142`.

### Matrix Dimensions: $m = 10$ vs. a Square Matrix

The assignment says that a square matrix ($m = n$) **may be assumed**. This project instead uses a rectangular matrix with $m = 10$ rows and $n$ columns. The kernels themselves accept any $m$ and $n$.

A square matrix at the required sizes cannot be stored on any ordinary computer:

| $n$ | $A$ with $m = 10$ | $X$ | Total with $m = 10$ | $A$ if square ($m = n$) |
| ---: | ---: | ---: | ---: | ---: |
| 1,003 | 39 KiB | 3.9 KiB | 43 KiB | 3.8 MiB |
| $2^{20}$ | 40 MiB | 4 MiB | 44 MiB | 4 TiB |
| $2^{25}$ | 1.25 GiB | 128 MiB | 1.375 GiB | 4 PiB |
| $2^{28}$ | 10 GiB | 1 GiB | 11 GiB | 256 PiB |
| $2^{29}$ | 20 GiB | 2 GiB | 22 GiB | 1 EiB |
| $2^{30}$ | 40 GiB | 4 GiB | 44 GiB | 4 EiB |

The four output buffers and the reference buffer hold only $m$ values each (160 bytes when $m = 10$), so they are negligible.

Because a square $2^{20} \times 2^{20}$ matrix alone needs 4 TiB, $n$ is interpreted as the length of the input vector $X$, and the row count is fixed at a small value. **This is a deliberate deviation from the square-matrix assumption, not an implementation of it.** To show that the kernels also work for square matrices, separately labeled square tests ($1003 \times 1003$ and $8192 \times 8192$) are included in Section 4.

### Matrix and Vector Initialization

The matrix $A$ and input vector $X$ are initialized with deterministic, repeating patterns instead of random values. All four kernels and the reference therefore read exactly the same arrays.

```c
for (size_t i = 0; i < m * n; i++)
    A[i] = (float)((i % 17) + 1) * 0.1f;
```

The matrix elements cycle through values from about 0.1 to 1.7.

```c
for (size_t i = 0; i < n; i++)
    X[i] = (float)((i % 13) + 1) * 0.1f;
```

The vector elements cycle through values from about 0.1 to 1.3. The average product $A_{ij}X_j$ is about 0.63, so each output element is roughly $0.63 \times n$.

The index `i` is a 64-bit `size_t`, so `m * n` does not overflow for any tested size. All buffer sizes are computed with an overflow-checked multiplication before allocation.

**Note on $n = 1003$:** because $1003 = 17 \times 59$, every row of the $10 \times 1003$ matrix starts at the same point in the modulo-17 pattern, so all ten rows are identical. A bug in the row offset would not be visible in that test. For that reason, the boundary tests were extended to $n = 1004$–$1010$, $4097$, $4099$ and $4103$, and an exact-integer self-test was added (see Section 3.3).

### Test Cases

| Test case | $m$ | $n$ | Status on the test machine |
| --- | ---: | ---: | --- |
| Boundary test | 10 | 1,003 | Run (Debug and Release) |
| Extra boundary tests | 10 | 1,004–1,010, 4,097, 4,099, 4,103 | Run (Debug and Release) |
| Case 1 | 10 | $2^{20}$ = 1,048,576 | Run (Debug and Release) |
| Case 2 | 10 | $2^{25}$ = 33,554,432 | Run (Debug and Release) |
| Case 3 | 10 | $2^{30}$ = 1,073,741,824 | **Not run:** needs 44 GiB; the machine has 7.7 GiB |
| Fallback | 10 | $2^{29}$ and $2^{28}$ | **Not run:** needs 22 GiB and 11 GiB |
| Reduced-row fallback | **1** | $2^{28}$ = 268,435,456 | Run (Debug and Release); needs 2 GiB |
| Square (extra) | 1,003 | 1,003 | Run (Debug and Release) |
| Square (extra) | 8,192 | 8,192 | Run (Debug and Release) |

See Section 6 for the full memory-feasibility discussion.

## 3. Program Output and Correctness

### 3.1 Screenshots

The program prints, for every kernel: the average execution time, the speedup over C, the first five and last five elements of $Y$, the required check against C, and the double-reference diagnostic.

> **Screenshots still to be captured.** The screenshots below must be retaken with the current version of the program. The commands for each one are listed in Section 11. Until then, the exact program output is reproduced as text in Section 3.5.

#### Debug Version

`[INSERT SCREENSHOT: screenshots/debug_n1003.png — build\Debug\matvec.exe 1003 10 30]`

`[INSERT SCREENSHOT: screenshots/debug_n2p20.png — build\Debug\matvec.exe 1048576 10 30]`

`[INSERT SCREENSHOT: screenshots/debug_n2p25.png — build\Debug\matvec.exe 33554432 10 30]`

`[INSERT SCREENSHOT: screenshots/debug_n2p28_m1.png — build\Debug\matvec.exe 268435456 1 30]`

#### Release Version

`[INSERT SCREENSHOT: screenshots/release_n1003.png — build\Release\matvec.exe 1003 10 30]`

`[INSERT SCREENSHOT: screenshots/release_n2p20.png — build\Release\matvec.exe 1048576 10 30]`

`[INSERT SCREENSHOT: screenshots/release_n2p25.png — build\Release\matvec.exe 33554432 10 30]`

`[INSERT SCREENSHOT: screenshots/release_n2p28_m1.png — build\Release\matvec.exe 268435456 1 30]`

#### $n = 2^{30}$ (not run)

`[INSERT SCREENSHOT: screenshots/n2p30_not_run.png — build\Release\matvec.exe 1073741824 10 30, showing the NOT RUN memory message]`

#### Self-test

`[INSERT SCREENSHOT: screenshots/selftest.png — build\Release\matvec.exe --selftest]`

#### Historical screenshots (before the fix)

These were taken with the earlier version of the program, which checked only against the double-precision reference and used an unblocked accumulation. They are kept for reference only. Their correctness results are explained in Section 3.4.

<details>
<summary>Show historical screenshots</summary>

Debug, $n = 2^{20}$: ![220 debug](screenshots/220_debug.png)

Debug, $n = 2^{25}$: ![225 debug](screenshots/225_debug.png)

Release, $n = 2^{20}$: ![220 release](screenshots/220_release.png)

Release, $n = 2^{25}$: ![225 release](screenshots/225_release.png)

Boundary, $n = 1003$: ![1003](screenshots/1003.png)

</details>

### 3.2 Correctness Checking and Tolerance

The program performs two separate comparisons, and they answer different questions.

**1. Required check: each assembly kernel against C.** As the assignment requires, the C output is the answer key. The scalar, XMM and YMM outputs are each compared with it element by element.

**2. Additional diagnostic: each kernel against a double-precision reference.** A separate function, `kernel_reference`, computes every row with an ordinary, unblocked `double` accumulation. This measures how close each kernel is to the mathematically correct value. Agreeing with C does not prove a result is accurate, which is exactly what happened before the fix (Section 3.4).

Both comparisons use the same relative tolerance of 0.5%. For each output element:

$$
\text{Difference}_i = \left|Y_{\text{expected},i} - Y_{\text{test},i}\right|
$$

$$
\text{Allowed Error}_i = 0.005 \times \max\left(1, \left|Y_{\text{expected},i}\right|\right)
$$

The $\max(1, \cdot)$ term prevents a division-like blow-up when the expected value is near zero. An element passes when $\text{Difference}_i \le \text{Allowed Error}_i$. A kernel passes only when **all** $m$ elements pass.

For each kernel and each check, the program reports the maximum absolute difference, the maximum relative difference, and the number of elements outside the tolerance. The test is written as `!(difference <= allowed)`, so a `NaN` result counts as a failure. The output buffers are filled with `NaN` before the kernels run, so an element that a kernel never writes is detected.

Floating-point addition is not associative. The four kernels add the same products in different orders, so their results are not bit-identical. The tolerance accepts ordinary rounding differences, while still catching a missing row, a skipped element or a wrong index, which would change the result by far more than 0.5%.

### 3.3 Exact-Integer Self-Test (Indexing and Tail Handling)

A tolerance check cannot prove that every element is processed exactly once. The program therefore also has a self-test mode:

```
build\Release\matvec.exe --selftest
```

It fills $A$ with whole numbers that depend on both the row and the column, and $X$ with whole numbers 1–5. All partial sums are whole numbers below $2^{24}$, which `float` represents exactly, so every kernel must return the exact answer regardless of its addition order. The arrays are surrounded by "poison" values (1000.0) so that an out-of-bounds read changes the result. Two sentinel values after $Y_{m-1}$ detect out-of-bounds writes.

The test runs all four kernels for $m \in \{1, 3, 10\}$ and $n = 1, 2, \dots, 40$ (every remainder modulo 4 and 8, including $n < 4$ and $n < 8$). It also covers the boundary sizes 1003–1011, 4095–4097 and 4103–4105 (around the 4096-element block boundary), 8191–8193 and 12291.

**Result:** 708 kernel runs, 0 failures, in both Debug and Release. To confirm that the test actually detects bugs, it was also run against deliberately broken kernels. A wrong row stride in the YMM kernel produced 118 failures, and a scalar tail that skipped every other element in the XMM kernel produced 87 failures.

### 3.4 Root Cause of the Earlier Correctness Failures

**Earlier behavior (reproduced by rebuilding the previous commit).** At $n = 2^{25}$ the original program produced:

| Kernel | First output $Y_0$ | Error vs. double reference |
| --- | ---: | ---: |
| Double reference | 21,139,291.85 | — |
| C | 19,294,918 | −8.73% |
| Scalar x86-64 | 19,294,918 | −8.73% |
| SIMD XMM | 20,955,688 | −0.87% |
| SIMD YMM | 21,081,736 | −0.27% |

The original program compared only against the double reference, so it reported C, scalar and XMM as FAIL and YMM as PASS. Had it compared against C as the assignment requires, **XMM and YMM would have failed** (they differed from C by 8.6% and 9.3%), even though they were the more accurate results.

**There was no indexing, loop-bound, tail-handling or calling-convention bug.** This was verified in three ways:

1. The exact-integer self-test (Section 3.3) passes for the original kernels as well as the current ones.
2. A plain C simulation that adds the same products with one `float` accumulator, four `float` lanes and eight `float` lanes reproduces the C, XMM and YMM outputs above **bit for bit**.
3. Reading the assembly confirmed the correct Windows x64 calling convention (fifth argument at `[rsp+96]` after seven pushes), the preservation of all non-volatile registers used, a row stride of `n * 4` bytes, the four-lane and eight-lane reductions, and `vzeroupper` before returning from the YMM kernel.

**Actual cause: accumulation error in single precision.** Every kernel used one running `float` sum per row (per lane for SIMD) across all $n$ columns. A `float` has a 24-bit significand, so the gap between neighboring representable values grows with the size of the number:

| Size of the running sum | Gap between adjacent `float` values |
| --- | ---: |
| $2^{21}$ to $2^{22}$ (about 2–4 million) | 0.25 |
| $2^{22}$ to $2^{23}$ (about 4–8 million) | 0.5 |
| $2^{23}$ to $2^{24}$ (about 8–17 million) | 1 |
| above $2^{24}$ (above about 17 million) | 2 |

Each new product is about 0.01–2.2, with an average of 0.63. Once the C sum passes about 8 million, every addition is rounded to the nearest whole number, and once it passes about 16.8 million, to the nearest even number. Most products are smaller than 1, so they are rounded down, often all the way to zero. The final C result was therefore 8.7% too small.

The SIMD kernels split the same sum across four or eight independent lanes. Each lane only reaches $1/4$ or $1/8$ of the total (about 5.3 million or 2.6 million), where the gaps are smaller, so less is lost. That is why XMM was closer to the reference and YMM closer still. The difference between the kernels was purely a consequence of their different addition orders. It was not a bug in the SIMD kernels, but it did make the C answer key unreliable at large $n$.

**Fix: blocked summation, applied identically to all four kernels.** Each row is now processed in blocks of 4,096 columns (`BLOCK_SIZE` in `main.c` and in each `.asm` file). Within a block, the kernel accumulates exactly as before: one `float` sum for C and scalar assembly, four lanes for XMM, eight lanes for YMM. At the end of each block, the block's sum is added to a `float` row total, and the block sum is reset to zero. A block sum stays below about 2,600, where the `float` gap is about 0.0002, so the small products are no longer rounded away.

* All arithmetic is still single precision. No kernel was converted to `double`.
* The tolerance (0.5%) and the input data were not changed.
* The vector loops are unchanged; the extra work is one reduction and one addition per 4,096 elements, which had no measurable cost (see Section 5.5).
* `BLOCK_SIZE` is a multiple of 8, so only the last block of a row can have a scalar tail.

Before choosing the block size, the error at $n = 2^{20}$, $2^{25}$, $2^{28}$ and $2^{30}$ was simulated for block sizes from 1,024 to 65,536. With 4,096, the error stayed at or below 0.024% for all sizes up to $2^{30}$. All Release and Debug builds are compiled with `/fp:precise` (now set explicitly in the project file), which prevents the compiler from reordering the C additions.

### 3.5 Recorded Results After the Fix

All results below come from the current program. Debug and Release produced **bit-identical** outputs for every test, so the values are shown once.

**Required check (each kernel vs. C), 0.5% tolerance:**

| Test case | Scalar x86-64 | SIMD XMM | SIMD YMM | Largest relative difference from C |
| --- | --- | --- | --- | ---: |
| $n = 1003$ ($m = 10$) | Pass (0/10 outside) | Pass (0/10) | Pass (0/10) | 0.000039% |
| $n = 1004$–$1010$, 4097, 4099, 4103 | Pass | Pass | Pass | 0.000331% |
| $n = 2^{20}$ ($m = 10$) | Pass (0/10) | Pass (0/10) | Pass (0/10) | 0.000369% |
| $n = 2^{25}$ ($m = 10$) | Pass (0/10) | Pass (0/10) | Pass (0/10) | 0.000473% |
| $n = 2^{28}$ ($m = 1$, fallback) | Pass (0/1) | Pass (0/1) | Pass (0/1) | 0.000161% |
| Square $1003 \times 1003$ | Pass (0/1003) | Pass (0/1003) | Pass (0/1003) | 0.000039% |
| Square $8192 \times 8192$ | Pass (0/8192) | Pass (0/8192) | Pass (0/8192) | 0.000322% |
| $n = 2^{30}$, $2^{29}$, $2^{28}$ ($m = 10$) | Not run (insufficient memory) | — | — | — |

The scalar assembly result is bit-identical to C in every test (maximum difference 0), because both add the products in exactly the same order.

**Additional diagnostic (each kernel vs. double reference), largest relative error:**

| Test case | C | Scalar x86-64 | SIMD XMM | SIMD YMM |
| --- | ---: | ---: | ---: | ---: |
| $n = 1003$ | 0.000035% | 0.000035% | 0.000004% | 0.000004% |
| $n = 2^{20}$ | 0.000280% | 0.000280% | 0.000089% | 0.000051% |
| $n = 2^{25}$ | 0.000370% | 0.000370% | 0.000842% | 0.000842% |
| $n = 2^{28}$ ($m = 1$) | 0.023638% | 0.023638% | 0.023799% | 0.023799% |
| Square $8192 \times 8192$ | 0.000281% | 0.000281% | 0.000042% | 0.000027% |

All kernels pass the diagnostic as well. At $n = 2^{25}$ the error of C dropped from 8.73% to 0.00037%. The larger error at $2^{28}$ (0.024%) comes from the row total: it reaches about 169 million, where the `float` gap is 16, and 65,536 block sums are added to it. This is still about 20 times below the tolerance.

**First and last five output values (Release, identical in Debug):**

<details>
<summary>n = 1003 (m = 10)</summary>

```
Double reference: first Y = 630.9900, last Y = 630.9900
C / Scalar   first: 630.9898 630.9898 630.9898 630.9898 630.9898   last: 630.9898 630.9898 630.9898 630.9898 630.9898
XMM / YMM    first: 630.9901 630.9901 630.9901 630.9901 630.9901   last: 630.9901 630.9901 630.9901 630.9901 630.9901
```
(All rows are equal because $1003$ is a multiple of 17; see Section 2.)
</details>

<details>
<summary>n = 2^20 (m = 10)</summary>

```
Double reference: first Y = 660599.8981, last Y = 660602.6581
C / Scalar  first: 660601.5000 660601.6875 660601.1875 660599.9375 660599.9375
            last:  660601.9375 660603.0000 660603.5000 660603.1250 660604.3125
SIMD XMM    first: 660599.6875 660599.7500 660599.0000 660597.7500 660598.0000
            last:  660600.0625 660600.9375 660601.0625 660601.0000 660602.6875
SIMD YMM    first: 660599.7500 660599.8125 660599.1250 660597.9375 660598.1250
            last:  660600.1875 660601.0000 660601.3125 660601.2500 660602.7500
```
</details>

<details>
<summary>n = 2^25 (m = 10) — full program output, Release</summary>

```
========== Matrix-Vector Product ==========
Build configuration: Release
Matrix dimensions: 10 x 33554432 (m x n), X length 33554432, Y length 10
Repetitions: 30
Block size: 4096 columns
CPU: AVX supported, AVX2 supported
Memory required: 1.375 GiB (A 1.250 GiB, X 0.125 GiB); physical RAM 7.70 GiB total, 1.96 GiB available

Checking Scalar x86-64 against C...
Checking SIMD XMM against C...
Checking SIMD YMM against C...

--- C ---
Average execution time: 345.269953 ms   Speedup vs C: 1.00x
First 5 Y values: 21139216.0000 21139224.0000 21139214.0000 21139218.0000 21139214.0000
Last 5 Y values:  21139218.0000 21139216.0000 21139214.0000 21139216.0000 21139214.0000
Check vs C (required):   BASELINE (answer key)
Check vs double ref:     PASS  max abs err=78.2699, max rel err=0.000370%, outside tolerance=0/10

--- Scalar x86-64 ---
Average execution time: 337.659943 ms   Speedup vs C: 1.02x
First 5 Y values: 21139216.0000 21139224.0000 21139214.0000 21139218.0000 21139214.0000
Last 5 Y values:  21139218.0000 21139216.0000 21139214.0000 21139216.0000 21139214.0000
Check vs C (required):   PASS  max abs diff=0.0000, max rel diff=0.000000%, outside tolerance=0/10
Check vs double ref:     PASS  max abs err=78.2699, max rel err=0.000370%, outside tolerance=0/10

--- SIMD XMM ---
Average execution time: 164.062660 ms   Speedup vs C: 2.10x
First 5 Y values: 21139118.0000 21139128.0000 21139114.0000 21139124.0000 21139114.0000
Last 5 Y values:  21139124.0000 21139116.0000 21139124.0000 21139116.0000 21139120.0000
Check vs C (required):   PASS  max abs diff=100.0000, max rel diff=0.000473%, outside tolerance=0/10
Check vs double ref:     PASS  max abs err=178.0899, max rel err=0.000842%, outside tolerance=0/10

--- SIMD YMM ---
Average execution time: 158.816643 ms   Speedup vs C: 2.17x
First 5 Y values: 21139118.0000 21139128.0000 21139114.0000 21139124.0000 21139114.0000
Last 5 Y values:  21139124.0000 21139116.0000 21139124.0000 21139116.0000 21139120.0000
Check vs C (required):   PASS  max abs diff=100.0000, max rel diff=0.000473%, outside tolerance=0/10
Check vs double ref:     PASS  max abs err=178.0899, max rel err=0.000842%, outside tolerance=0/10

Double reference: first Y = 21139291.8499, last Y = 21139291.8799
Tolerance: 0.5% relative (allowed = 0.005 * max(1, |expected|))
Overall check vs C: ALL PASS
```
</details>

<details>
<summary>n = 2^28, m = 1 (reduced-row fallback)</summary>

```
Double reference: Y = 169114343.4034
C / Scalar  Y = 169074368.0000
XMM / YMM   Y = 169074096.0000
```
</details>

### 3.6 Correctness by Kernel

**C.** The baseline. After the fix, its error against the double reference is at most 0.00037% for $n \le 2^{25}$ and 0.024% at $n = 2^{28}$. Before the fix it was 8.7% at $2^{25}$, which is why it could not serve as a reliable answer key.

**Scalar x86-64.** Uses the same block structure and the same addition order as C, and produces bit-identical outputs in every test. This was verified independently of C by the exact-integer self-test, not just inferred from matching C.

**SIMD XMM.** Each 4,096-column block is accumulated in four lanes, reduced with two `vhaddps`, and any leftover elements (only possible in the last block) are added with scalar instructions. It matches C within 0.00047% in every test.

**SIMD YMM.** Each block is accumulated in eight lanes. The upper 128 bits are added to the lower 128 bits with `vextractf128` + `vaddps`, then reduced with two `vhaddps`, followed by the scalar tail. It matches C within 0.00047% in every test. At $2^{25}$ XMM and YMM produce identical outputs: their block sums differ slightly but round to the same values when added to the row total.

### 3.7 SIMD Boundary Checks

The boundary tests use values of $n$ that are not divisible by four or eight, so the scalar-tail code must run:

| $n$ | $n \bmod 4$ | $n \bmod 8$ | Result (Debug and Release) |
| ---: | ---: | ---: | --- |
| 1003 | 3 | 3 | Pass |
| 1004 | 0 | 4 | Pass |
| 1005 | 1 | 5 | Pass |
| 1006 | 2 | 6 | Pass |
| 1007 | 3 | 7 | Pass |
| 1008 | 0 | 0 | Pass |
| 1009 | 1 | 1 | Pass |
| 1010 | 2 | 2 | Pass |
| 4097 | 1 | 1 | Pass (tail in the second block) |
| 4099 | 3 | 3 | Pass |
| 4103 | 3 | 7 | Pass |

Together with the exact-integer self-test ($n = 1$ to $40$ and the block-boundary sizes), this covers every remainder modulo 4 and 8, rows shorter than one vector, and rows that end partway through a second block.

## 4. Comparative Execution-Time Results

Execution times are in milliseconds (ms). Each time is the average of 30 back-to-back kernel calls measured with one `QueryPerformanceCounter` interval, after one untimed warm-up call. Initialization, allocation, correctness checking and printing are all outside the timed region.

For $n = 2^{20}$ and $2^{25}$ the program was run **three times**. The tables show the mean of the three 30-run averages (90 calls in total), and the range across the three runs. The other sizes were run once (30 calls).

$$
\text{Speedup} = \frac{\text{Average C execution time}}{\text{Average execution time of the tested kernel}}
$$

Debug and Release results are shown in separate tables and must not be compared with each other: the C kernel is compiled differently in each, while the assembly kernels are identical.

### 4.1 Release Configuration

**$n = 1003$, $m = 10$ (single run)**

| Kernel | Time (ms) | Speedup vs. C |
| --- | ---: | ---: |
| C | 0.009323 | 1.00× |
| Scalar x86-64 | 0.009710 | 0.96× |
| SIMD XMM | 0.002337 | 3.99× |
| SIMD YMM | 0.001087 | 8.58× |

**$n = 2^{20}$, $m = 10$ (mean of 3 runs)**

| Kernel | Time (ms) | Range (ms) | Speedup vs. C |
| --- | ---: | ---: | ---: |
| C | 11.665 | 11.149–12.562 | 1.00× |
| Scalar x86-64 | 11.020 | 10.624–11.296 | 1.06× |
| SIMD XMM | 5.022 | 4.847–5.233 | 2.32× |
| SIMD YMM | 3.980 | 3.673–4.248 | 2.93× |

**$n = 2^{25}$, $m = 10$ (mean of 3 runs)**

| Kernel | Time (ms) | Range (ms) | Speedup vs. C |
| --- | ---: | ---: | ---: |
| C | 344.976 | 339.243–350.414 | 1.00× |
| Scalar x86-64 | 338.145 | 337.660–338.998 | 1.02× |
| SIMD XMM | 163.140 | 162.399–164.063 | 2.11× |
| SIMD YMM | 158.416 | 157.777–158.817 | 2.18× |

**$n = 2^{28}$, $m = 1$ — reduced-row fallback (single run)**

| Kernel | Time (ms) | Speedup vs. C |
| --- | ---: | ---: |
| C | 303.297 | 1.00× |
| Scalar x86-64 | 273.603 | 1.11× |
| SIMD XMM | 132.096 | 2.30× |
| SIMD YMM | 124.836 | 2.43× |

**Square matrices (single run)**

| Kernel | $1003 \times 1003$ (ms) | Speedup | $8192 \times 8192$ (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| C | 1.188 | 1.00× | 70.488 | 1.00× |
| Scalar x86-64 | 1.121 | 1.06× | 68.907 | 1.02× |
| SIMD XMM | 0.413 | 2.87× | 28.277 | 2.49× |
| SIMD YMM | 0.172 | 6.90× | 17.811 | 3.96× |

### 4.2 Debug Configuration

**$n = 1003$, $m = 10$ (single run)**

| Kernel | Time (ms) | Speedup vs. C |
| --- | ---: | ---: |
| C | 0.022563 | 1.00× |
| Scalar x86-64 | 0.009723 | 2.32× |
| SIMD XMM | 0.002350 | 9.60× |
| SIMD YMM | 0.001423 | 15.85× |

**$n = 2^{20}$, $m = 10$ (mean of 3 runs)**

| Kernel | Time (ms) | Range (ms) | Speedup vs. C |
| --- | ---: | ---: | ---: |
| C | 23.693 | 23.457–23.843 | 1.00× |
| Scalar x86-64 | 10.748 | 10.324–11.393 | 2.20× |
| SIMD XMM | 5.746 | 3.786–7.038 | 4.12× |
| SIMD YMM | 3.698 | 3.280–3.939 | 6.41× |

**$n = 2^{25}$, $m = 10$ (mean of 3 runs)**

| Kernel | Time (ms) | Range (ms) | Speedup vs. C |
| --- | ---: | ---: | ---: |
| C | 739.024 | 734.742–741.402 | 1.00× |
| Scalar x86-64 | 340.916 | 337.797–346.652 | 2.17× |
| SIMD XMM | 163.603 | 162.470–164.720 | 4.52× |
| SIMD YMM | 161.894 | 158.616–165.298 | 4.57× |

**$n = 2^{28}$, $m = 1$ — reduced-row fallback (single run)**

| Kernel | Time (ms) | Speedup vs. C |
| --- | ---: | ---: |
| C | 632.471 | 1.00× |
| Scalar x86-64 | 299.116 | 2.11× |
| SIMD XMM | 156.930 | 4.03× |
| SIMD YMM | 149.821 | 4.22× |

**Square matrices (single run)**

| Kernel | $1003 \times 1003$ (ms) | Speedup | $8192 \times 8192$ (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| C | 2.508 | 1.00× | 162.892 | 1.00× |
| Scalar x86-64 | 1.075 | 2.33× | 71.778 | 2.27× |
| SIMD XMM | 0.277 | 9.07× | 24.511 | 6.65× |
| SIMD YMM | 0.141 | 17.81× | 17.973 | 9.06× |

### 4.3 Historical Timings (Before the Fix)

The previous version of this README reported the following Release times. They were measured with the old, unblocked kernels and under different system conditions, so they are **not directly comparable** with the tables above.

| Kernel | $n = 2^{20}$ (ms) | $n = 2^{25}$ (ms) |
| --- | ---: | ---: |
| C | 8.324813 | 268.404857 |
| Scalar x86-64 | 8.932587 | 276.641373 |
| SIMD XMM | 2.884263 | 104.514147 |
| SIMD YMM | 2.224857 | 92.106480 |

## 5. Analysis

### 5.1 C vs. Scalar x86-64 Assembly

In **Release**, scalar assembly was 0.96×–1.11× the speed of C, essentially the same. The generated code for the C kernel (inspected with `/FAs`) uses scalar `mulss`/`addss` instructions and no packed instructions. Under `/fp:precise`, the compiler is not allowed to reorder the floating-point additions, so it cannot vectorize the loop. Both kernels therefore do the same work: one multiplication and one addition per element on a single chain of dependent additions.

At $n = 2^{25}$ both kernels process about one element per nanosecond (335.5 million elements in about 340 ms). That is consistent with the roughly 4-cycle latency of each dependent `addss` at the CPU's turbo frequency. The loop is limited by that dependency chain, not by the instruction count.

In **Debug**, C is about 2.2× slower than scalar assembly, because unoptimized code keeps `partial` and `total` on the stack and reloads them on every iteration. The assembly kernels are not affected by the compiler settings and take the same time in both configurations. The Debug "speedups" are therefore mostly a measure of how slow unoptimized C is.

### 5.2 SIMD XMM Performance

XMM processes four elements per vector instruction. When the data fits in cache ($n = 1003$, 39 KiB), it achieved **3.99×** over C in Release, almost exactly its vector width. For the large inputs, the speedup fell to 2.32× at $2^{20}$ and 2.11× at $2^{25}$.

### 5.3 SIMD YMM Performance

YMM processes eight elements per vector instruction. At $n = 1003$ it achieved **8.58×** over C in Release, again close to its vector width. It was the fastest kernel in every Release test. However, at $n = 2^{25}$ it was only 3% faster than XMM (158.4 ms vs. 163.1 ms).

### 5.4 Why the Measured Speedup Is Not Equal to the Vector Width

* **Memory bandwidth is the main limit for large inputs.** At $n = 2^{25}$, each call reads the 1.25 GiB matrix once. The 128 MiB vector $X$ does not fit in the 8 MB L3 cache, so it is read again for each of the 10 rows, which adds another 1.25 GiB. Moving about 2.7 GB in 158 ms is about 17 GB/s, and XMM reaches nearly the same rate. Once both SIMD kernels are waiting on memory, a wider register cannot help, which is why XMM and YMM are almost equal at $2^{25}$ and at $2^{28}$.
* **In-cache data shows the true SIMD effect.** With small data ($n = 1003$), nothing waits on memory, and every kernel is limited by its single chain of dependent additions. Each XMM addition handles four elements and each YMM addition eight, so the speedups (3.99× and 8.58×) match the vector widths.
* **Medium sizes fall in between.** At $n = 2^{20}$ the matrix (40 MiB) is streamed from memory, but $X$ (4 MiB) can stay in the L3 cache. The square $8192 \times 8192$ case (a 256 MiB matrix, but only a 32 KiB vector) behaves similarly. Both give speedups between the two extremes.
* **One accumulator per kernel.** Each kernel uses a single accumulator register, so every addition waits for the previous one. Using several independent accumulators would let the processor overlap additions and would speed up in-cache sizes further. It would not help the memory-bound sizes.
* **Reduction and tail overhead** are small. There is one horizontal reduction per 4,096 elements, and at most seven scalar tail elements per row.
* **Measurement variation.** Background activity and CPU frequency changes on a laptop cause run-to-run variation. Outside the three recorded runs, single runs of the C kernel at $2^{25}$ took 443 ms and 377.5 ms, compared with 339–350 ms in the table. At $2^{20}$ another run gave 10.47 ms, below the table's range. The SIMD times varied much less (162–165 ms at $2^{25}$). That is why the main sizes were measured three times, and why speedups should be read as approximate (about ±10%).

### 5.5 Cost of the Accuracy Fix

The blocked summation does not measurably slow the kernels. The previous commit, rebuilt and run on the same machine at $n = 2^{25}$ (Release), measured C 431.8 ms, scalar 381.5 ms, XMM 214.2 ms and YMM 202.4 ms. The new kernels measured 339–350, 338–339, 162–164 and 158–159 ms respectively. The differences come from system conditions rather than the added block loop.

## 6. Memory Requirements and the $n = 2^{30}$ Test

With $m = 10$, the program needs $(10n + n) \times 4$ bytes for $A$ and $X$, plus a negligible amount for the outputs. The full table is in Section 2.

| Size | Memory needed ($m = 10$) | Test machine (7.70 GiB RAM, about 2 GiB free) |
| --- | ---: | --- |
| $n = 2^{30}$ | 44 GiB | Not feasible: more than five times the installed RAM, and above the 20 GiB Windows commit limit |
| $n = 2^{29}$ | 22 GiB | Not feasible: above the commit limit |
| $n = 2^{28}$ | 11 GiB | Not feasible: larger than physical RAM; the run would page constantly and the timings would be meaningless |
| $n = 2^{25}$ | 1.375 GiB | Run successfully |

Before allocating anything, the program computes the required memory with overflow-checked multiplication and compares it with the installed RAM (`GlobalMemoryStatusEx`). If the requirement exceeds 80% of physical memory, it prints `NOT RUN` and exits with code 2 instead of allocating. If the requirement exceeds the memory currently available, it prints a warning that paging may distort the timings. Any `malloc` failure is also reported cleanly. The $2^{30}$, $2^{29}$ and $2^{28}$ cases were attempted this way in both Debug and Release, and all three were refused by this check:

```
Matrix dimensions: 10 x 1073741824 (m x n), X length 1073741824, Y length 10
Memory required: 44.000 GiB (A 40.000 GiB, X 4.000 GiB); physical RAM 7.70 GiB total, 2.12 GiB available
NOT RUN: required memory exceeds 80% of physical RAM. Use a smaller n or m (or --force at your own risk).
```

**The largest size tested with the original $m = 10$ configuration is $n = 2^{25}$.** As a clearly labeled substitute for the required $2^{28}$ fallback, the program was run with **$m = 1$ and $n = 2^{28}$** (2 GiB), which exercises the kernels on vectors of $2^{28}$ elements. This is a reduced-row configuration and is **not** a completion of the $m = 10$, $n = 2^{28}$ test. $m = 1$, $n = 2^{29}$ (4 GiB) would pass the 80% check but exceeds the memory that was free during testing, so it was not run. It could be attempted after closing other programs:

```
build\Release\matvec.exe 536870912 1 30
```

A computer with at least 16 GiB of RAM (for $m = 10$, $n = 2^{28}$) or 64 GiB (for $m = 10$, $n = 2^{30}$) would be needed for the full-size tests.

## 7. Boundary Handling in SIMD

SIMD instructions operate on fixed-width vectors: XMM handles four `float` values at once and YMM handles eight. When $n$ is not divisible by the vector width, the last elements cannot form a complete vector. Each SIMD kernel therefore processes every block of a row in three steps:

1. A vector loop over complete groups of four (XMM) or eight (YMM) elements.
2. A horizontal reduction of the vector lanes into one value.
3. A scalar-tail loop over the remaining elements (fewer than four or eight).

Because the block size (4,096) is a multiple of eight, only the last block of a row can have a tail. The loop condition checks that at least four (or eight) elements remain *before* loading a vector, so the kernels never read past the end of a row. This was confirmed by the poisoned memory in the self-test. The boundary tests in Section 3.7 and the self-test in Section 3.3 cover every remainder modulo 4 and 8.

## 8. Problems Encountered and Solutions

### 8.1 Memory Limitations for Large Input Sizes

Even with only 10 rows, $n = 2^{28}$ to $2^{30}$ need 11–44 GiB, far more than the 7.7 GiB on the test machine. Earlier attempts at large sizes caused heavy memory pressure and a slow system.

**Solution:** the program now calculates the memory requirement first and refuses sizes that cannot fit, instead of allocating blindly. The reduced-row $m = 1$, $n = 2^{28}$ test was added as a clearly labeled substitute (Section 6).

### 8.2 Floating-Point Discrepancies at Large $n$

At $n = 2^{25}$, the four kernels disagreed by up to 9%, and the C baseline was 8.7% below the true value. With C as the answer key, the more accurate SIMD kernels would have been marked as failing.

**Investigation:** the exact-integer self-test ruled out indexing and tail bugs, and a C simulation reproduced every kernel's output exactly from its addition order alone (Section 3.4).

**Solution:** blocked summation in all four kernels, keeping single precision and the original 0.5% tolerance. All kernels now agree with C within 0.0005%, and with the double reference within 0.03%.

### 8.3 Correctness Check Did Not Follow the Assignment

The earlier program compared all kernels only against the double-precision reference. The `correctness_check` function that compared against C existed but was never called. It also stopped at the first mismatch and did not detect `NaN`.

**Solution:** the program now runs the required comparison against C and, separately, the double-reference diagnostic. It counts every element outside the tolerance, reports the maximum absolute and relative differences, and treats `NaN` as a failure.

### 8.4 SIMD Boundary Handling and a Hidden Weakness in the $n = 1003$ Test

Without a scalar-tail loop, the leftover elements would be skipped. The kernels already handled the tail correctly. However, with this initialization, $n = 1003$ produces ten identical rows (Section 2), so that test alone could not detect a row-offset error.

**Solution:** additional boundary sizes, plus an exact-integer self-test with row-dependent values. The self-test was itself checked by confirming that it catches deliberately introduced bugs.

### 8.5 Build Portability

The project's pre-build step called NASM at a fixed path, `C:\mingw64\bin\nasm.exe`, which did not exist on the test machine. NASM was installed under `C:\Program Files (x86)\SASM\NASM\`.

**Solution:** the project file now uses a `NasmPath` property. It checks the `NASM` environment variable first, then common install locations, then `PATH`. `build.cmd` provides a command-line build that does not depend on the Visual Studio version.

### 8.6 Experimental Methodology

The same input data, the same C answer key and the same timing method are used for all four kernels. Timings vary between runs on a laptop, so the main sizes were measured three times and reported with their range. The results still apply only to the tested machine, compiler and conditions.

## 9. AHA Moments and Key Learnings

* **Matching the answer key is not the same as being correct.** Before the fix, scalar assembly matched C exactly, but both were 8.7% wrong. The SIMD kernels "disagreed" with C because they were *more* accurate.
* **The order of floating-point additions matters.** Splitting a sum across four or eight SIMD lanes made it more accurate by accident, because each lane's sum stayed smaller. Blocked summation does the same thing on purpose, for every kernel.
* **A tolerance check cannot find every bug.** An exact-integer test, where any correct order gives the same answer, separates real indexing bugs from rounding effects.
* **Test data can hide bugs.** With $n = 1003$, a multiple of the pattern length 17, every row was identical.
* **Writing assembly does not guarantee better performance.** Optimized C and scalar assembly perform within a few percent of each other because both are limited by the same chain of dependent additions.
* **Wider registers help only until memory becomes the limit.** YMM reached 8.6× in cache but only 2.2× when streaming 2.7 GB from memory, almost the same as XMM.
* **Memory capacity limits which experiments are possible.** Even with 10 rows, $n = 2^{30}$ needs 44 GiB.

## 10. Conclusion

All four kernels compute the matrix-vector product correctly. In every test that fit in memory, in both Debug and Release, the scalar, XMM and YMM outputs agree with the C answer key within 0.0005%, and every kernel agrees with an independent double-precision reference within 0.03%. The earlier large-$n$ failures were caused by single-precision accumulation error in a single running sum, not by an implementation bug. They were fixed with blocked summation, without changing the precision, the tolerance or the input data.

In the Release tests, SIMD YMM was the fastest kernel, followed by SIMD XMM. When the data fits in cache, their speedups match their vector widths (8.58× and 3.99× at $n = 1003$). For large inputs, memory bandwidth limits both to about 2.1–2.2× at $n = 2^{25}$. Scalar assembly performs about the same as optimized C.

The $n = 2^{30}$ test, and the $2^{29}$ and $2^{28}$ fallbacks with $m = 10$, could not be run because they need 11–44 GiB on a 7.7 GiB machine. The largest completed sizes are $n = 2^{25}$ with $m = 10$, and $n = 2^{28}$ with $m = 1$.

## 11. Reproducibility and Future Work

### Building from the Command Line (cmd.exe)

Requirements: Visual Studio or the Visual Studio Build Tools with the **Desktop development with C++** workload (any version from 2019 onward), and NASM.

```bat
cd /d C:\path\to\SIMD-programming-project

rem Build both configurations -> build\Debug\matvec.exe and build\Release\matvec.exe
build.cmd

rem If NASM is not found automatically:
set NASM=C:\path\to\nasm.exe
build.cmd
```

`build.cmd` finds a Visual Studio installation that has the C++ tools, assembles the three `.asm` files, and compiles `main.c` with the same options as the Visual Studio project (Debug: `/Od /RTC1 /MDd`; Release: `/O2 /Oi /GL /Gy /MD` with `/LTCG`; both with `/fp:precise`).

### Running the Tests

```bat
rem Full test matrix (self-test, boundary tests, 2^20, 2^25, 2^28-2^30 attempts, square tests).
rem Each run's output is saved to results\<config>\*.txt
run_tests.cmd Release
run_tests.cmd Debug

rem Individual runs:  matvec.exe [n] [m] [reps]
build\Release\matvec.exe --selftest
build\Release\matvec.exe 1003 10 30
build\Release\matvec.exe 1048576 10 30
build\Release\matvec.exe 33554432 10 30
build\Release\matvec.exe 268435456 1 30
build\Release\matvec.exe 1073741824 10 30
```

The defaults are $n = 2^{25}$, $m = 10$ and 30 repetitions. $n$ can also be written as `2^25` when the program is started from Visual Studio, but not from cmd.exe, where `^` is a special character. The exit code is 0 if all kernels match C, 3 if any kernel fails the check, 2 if the size was not run for lack of memory, and 1 for other errors.

### Building in Visual Studio

1. Open `SIMD-programming-project.vcxproj` in Visual Studio.
2. If the v145 toolset is not installed, use **Project → Properties → General → Platform Toolset** to select an installed one (for example v143 for Visual Studio 2022).
3. Select the **x64** platform and the **Debug** or **Release** configuration.
4. Under **Project → Properties → Debugging → Command Arguments**, enter the test arguments, for example `33554432 10 30`.
5. Press **Ctrl+F5** to run without the debugger. The pre-build step assembles the `.asm` files with NASM.

### Capturing the Screenshots

Run each command listed in Section 3.1 in a cmd window (or with Ctrl+F5 from Visual Studio). Take a screenshot with **Win+Shift+S**, save it under `screenshots\` with the file name given in the placeholder, and replace the placeholder line with `![description](screenshots/<file>.png)`.

### Future Work

* Run $n = 2^{28}$ to $2^{30}$ with $m = 10$ on a machine with 16–64 GiB of RAM.
* Use several independent accumulators in each kernel to overlap additions and improve in-cache performance.
* Use fused multiply-add (`vfmadd231ps`), keeping in mind that it changes rounding and therefore the comparison with C.
* Measure with fewer background processes and at a fixed CPU frequency to reduce timing variation.
* Compare compiler options such as `/arch:AVX2` and `/fp:fast` for the C baseline, which allow the compiler to vectorize the loop.
