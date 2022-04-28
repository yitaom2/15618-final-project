#define VEC_SHIFT 3
#define VEC_SIZE (1 << VEC_SHIFT)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "fft.h"
using namespace std;

len_t power_of_two(len_t n) {
    int ct = 0;
    len_t cp_n = n;
    while (cp_n != 1) {
        cp_n >>= 1;
        ct += 1;
    }
    if (n != 1 << ct) {
        return 1 << (ct + 1);
    } else {
        return n;
    }
}

complex<double> e_imaginary(double x) {
    complex<double> real_part = cos(x);
    complex<double> comp_part = complex<double>(0, 1) * complex<double>(sin(x), 0);
    return real_part + comp_part;
}

complex_simd operator+ (const complex_simd &a, const complex_simd &b) {
    complex_simd result;
    result.re = a.re + b.re;
    result.im = a.im + b.im;
    return result;
}

complex_simd operator- (const complex_simd &a, const complex_simd &b) {
    complex_simd result;
    result.re = a.re - b.re;
    result.im = a.im - b.im;
    return result;
}

complex_simd operator* (const complex_simd &a, const complex_simd &b) {
    complex_simd result;
    result.re = a.re * b.re - a.im * b.im;
    result.im = a.re * b.im + a.im * b.re;
    return result;
}

void load_to_simd_vector(complex<double> *src, complex_simd *dst, int n) {
    float *real = (float *) aligned_alloc(256, VEC_SIZE * sizeof(float));
    float *imag = (float *) aligned_alloc(256, VEC_SIZE * sizeof(float));
    for (len_t i = 0; i < n / VEC_SIZE; i++) {
        for (int j = 0; j < VEC_SIZE; j++) {
            real[j] = src[i * VEC_SIZE + j].real();
            imag[j] = src[i * VEC_SIZE + j].imag();
        }
        dst[i].re = _mm256_load_ps(real);
        dst[i].im = _mm256_load_ps(imag);
    }
    free(real);
    free(imag);
}

void store_from_simd_vector(complex_simd *src, complex<double> *dst, int n) {
    float *real = (float *) aligned_alloc(256, VEC_SIZE * sizeof(float));
    float *imag = (float *) aligned_alloc(256, VEC_SIZE * sizeof(float));
    for (len_t i = 0; i < n / VEC_SIZE; i++) {
        _mm256_store_ps(real, src[i].re);
        _mm256_store_ps(imag, src[i].im);
        for (int j = 0; j < VEC_SIZE; j++) {
            dst[VEC_SIZE * i + j] = complex<double>(real[j], imag[j]);
        }
    }
    free(real);
    free(imag);
}

void fft_SIMD(complex<double>* output, complex<double>* ws, len_t n, bool reverse, int num_threads) {
    len_t shift = 0, p = 0, block_size = 2;
    for (len_t x = n; x > 1; x >>= 1, shift++);

    for (; block_size <= VEC_SIZE; block_size *= 2, p++) {
        len_t half_block_size = block_size >> 1;
        shift -= 1;
        #pragma omp parallel for schedule(static),num_threads(num_threads)
        for (len_t i = 0; i < n / 2; i++) {
            len_t j = i & (half_block_size - 1);
            len_t index = ((i >> p) << (p + 1)) + j;
            complex<double> x = output[index];
            complex<double> y = output[index + half_block_size] * ws[j << shift];
            output[index] = x + y;
            output[index + half_block_size] = x - y;
        }
    }

    float *real = (float *) aligned_alloc(256, VEC_SIZE * sizeof(float));
    float *imag = (float *) aligned_alloc(256, VEC_SIZE * sizeof(float));
    complex_simd *out_simd = (complex_simd *) aligned_alloc(256, n / VEC_SIZE * sizeof(complex_simd));
    complex_simd *ws_simd = (complex_simd *) aligned_alloc(256, n / 2 / VEC_SIZE * sizeof(complex_simd));

    load_to_simd_vector(output, out_simd, n);
    
    for (; block_size <= n; block_size <<= 1, p++) {
        shift -= 1;
        len_t half_block_size = block_size >> 1;
        len_t half_block_size_simd = half_block_size >> VEC_SHIFT;
        len_t half_block_mask = half_block_size - 1;
        
        for (len_t i = 0; i < half_block_size >> VEC_SHIFT; i++) {
            for (len_t _ = 0, j = i * VEC_SIZE; _ < VEC_SIZE; _++, j++) {
                len_t index = (j & half_block_mask) << shift;
                real[_] = ws[index].real();
                imag[_] = ws[index].imag();
            }
            ws_simd[i].re = _mm256_load_ps(real);
            ws_simd[i].im = _mm256_load_ps(imag);
        }

        #pragma omp parallel for schedule(static),num_threads(num_threads)
        for (len_t i = 0; i < n / 2; i += VEC_SIZE) {
            len_t j = i & half_block_mask;
            len_t index = (((i >> p) << (p + 1)) + j) >> VEC_SHIFT;
            complex_simd x = out_simd[index];
            complex_simd y = out_simd[index + half_block_size_simd] * ws_simd[j >> VEC_SHIFT];
            out_simd[index] = x + y;
            out_simd[index + half_block_size_simd] = x - y;
        }
    }

    store_from_simd_vector(out_simd, output, n);

    if (reverse) {
        for (len_t i = 0; i < n; i++) {
            output[i] *= complex<double>(1/(double)n, 0);
        }
    }

    free(real);
    free(imag);
    free(ws_simd);
    free(out_simd);
}

void raw_fft_itr(complex<double>* output, complex<double>* ws, len_t n, bool reverse, int num_threads) {
    int shift = 0;
    for (int x = n; x > 1; x >>= 1, shift++);
    for (len_t block_size = 2, p = 0; block_size <= n; block_size *= 2, p++) {
        len_t half_block_size = block_size >> 1;
        shift -= 1;
        #pragma omp parallel for schedule(static),num_threads(num_threads)
        for (len_t i = 0; i < n / 2; i++) {
            len_t j = i & (half_block_size - 1);
            len_t index = ((i >> p) << (p + 1)) + j;
            complex<double> x = output[index];
            complex<double> y = output[index + half_block_size] * ws[j << shift];
            output[index] = x + y;
            output[index + half_block_size] = x - y;
        }
    }
    if (reverse) {
        for (len_t i = 0; i < n; i++) {
            output[i] *= complex<double>(1/(double)n, 0);
        }
    }
}

fft_plan fft_plan_dft_1d(len_t n, std::complex<double> *in, std::complex<double> *out, bool reverse, int num_threads = 1, bool SIMD = false) {
    len_t upper_n = power_of_two(n);
    complex<double> *out_pad = (complex<double>*) calloc(upper_n, sizeof(complex<double>));
    complex<double> *ws = (complex<double>*) calloc(upper_n, sizeof(complex<double>));
    int *index_mapping = (int*) calloc(upper_n, sizeof(int));
    // swap
    int bit=0;
    while ((1<<bit) < upper_n) bit++;
    for (int i = 0; i < upper_n - 1; i++) {
        index_mapping[i] = (index_mapping[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    }
    index_mapping[upper_n - 1] = upper_n - 1;
    for (int i = 0; i < upper_n; i++)
        out_pad[index_mapping[i]] = in[i];
    free(index_mapping);
    // precompute unit root
    ws[0] = complex<double>(1, 0);
    ws[1] = e_imaginary(double(2 * M_PI) / double(upper_n) * (reverse ? -1 : 1)); 
    for (int i = 2; i < upper_n / 2; i++)
        ws[i] = ws[i - 1] * ws[1];

    fft_plan plan = fft_plan{n, upper_n, in, out, out_pad, ws, reverse, num_threads, SIMD};

    return plan;
}

void fft_execute(fft_plan &plan) {
    if (plan.SIMD && plan.upper_n > 8)
        fft_SIMD(plan.out_pad, plan.ws, plan.upper_n, plan.reverse, plan.num_threads);
    else 
        raw_fft_itr(plan.out_pad, plan.ws, plan.upper_n, plan.reverse, plan.num_threads);
    for (len_t i = 0; i < plan.n; i++)
        plan.out[i] = plan.out_pad[i];
}

void fft_destroy_plan(fft_plan &plan) {
    free(plan.out_pad);
    free(plan.ws);
}