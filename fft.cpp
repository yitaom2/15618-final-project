#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "fft.h"
#include "immintrin.h"
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

void raw_fft_itr(complex<double>* output, complex<double>* ws, len_t n, bool reverse, int num_threads) {
    len_t block_size = 2;
    for (; block_size <= (16 <= n ? 16 : n); block_size *= 2) {
        len_t half_block_size = block_size >> 1;
        int step = n / block_size;
        for (len_t i = 0; i < n; i += block_size) {
            for (int j = 0; j < half_block_size; j++) {
                complex<double> x = output[i + j];
                complex<double> y = output[i + j + half_block_size] * ws[j * step];
                output[i + j] = x + y;
                output[i + j + half_block_size] = x - y;
            }
        }
    }
    __m256* output_simd_real = (__m256*) aligned_alloc(256, n / 8 * sizeof(__m256));
    __m256* output_simd_imag = (__m256*) aligned_alloc(256, n / 8 * sizeof(__m256));
    // alignas(256) __m256 output_simd_real[n/8];
    // alignas(256) __m256 output_simd_imag[n/8];
    // __m256* ws_simd_real;
    // __m256* ws_simd_imag;
    for (len_t i = 0; i < n/8; i++) {
        float real[8], imag[8];
        for (int j = 0; j < 8; j++) {
            real[j] = output[i*8+j].real();
            imag[j] = output[i*8+j].imag();
        }
        output_simd_real[i] = _mm256_load_ps(&real[0]);
        output_simd_imag[i] = _mm256_load_ps(&imag[0]);
    }

    __m256 *ws_simd_real = (__m256*) aligned_alloc(256, n / 8 * sizeof(__m256));
    __m256 *ws_simd_imag = (__m256*) aligned_alloc(256, n / 8 * sizeof(__m256));
    for (; block_size <= n; block_size *= 2) {
        len_t half_block_size = block_size >> 1;
        len_t half_block_size_simd = half_block_size >> 3;
        int step = n / block_size;
        
        for (len_t i = 0; i < half_block_size_simd; i++) {
            float real[8], imag[8];
            for (int j = 0; j < 8; j++) {
                real[j] = ws[(8*i+j) * step].real();
                imag[j] = ws[(8*i+j) * step].imag();
            }
            ws_simd_real[i] = _mm256_load_ps(&real[0]);
            ws_simd_imag[i] = _mm256_load_ps(&imag[0]);
        }
        for (len_t i = 0; i < n; i += block_size) {
            int i_simd = i >> 3;
            for (int j = 0; j < half_block_size_simd; j++) {
                __m256 x_real = output_simd_real[i_simd + j];
                __m256 x_imag = output_simd_imag[i_simd + j];
                __m256 y_real = _mm256_sub_ps(_mm256_mul_ps(output_simd_real[i_simd + j + half_block_size_simd], ws_simd_real[j]), _mm256_mul_ps(output_simd_imag[i_simd + j + half_block_size_simd], ws_simd_imag[j]));
                __m256 y_imag = _mm256_add_ps(_mm256_mul_ps(output_simd_real[i_simd + j + half_block_size_simd], ws_simd_imag[j]), _mm256_mul_ps(output_simd_imag[i_simd + j + half_block_size_simd], ws_simd_real[j]));
                output_simd_real[i_simd + j] = _mm256_add_ps(x_real, y_real);
                output_simd_imag[i_simd + j] = _mm256_add_ps(x_imag, y_imag);
                output_simd_real[i_simd + j + half_block_size_simd] = _mm256_sub_ps(x_real, y_real);
                output_simd_imag[i_simd + j + half_block_size_simd] = _mm256_sub_ps(x_imag, y_imag);
            }
        }
    }
    free(ws_simd_real);
    free(ws_simd_imag);

    for (len_t i = 0; i < n/8; i++) {
        float real[8], imag[8];
        _mm256_store_ps(&real[0], output_simd_real[i]);
        _mm256_store_ps(&imag[0], output_simd_imag[i]);
        for (int j = 0; j < 8; j++) {
            output[8*i+j] = complex<double>(real[j], imag[j]);
        }
    }
    free(output_simd_real);
    free(output_simd_imag);

    if (reverse) {
        for (len_t i = 0; i < n; i++) {
            output[i] *= complex<double>(1/(double)n, 0);
        }
    }


    /*for (len_t block_size = 2; block_size <= n; block_size *= 2) {
        len_t half_block_size = block_size >> 1;
        int step = n / block_size;
        // #pragma omp parallel for num_threads(1)
        for (len_t i = 0; i < n; i += block_size) {
            for (int j = 0; j < half_block_size; j++) {
                complex<double> x = output[i + j];
                complex<double> y = output[i + j + half_block_size] * ws[j * step];
                output[i + j] = x + y;
                output[i + j + half_block_size] = x - y;
            }
        }
    }
    if (reverse) {
        for (len_t i = 0; i < n; i++) {
            output[i] *= complex<double>(1/(double)n, 0);
        }
    }*/
}

fft_plan fft_plan_dft_1d(len_t n, std::complex<double> *in, std::complex<double> *out, bool reverse, int num_threads = 1) {
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
    
    return fft_plan{n, upper_n, in, out, out_pad, ws, reverse, num_threads};
}

void fft_execute(fft_plan &plan) {
    raw_fft_itr(plan.out_pad, plan.ws, plan.upper_n, plan.reverse, plan.num_threads);
    for (len_t i = 0; i < plan.n; i++)
        plan.out[i] = plan.out_pad[i];
}

void fft_destroy_plan(fft_plan &plan) {
    free(plan.out_pad);
    free(plan.ws);
}