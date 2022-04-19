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

void raw_fft_itr(complex<double>* input, complex<double>* output, complex<double>* tmp, len_t n, bool reverse) {
    for (len_t i = 0; i < n; i++) {
        output[i] = input[i];
    }
    for (len_t i = 0; i < log2(n); i++) {
        len_t block_size = (n >> i);
        #pragma omp parallel for num_threads(1)
        for (len_t block_idx = 0; block_idx < (1 << i); block_idx++) {
            len_t startidx = block_idx * block_size;
            for (int j = 0; j < block_size; j++) {
                if (j % 2 == 0) tmp[startidx + j/2] = output[startidx + j];
                else tmp[startidx + block_size/2 + (j-1)/2] = output[startidx + j];
            }
        }
        for (int j = 0; j < n; j++) output[j] = tmp[j];
    }
    for (len_t i = 1; i <= log2(n); i++) {
        len_t block_size = (1 << i);
        complex<double> w = e_imaginary(double(2 * M_PI) / double(block_size));
        if (reverse) {
            w = e_imaginary(-double(2 * M_PI) / double(block_size));
        }
        #pragma omp parallel for num_threads(1)
        for (len_t block_idx = 0; block_idx < (n >> i); block_idx++) {
            len_t startidx = block_idx * block_size;
            for (int j = 0; j < block_size/2; j++) {
                tmp[startidx + j] = output[startidx + j] + pow(w, j) * output[startidx + block_size/2 + j];
                tmp[startidx + j + block_size/2] = output[startidx + j] - pow(w, j) * output[startidx + block_size/2 + j];
            }
        }
        for (int j = 0; j < n; j++) output[j] = tmp[j];
    }
    if (reverse) {
        for (len_t i = 0; i < n; i++) {
            output[i] *= complex<double>(1/(double)n, 0);
        }
    }
}

fft_plan fft_plan_dft_1d(len_t n, std::complex<double> *in, std::complex<double> *out, bool reverse) {
    len_t upper_n = power_of_two(n);
    complex<double> *in_concate = (complex<double>*) calloc(upper_n, sizeof(complex<double>));
    complex<double> *out_concate = (complex<double>*) calloc(upper_n, sizeof(complex<double>));
    complex<double> *tmp = (complex<double>*) calloc(upper_n, sizeof(complex<double>));
    for (len_t i = 0; i < n; i++)
        in_concate[i] = in[i];
    return fft_plan{n, upper_n, in, out, tmp, in_concate, out_concate, reverse};
}

void fft_execute(fft_plan &plan) {
    raw_fft_itr(plan.in_concate, plan.out_concate, plan.tmp, plan.upper_n, plan.reverse);
    for (len_t i = 0; i < plan.n; i++)
        plan.out[i] = plan.out_concate[i];
}

void fft_destroy_plan(fft_plan &plan) {
    free(plan.tmp);
    free(plan.in_concate);
    free(plan.out_concate);
}