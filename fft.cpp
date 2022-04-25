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

void raw_fft_itr(complex<double>* output, complex<double>* ws, len_t n, bool reverse, int num_threads) {
    for (len_t block_size = 2, p = 0; block_size <= n; block_size *= 2, p++) {
        len_t half_block_size = block_size >> 1;
        int step = n / block_size;
        #pragma omp parallel for schedule(static),num_threads(num_threads)
        for (len_t i = 0; i < n / 2; i++) {
            len_t j = i & (half_block_size - 1);
            len_t index = ((i >> p) << (p + 1)) + j;
            complex<double> x = output[index];
            complex<double> y = output[index + half_block_size] * ws[j * step];
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