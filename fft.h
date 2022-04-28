#ifndef FFT_H
#define FFT_H
#endif

#include <complex>

typedef long long int len_t;

typedef struct fft_plan {
    len_t n, upper_n;
    std::complex<double> *in, *out, *out_pad, *ws;
    bool reverse;
    int num_threads;
} fft_plan;

fft_plan fft_plan_dft_1d(len_t n, std::complex<double> *in, std::complex<double> *out, bool reverse, int num_threads);

void fft_execute(fft_plan &plan);

void fft_destroy_plan(fft_plan &plan);

// TODO: define complex number as structure of two int