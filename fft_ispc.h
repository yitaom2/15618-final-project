#ifndef FFT_ISPC
#define FFT_ISPC
#endif

#include "fft_ispc_core.h"
#include <complex>

struct fft_ispc_plan {
    int n, upper_n;
    complex_ispc *in, *out, *out_pad, *ws;
    bool reverse;
};

fft_plan_ispc fft_plan_ispc_1d(int n, complex_ispc *in, complex_ispc *out, bool reverse);

void fft_execute_ispc(fft_plan_ispc &plan);

void fft_destroy_plan_ispc(fft_plan_ispc &plan);