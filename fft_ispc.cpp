#include <math.h>
#include "fft_ispc.h"

complex_ispc e_imaginary(double x) {
    return complex_ispc{cos(x), sin(x)};
}

fft_plan_ispc fft_plan_ispc_1d(int n, complex_ispc *in, complex_ispc *out, bool reverse) {
    int upper_n = 1;
    while (upper_n < n) upper_n <<= 1;

    complex_ispc *out_pad = (complex_ispc *) calloc(upper_n, sizeof(complex_ispc));
    complex_ispc *ws = (complex_ispc *) calloc(upper_n, sizeof(complex_ispc));
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
    ws[0].re = 0;
    ws[1] = e_imaginary(double(2 * M_PI) / double(upper_n) * (reverse ? -1 : 1)); 
    for (int i = 2; i < upper_n / 2; i++)
        ws[i] = ws[i - 1] * ws[1];
    return fft_plan_ispc{n, upper_n, in, out, out_pad, ws, reverse};
}

void fft_execute_ispc(fft_plan_ispc &plan) {
    fft_ispc(plan.out_pad, plan.ws, plan.upper_n, plan.reverse);
    for (len_t i = 0; i < plan.n; i++)
        plan.out[i] = plan.out_pad[i];
}

void fft_destroy_plan_ispc(fft_plan_ispc &plan) {
    free(plan.out_pad);
    free(plan.ws);
}