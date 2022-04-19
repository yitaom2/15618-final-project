#include <cstdio>
#include <cstdlib>
#include <complex>
#include <cstring>
#include "fft.h"
using namespace std;

int main(int argc, char **argv) {
    int n = strlen(argv[1]);
    complex<double> *in = (complex<double>*) calloc(n, sizeof(complex<double>));
    complex<double> *out = (complex<double>*) calloc(n, sizeof(complex<double>));
    for (int i = 0; i < n; i++)
        in[i] = complex<double>(argv[1][i] - '0', 0);

    fft_plan plan = fft_plan_dft_1d(n, in, out, false);
    fft_execute(plan);
    fft_destroy_plan(plan);
    
    plan = fft_plan_dft_1d(n, out, in, true);
    fft_execute(plan);
    fft_destroy_plan(plan);

    for (int i = 0; i < n; i++) printf("%d", (int)(in[i].real() + 0.5));
    puts("");
    free(in);
    free(out);
    return 0;
}