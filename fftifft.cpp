#include <cstdio>
#include <cstdlib>
#include <complex>
#include <cstring>
#include "fft.h"
#include "fft_ispc.h"
#include "CycleTimer.h"
using namespace std;

#define DEBUG 1

int N = 1 << 3;

void bm_serial() {
    int n = N;
    complex<double> *in = (complex<double>*) calloc(n, sizeof(complex<double>));
    complex<double> *out = (complex<double>*) calloc(n, sizeof(complex<double>));
    for (len_t i = 0; i < n; i++) {
        in[i] = complex<double>(i, 0);
    }
    int num_threads = 1;
    fft_plan plan = fft_plan_dft_1d(n, in, out, false, num_threads);
    double startTime = CycleTimer::currentSeconds(); 
    for (int i = 0; i < 1; i++)
        fft_execute(plan);  
    double time = CycleTimer::currentSeconds() - startTime;
    fft_destroy_plan(plan);
    for (int i = 0; i < 4; i++)
        printf("%.3f %.3f, ", out[i].real(), out[i].imag());
    puts("");
    free(in);
    free(out);
    printf("[fft serial]:\t\t[%.3f] ms\n", time * 1000 / 10);
}

void bm_ispc() {
    int n = N;
    complex_ispc *input = (complex_ispc*) calloc(n, sizeof(complex_ispc));
    complex_ispc *out = (complex_ispc*) calloc(n, sizeof(complex_ispc));
    for (len_t i = 0; i < n; i++) {
        input[i].re = i;
        input[i].im = 0;
    }
    fft_plan_ispc plan = fft_plan_ispc_1d(n, input, out, false);

    double startTime = CycleTimer::currentSeconds(); 
    for (int i = 0; i < 1; i++)
        fft_execute_ispc(plan);
    double time = CycleTimer::currentSeconds() - startTime;

    for (int i = 0; i < 4; i++)
        printf("%.3f %.3f, ", out[i].re, out[i].im);
    puts("");

    fft_destroy_plan_ispc(plan);
    free(input);
    free(out);

    printf("[fft ispc]:\t\t[%.3f] ms\n", time * 1000 / 10);
}

int main(int argc, char **argv) {
    int n;
    complex<double> *in = NULL;
    complex<double> *out = NULL;
    if (DEBUG) {
        n = 1 << 20;
        in = (complex<double>*) calloc(n, sizeof(complex<double>));
        out = (complex<double>*) calloc(n, sizeof(complex<double>));

        srand(time(NULL));
        for (len_t i = 0; i < n; i++) {
            double d1 = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);
            double d2 = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);;
            in[i] = complex<double>(d1, d2);
        }
    } else {
        n = strlen(argv[1]);
        in = (complex<double>*) calloc(n, sizeof(complex<double>));
        out = (complex<double>*) calloc(n, sizeof(complex<double>));
        for (int i = 0; i < n; i++)
            in[i] = complex<double>(argv[1][i] - '0', 0);
    }
    srand(time(NULL));
    bm_serial();
    bm_ispc();

    /*
    plan = fft_plan_dft_1d(n, out, in, true, num_threads);
    fft_execute(plan);
    fft_destroy_plan(plan);
    
    //for (int i = 0; i < n; i++) printf("%d", (int)(in[i].real() + 0.5));
    puts("");
    */

    return 0;
}