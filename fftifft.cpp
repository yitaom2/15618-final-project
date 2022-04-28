#include <cstdio>
#include <cstdlib>
#include <complex>
#include <cstring>
#include "fft.h"
using namespace std;

#define DEBUG 1

int main(int argc, char **argv) {
    int n;
    complex<double> *in = NULL;
    complex<double> *out = NULL;
    if (DEBUG) {
        n = 1 << 19;
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
    int num_threads = 1;
    fft_plan plan = fft_plan_dft_1d(n, in, out, false, num_threads);
    for(int i = 0; i < 10; i++) fft_execute(plan);
    fft_destroy_plan(plan);
    
    plan = fft_plan_dft_1d(n, out, in, true, num_threads);
    for(int i = 0; i < 10; i++) fft_execute(plan);
    fft_destroy_plan(plan);
    
    // for (int i = 0; i < n; i++) printf("%d", (int)(in[i].real() + 0.5));
    puts("");
    free(in);
    free(out);
    return 0;
}