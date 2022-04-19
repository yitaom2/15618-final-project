#include <cstdio>
#include <vector>
#include <complex>
#include <fftw3.h>
#include <benchmark/benchmark.h>

static void bm_fftw3(benchmark::State& state) {
    int N = 1000000;
    fftw_complex *in, *out;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    while (state.KeepRunning()) {
        fftw_execute(plan);
        benchmark::DoNotOptimize(out);
    }
    
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}


int main(int argc, char** argv) {
    benchmark::RegisterBenchmark("fftw3", bm_fftw3);
 
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}