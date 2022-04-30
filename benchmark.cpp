#include <cstdio>
#include <vector>
#include <complex>
#include <fftw3.h>
#include <benchmark/benchmark.h>
#include "fft.h"
#include "fft_cuda.h"

const int N = 1 << 20;

static void bm_fftw3(benchmark::State& state) {
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

static void bm_fft_single_thread(benchmark::State& state) {
    srand(time(NULL));
    std::complex<double> *input = (std::complex<double>*) calloc(N, sizeof(std::complex<double>));
    std::complex<double> *out = (std::complex<double>*) calloc(N, sizeof(std::complex<double>));
    for (len_t i = 0; i < N; i++) {
        double d1 = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);
        double d2 = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);;
        input[i] = std::complex<double>(d1, d2);
    }
    fft_plan plan = fft_plan_dft_1d(N, input, out, false, 1, false);

    while (state.KeepRunning()) {
        fft_execute(plan);
        benchmark::DoNotOptimize(out); 
    }
    fft_destroy_plan(plan);
    free(input);
    free(out);
}

static void bm_fft_multithread(benchmark::State& state) {
    srand(time(NULL));
    std::complex<double> *input = (std::complex<double>*) calloc(N, sizeof(std::complex<double>));
    std::complex<double> *out = (std::complex<double>*) calloc(N, sizeof(std::complex<double>));
    for (len_t i = 0; i < N; i++) {
        double d1 = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);
        double d2 = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);;
        input[i] = std::complex<double>(d1, d2);
    }
    fft_plan plan = fft_plan_dft_1d(N, input, out, false, 8, false);

    while (state.KeepRunning()) {
        fft_execute(plan);
        benchmark::DoNotOptimize(out); 
    }
    fft_destroy_plan(plan);
    free(input);
    free(out);
}

static void bm_fft_simd(benchmark::State& state) {
    srand(time(NULL));
    std::complex<double> *input = (std::complex<double>*) calloc(N, sizeof(std::complex<double>));
    std::complex<double> *out = (std::complex<double>*) calloc(N, sizeof(std::complex<double>));
    for (len_t i = 0; i < N; i++) {
        double d1 = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);
        double d2 = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);;
        input[i] = std::complex<double>(d1, d2);
    }
    fft_plan plan = fft_plan_dft_1d(N, input, out, false, 1, true);

    while (state.KeepRunning()) {
        fft_execute(plan);
        benchmark::DoNotOptimize(out); 
    }
    fft_destroy_plan(plan);
    free(input);
    free(out);
}

static void bm_fft_simd_multithread(benchmark::State& state) {
    srand(time(NULL));
    std::complex<double> *input = (std::complex<double>*) calloc(N, sizeof(std::complex<double>));
    std::complex<double> *out = (std::complex<double>*) calloc(N, sizeof(std::complex<double>));
    for (len_t i = 0; i < N; i++) {
        double d1 = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);
        double d2 = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);;
        input[i] = std::complex<double>(d1, d2);
    }
    fft_plan plan = fft_plan_dft_1d(N, input, out, false, 8, true);

    while (state.KeepRunning()) {
        fft_execute(plan);
        benchmark::DoNotOptimize(out); 
    }
    fft_destroy_plan(plan);
    free(input);
    free(out);
}

static void bm_fft_cuda(benchmark::State& state) {
    srand(time(NULL));
    int n = N;
    int batch = 1;
    cpxcuda *in = (cpxcuda *) calloc(n * batch, sizeof(cpxcuda));
    cpxcuda *out = (cpxcuda *) calloc(n * batch, sizeof(cpxcuda));
    
    for (int i = 0; i < n * batch; i += n)
        for (int j = 0; j < n; j++) {
        double d1 = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);
        double d2 = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);;
        in[i + j].re = d1;
        in[i + j].im = d2;
    }
    
    fft_plan_cuda plan = fft_plan_cuda_1d(n, batch, in, out, false);

    while (state.KeepRunning()) {
        fft_execute_plan_cuda(plan);
        benchmark::DoNotOptimize(out); 
    }
    fft_destroy_plan_cuda(plan);
    free(in);
    free(out);
}

int main(int argc, char** argv) {
    benchmark::RegisterBenchmark("fftw3", bm_fftw3);
    benchmark::RegisterBenchmark("fft_single_thread", bm_fft_single_thread);
    benchmark::RegisterBenchmark("fft_multithread", bm_fft_multithread);
    benchmark::RegisterBenchmark("fft_simd", bm_fft_simd);
    benchmark::RegisterBenchmark("fft_simd_multithread", bm_fft_simd_multithread);
    benchmark::RegisterBenchmark("fft_cuda", bm_fft_cuda);
 
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}