#ifndef FFT_CUDA_H
#define FFT_CUDA_H
#endif

#include <complex>


struct fft_plan_cuda {
    
};

fft_plan_cuda fft_plan_cuda_1d();

void fft_execute_plan_cuda(fft_plan_cuda &plan);

void fft_destroy_plan_cuda(fft_plan_cuda &plan);