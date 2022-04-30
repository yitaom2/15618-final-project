#include "fft_cuda.h"

int main() {
    fft_plan_cuda plan = fft_plan_cuda_1d();
    fft_execute_plan_cuda(plan);
    fft_destroy_plan_cuda(plan);
    return 0;
}