#include <complex>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

#define DEBUG 0
#define TIME 1

typedef long long int len_t;
const int n = 1 << 20;
complex<double> input[n], tmp[n], output[n];

len_t power_of_two(len_t n) {
    int ct = 0;
    len_t cp_n = n;
    while (cp_n != 1) {
        cp_n >>= 1;
        ct += 1;
    }
    if (n != 1 << ct) {
        return 1 << (ct + 1);
    } else {
        return n;
    }
}

complex<double> e_imaginary(double x) {
    complex<double> real_part = cos(x);
    complex<double> comp_part = complex<double>(0, 1) * complex<double>(sin(x), 0);
    return real_part + comp_part;
}

void raw_fft_itr(complex<double>* input, complex<double>* output, len_t n, bool reverse) {
    for (len_t i = 0; i < n; i++) {
        output[i] = input[i];
    }
    for (len_t i = 0; i < log2(n); i++) {
        len_t block_size = (n >> i);
        #pragma omp parallel for num_threads(1)
        for (len_t block_idx = 0; block_idx < (1 << i); block_idx++) {
            len_t startidx = block_idx * block_size;
            for (int j = 0; j < block_size; j++) {
                if (j % 2 == 0) tmp[startidx + j/2] = output[startidx + j];
                else tmp[startidx + block_size/2 + (j-1)/2] = output[startidx + j];
            }
        }
        for (int j = 0; j < n; j++) output[j] = tmp[j];
    }
    for (len_t i = 1; i <= log2(n); i++) {
        len_t block_size = (1 << i);
        complex<double> w = e_imaginary(double(2 * M_PI) / double(block_size));
        if (reverse) {
            w = e_imaginary(-double(2 * M_PI) / double(block_size));
        }
        #pragma omp parallel for num_threads(1)
        for (len_t block_idx = 0; block_idx < (n >> i); block_idx++) {
            len_t startidx = block_idx * block_size;
            for (int j = 0; j < block_size/2; j++) {
                tmp[startidx + j] = output[startidx + j] + pow(w, j) * output[startidx + block_size/2 + j];
                tmp[startidx + j + block_size/2] = output[startidx + j] - pow(w, j) * output[startidx + block_size/2 + j];
            }
        }
        for (int j = 0; j < n; j++) output[j] = tmp[j];
    }
}

void raw_fft_recur(complex<double>* input, complex<double>* output, len_t n, bool reverse) {
    if (n == 1) {
        output[0] = input[0];
        return;
    }
    complex<double> w = e_imaginary(double(2 * M_PI) / double(n));
    if (reverse) {
        w = e_imaginary(-double(2 * M_PI) / double(n));
    }
    complex<double> input1[n/2]; complex<double> input2[n/2];
    complex<double> output1[n/2]; complex<double> output2[n/2];
    for (len_t i = 0; i < n; i++) {
        if (i % 2 == 0) {
            input1[i / 2] = input[i];
        } else {
            input2[(i-1) / 2] = input[i];
        }
    }
    // #pragma omp parallel for 
    // {
        raw_fft_recur(input1, output1, n/2, reverse);
        raw_fft_recur(input2, output2, n/2, reverse);
    // }
    for (len_t i = 0; i < n/2; i++) {
        output[i] = output1[i] + pow(w, i) * output2[i];
        output[i + n/2] = output1[i] - pow(w, i) * output2[i];
    }
}

complex<double>* fft(complex<double>* input, len_t n, bool reverse, len_t &upper_n) {
    upper_n = power_of_two(n);
    if (upper_n != n && reverse) {
        printf("reverse shouldn't require concate, recheck prior code\n");
        exit(1);
    }
    complex<double> input_concate[upper_n];
    complex<double>* output_concate = (complex<double>*) calloc(upper_n, sizeof(complex<double>));
    for (len_t i = 0; i < upper_n; i++) {
        if (i < n) input_concate[i] = input[i];
        else input_concate[i] = complex<double>(0,0);
        output_concate[i] = complex<double>(0,0);
    }
    raw_fft_itr(input_concate, output_concate, upper_n, reverse);
    if (reverse) {
        for (len_t i = 0; i < upper_n; i++) {
            output_concate[i] *= complex<double>(1/(double)upper_n, 0);
        }
    }
    return output_concate;
}

int main() {
    srand(time(NULL));
    len_t upper_n = n;
    for (len_t i = 0; i < n; i++) {
        double d1 = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);
        double d2 = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);;
        input[i] = complex<double>(d1, d2);
    }

    if (DEBUG) {
        // input[0] = complex<double>(3, 0);
        // input[1] = complex<double>(1, 0);
        // input[2] = complex<double>(2, 0);
        // input[3] = complex<double>(0, 0);
        printf("Generated Input\n");
        for (int i = 0; i < n; i++) {
            printf("(%f,%f),", input[i].real(), input[i].imag());
        }
        printf("\n");
    }

    clock_t start = clock();
    raw_fft_itr(input, output, n, false);

    if (DEBUG) {
        printf("Sampled points from polynomial\n");
        for (int i = 0; i < upper_n; i++) {
            printf("(%f,%f),", output[i].real(), output[i].imag());
        }
        printf("\n");
    }

    raw_fft_itr(output, input, n, true);
    for (len_t i = 0; i < upper_n; i++) {
        input[i] *= complex<double>(1/(double)upper_n, 0);
    }

    clock_t end = clock();
    if (TIME) {
        printf("%f seconds\n", (float)(end - start) / CLOCKS_PER_SEC);
    }

    if (DEBUG) {
        printf("Reversed polynomial variables\n");
        for (int i = 0; i < upper_n; i++) {
            printf("(%f,%f),", input[i].real(), input[i].imag());
        }
        printf("\n");
    }
}

// reference: youtube channels: Reducible; 3Blue1Brown (tiny error in original python implementation)