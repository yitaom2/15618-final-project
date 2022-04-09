#include <complex>
#include <math.h>
using namespace std;

int power_of_two(int n) {
    int ct = 0;
    int cp_n = n;
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

void raw_fft(complex<double>* input, complex<double>* output, int n, bool reverse) {
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
    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) {
            input1[i / 2] = input[i];
        } else {
            input2[(i-1) / 2] = input[i];
        }
    }
    raw_fft(input1, output1, n/2, reverse);
    raw_fft(input2, output2, n/2, reverse);
    for (int i = 0; i < n/2; i++) {
        output[i] = output1[i] + pow(w, i) * output2[i];
        output[i + n/2] = output1[i] - pow(w, i) * output2[i];
    }
}

complex<double>* fft(complex<double>* input, int n, bool reverse, int &upper_n) {
    upper_n = power_of_two(n);
    if (upper_n != n && reverse) {
        printf("reverse shouldn't require concate, recheck prior code\n");
        exit(1);
    }
    complex<double> input_concate[upper_n];
    complex<double>* output_concate = (complex<double>*) calloc(upper_n, sizeof(complex<double>));
    for (int i = 0; i < upper_n; i++) {
        if (i < n) input_concate[i] = input[i];
        else input_concate[i] = complex<double>(0,0);
        output_concate[i] = complex<double>(0,0);
    }
    raw_fft(input_concate, output_concate, upper_n, reverse);
    if (reverse) {
        for (int i = 0; i < upper_n; i++) {
            output_concate[i] *= complex<double>(1/(double)upper_n, 0);
        }
    }
    return output_concate;
}

int main() {
    int n = 3;
    int upper_n = n;
    complex<double> input[3] = {complex<double>(3,0), complex<double>(1,0), complex<double>(2,0)};
    complex<double>* output = fft(input, n, false, upper_n);

    printf("Sampled points from polynomial\n");
    for (int i = 0; i < upper_n; i++) {
        printf("(%f,%f),", output[i].real(), output[i].imag());
    }
    printf("\n");

    int tmp_n;
    complex<double>* input_rev = fft(output, upper_n, true, tmp_n);
    printf("Reversed polynomial variables\n");
    for (int i = 0; i < upper_n; i++) {
        printf("(%f,%f),", input_rev[i].real(), input_rev[i].imag());
    }
    printf("\n");
}

// reference: youtube channels: Reducible; 3Blue1Brown (tiny error in original python implementation)