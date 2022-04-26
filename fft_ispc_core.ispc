struct complex_ispc {
    double re, im;
};

complex_ispc operator+ (const complex_ispc &a, const complex_ispc &b) {
    complex_ispc result;
    result.re = a.re + b.re;
    result.im = a.im + b.im;
    return result;
}

complex_ispc operator- (const complex_ispc &a, const complex_ispc &b) {
    complex_ispc result;
    result.re = a.re - b.re;
    result.im = a.im - b.im;
    return result;
}

complex_ispc operator* (complex_ispc a, complex_ispc b) {
    complex_ispc result;
    result.re = a.re * b.re - a.im * b.im;
    result.im = a.re * b.im + a.im * b.re;
    return result;
}

export void precompute_unit_root(uniform complex_ispc ws[], uniform int n) {
    for (int i = 2; i < n / 2; i++)
        ws[i] = ws[i - 1] * ws[1];
}

export void fft_ispc(uniform complex_ispc output[], uniform complex_ispc ws[], uniform int n, uniform bool reverse) {
    for (uniform int block_size = 2, p = 0; block_size <= n; block_size *= 2, p++) {
        uniform int half_block_size = block_size >> 1;
        uniform int step = n / block_size;
        foreach (i = 0 ... (n >> 1)) {
            int j = i & (half_block_size - 1);
            int index = ((i >> p) << (p + 1)) + j;
            complex_ispc x = output[index];
            complex_ispc y = output[index + half_block_size] * ws[j * step];
            output[index] = x + y;
            output[index + half_block_size] = x - y;
        }
    }
    if (reverse) {
        complex_ispc r;
        r.re = 1 / (double)n;
        r.im = 0;
        for (int i = 0; i < n; i++) {
            output[i] = output[i] * r;
        }
    }
}
