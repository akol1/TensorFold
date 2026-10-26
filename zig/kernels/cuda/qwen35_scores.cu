// A row's full-vocabulary statistics: log-sum-exp, the argmax with the lower id on a tie, and selected logits.

// stats[2r] log-sum-exp, stats[2r+1] argmax, stats[8+16r..] the labels' logits; labels[17r] their count, then ids.
extern "C" __global__ void __launch_bounds__(256) tf_q35_score_stats(
    const float* __restrict__ logits, int vocab, float* __restrict__ stats, const unsigned* __restrict__ labels) {
    const int t = threadIdx.x;
    const int r = blockIdx.x;
    const unsigned n = labels[r * 17];
    const float* x = logits + (size_t)r * vocab;
    if (t < 16) stats[8 + r * 16 + t] = t < n ? x[labels[r * 17 + 1 + t]] : 0.f;

    __shared__ float maxima[256];
    __shared__ int indices[256];
    __shared__ float sums[256];

    // The maximum and its lowest id: one thread's stride covers the row, then a tree that keeps the lower id on a tie.
    float m = -INFINITY;
    int best = 0;
    for (int i = t; i < vocab; i += 256) {
        if (x[i] > m || (x[i] == m && i < best)) {
            m = x[i];
            best = i;
        }
    }
    maxima[t] = m;
    indices[t] = best;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride) {
            const float b = maxima[t + stride];
            const int j = indices[t + stride];
            if (b > maxima[t] || (b == maxima[t] && j < indices[t])) {
                maxima[t] = b;
                indices[t] = j;
            }
        }
        __syncthreads();
    }

    // The shifted exponential sum, in a fixed tree so the bytes repeat.
    float s = 0.f;
    for (int i = t; i < vocab; i += 256) s += expf(x[i] - maxima[0]);
    sums[t] = s;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride) sums[t] += sums[t + stride];
        __syncthreads();
    }
    if (t == 0) {
        stats[r * 2] = maxima[0] + logf(sums[0]);
        reinterpret_cast<int*>(stats)[r * 2 + 1] = indices[0];
    }
}
