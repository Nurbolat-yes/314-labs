#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#define N 13027000ULL
#define MAX_THREADS 64

static inline uint32_t collatz_steps(uint64_t n) {
    uint32_t steps = 0;

    while (n > 1) {
        if ((n & 1ULL) == 0)
            n >>= 1;
        else
            n = 3 * n + 1;

        steps++;
    }

    return steps;
}

int main(void) {

    int threads = 4;   /* physical core count */
    omp_set_num_threads(threads);

    /* -------- Variant 1: False Sharing -------- */

    int hit_count[MAX_THREADS] = {0};

    double start1 = omp_get_wtime();

    #pragma omp parallel for
    for (uint64_t i = 1; i <= N; i++) {

        uint32_t steps = collatz_steps(i);

        if (steps > 100) {
            int tid = omp_get_thread_num();
            hit_count[tid]++;
        }
    }

    double end1 = omp_get_wtime();

    long long total1 = 0;

    for (int i = 0; i < threads; i++)
        total1 += hit_count[i];

    /* -------- Variant 2: Reduction -------- */

    long long total2 = 0;

    double start2 = omp_get_wtime();

    #pragma omp parallel for reduction(+:total2)
    for (uint64_t i = 1; i <= N; i++) {

        uint32_t steps = collatz_steps(i);

        if (steps > 100)
            total2++;
    }

    double end2 = omp_get_wtime();

    double time1 = end1 - start1;
    double time2 = end2 - start2;

    printf("Threads = %d\n\n", threads);

    printf("Variant 1 - Naive hit_count[tid]++\n");
    printf("Hits = %lld\n", total1);
    printf("Execution time = %.6f seconds\n", time1);
    printf("Throughput = %.2f iter/sec\n\n", N / time1);

    printf("Variant 2 - OpenMP Reduction\n");
    printf("Hits = %lld\n", total2);
    printf("Execution time = %.6f seconds\n", time2);
    printf("Throughput = %.2f iter/sec\n\n", N / time2);

    printf("Speedup Penalty Ratio = %.4f\n", time1 / time2);

    return 0;
}