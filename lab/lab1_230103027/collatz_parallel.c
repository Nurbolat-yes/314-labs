#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>
#include <omp.h>

#define N 13027000ULL
#define MOD 1000000007ULL

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

int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Usage: %s <threads>\n", argv[0]);
        return 1;
    }

    int threads = atoi(argv[1]);
    omp_set_num_threads(threads);

    uint32_t max_steps = 0;
    uint64_t checksum = 0;

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(max:max_steps) reduction(+:checksum)
    for (uint64_t i = 1; i <= N; i++) {
        uint32_t steps = collatz_steps(i);

        if (steps > max_steps)
            max_steps = steps;

        checksum += steps;
    }

    checksum %= MOD;

    double end = omp_get_wtime();

    printf("Threads = %d\n", threads);
    printf("N = %" PRIu64 "\n", (uint64_t)N);
    printf("Max stopping time = %u\n", max_steps);
    printf("Checksum = %" PRIu64 "\n", checksum);
    printf("Execution time = %.6f seconds\n", end - start);

    return 0;
}