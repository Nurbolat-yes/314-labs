#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#define N 13027000ULL

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

void run_test(const char *name, omp_sched_t schedule, int chunk) {

    long long total_hits = 0;

    omp_set_schedule(schedule, chunk);

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(runtime) reduction(+:total_hits)
    for (uint64_t i = 1; i <= N; i++) {
        uint32_t steps = collatz_steps(i);

        if (steps > 100)
            total_hits++;
    }

    double end = omp_get_wtime();

    printf("%-25s Time = %.6f s | Hits = %lld\n",
           name, end - start, total_hits);
}

int main(void) {

    omp_set_num_threads(4);

    printf("Threads = 4\n");
    printf("N = %llu\n\n", (unsigned long long)N);

    run_test("schedule(static)",
             omp_sched_static, 0);

    run_test("schedule(static,1000)",
             omp_sched_static, 1000);

    run_test("schedule(dynamic,100)",
             omp_sched_dynamic, 100);

    run_test("schedule(dynamic,10000)",
             omp_sched_dynamic, 10000);

    run_test("schedule(guided)",
             omp_sched_guided, 0);

    return 0;
}