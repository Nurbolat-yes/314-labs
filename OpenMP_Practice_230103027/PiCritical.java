public class PiCritical {

    static final long N = 1_000_000L;
    static final double STEP = 1.0 / N;

    static double runCritical(int threads) throws InterruptedException {

        final Object lock = new Object();
        double[] sharedSum = new double[1];
        Thread[] pool = new Thread[threads];

        long chunkSize = N / threads;

        for (int t = 0; t < threads; t++) {

            final long start = t * chunkSize;
            final long end =
                    (t == threads - 1) ? N : start + chunkSize;

            pool[t] = new Thread(() -> {

                for (long i = start; i < end; i++) {

                    double x = (i + 0.5) * STEP;
                    double term = 4.0 / (1.0 + x * x);

                    synchronized (lock) {
                        sharedSum[0] += term;
                    }
                }
            });

            pool[t].start();
        }

        for (Thread thread : pool) {
            thread.join();
        }

        return sharedSum[0] * STEP;
    }

    public static void main(String[] args) throws Exception {

        int[] threadCounts = {1, 2, 4, 8};

        System.out.println("Critical Section Benchmark");
        System.out.println("N = " + N);
        System.out.println();

        for (int p : threadCounts) {

            long start = System.nanoTime();
            double pi = runCritical(p);
            long end = System.nanoTime();

            double time = (end - start) / 1_000_000_000.0;
            double error = Math.abs(pi - Math.PI);

            System.out.printf(
                "P=%d | Pi=%.12f | Error=%.12e | Time=%.4f s%n",
                p, pi, error, time
            );
        }
    }
}