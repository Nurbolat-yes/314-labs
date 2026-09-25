import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class CpuSaturation {

    static final int WORK_PER_THREAD = 500_000_000;

    public static void main(String[] args) {

        int threads = 8;

        System.out.println("CPU Saturation Test");
        System.out.println("Threads: " + threads);
        System.out.println("Work per thread: " + WORK_PER_THREAD + " square roots");

        ForkJoinPool pool = new ForkJoinPool(threads);

        long start = System.nanoTime();

        try {
            double result = pool.submit(() ->
                IntStream.range(0, threads)
                    .parallel()
                    .mapToDouble(tid -> {
                        double sum = 0.0;

                        for (int i = 1; i <= WORK_PER_THREAD; i++) {
                            sum += Math.sqrt(i + tid);
                        }

                        return sum;
                    })
                    .sum()
            ).join();

            long end = System.nanoTime();

            double seconds = (end - start) / 1_000_000_000.0;

            System.out.printf("Execution time: %.4f seconds%n", seconds);
            System.out.printf("Result checksum: %.4f%n", result);

        } finally {
            pool.shutdown();
        }
    }
}