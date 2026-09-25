import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class ThreadSweep {

    public static void main(String[] args) {

        int[] threadCounts = {1, 2, 4, 8, 16, 32, 64};

        System.out.println("Thread Oversubscription Sweep");
        System.out.println("P,Time_ms");

        for (int p : threadCounts) {

            ForkJoinPool pool = new ForkJoinPool(p);

            long start = System.nanoTime();

            try {
                pool.submit(() ->
                    IntStream.range(0, p)
                        .parallel()
                        .forEach(i -> {
                            // Minimal work: force task execution.
                            double value = Math.sqrt(i + 1.0);
                        })
                ).join();
            } finally {
                pool.shutdown();
            }

            long end = System.nanoTime();

            double timeMs = (end - start) / 1_000_000.0;

            System.out.printf("%d,%.4f%n", p, timeMs);
        }
    }
}