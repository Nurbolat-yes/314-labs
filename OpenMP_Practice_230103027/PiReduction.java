import java.util.stream.LongStream;

public class PiReduction {

    static final long N = 100_000_000L;
    static final double STEP = 1.0 / N;

    static double runReduction(int threads) throws Exception {

        System.setProperty(
                "java.util.concurrent.ForkJoinPool.common.parallelism",
                String.valueOf(threads)
        );

        double sum = LongStream.range(0, N)
                .parallel()
                .mapToDouble(i -> {
                    double x = (i + 0.5) * STEP;
                    return 4.0 / (1.0 + x * x);
                })
                .sum();

        return sum * STEP;
    }

    public static void main(String[] args) throws Exception {

        int threads = Integer.parseInt(args[0]);

        long start = System.nanoTime();

        double calculatedPi = runReduction(threads);

        long end = System.nanoTime();

        double error = Math.abs(calculatedPi - Math.PI);
        double time = (end - start) / 1_000_000_000.0;

        System.out.println("Parallel Reduction - Pi Integration");
        System.out.println("N = " + N);

        System.out.printf(
                "P=%d | Pi=%.12f | Error=%.12e | Time=%.4f s%n",
                threads,
                calculatedPi,
                error,
                time
        );
    }
}