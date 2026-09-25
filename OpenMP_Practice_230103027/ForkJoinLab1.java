import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class ForkJoinLab1 {

    public static void main(String[] args) {

        int targetThreads = 4;

        System.out.println(
            "Master thread starting. Spawning thread team of size: "
            + targetThreads
        );

        ForkJoinPool customPool = new ForkJoinPool(targetThreads);

        try {
            customPool.submit(() -> {

                IntStream.range(0, targetThreads)
                        .parallel()
                        .forEach(idx -> {

                            long osTid = Thread.currentThread().threadId();
                            String threadName =
                                    Thread.currentThread().getName();

                            boolean isMaster = idx == 0;

                            System.out.printf(
                                "[%s] Logical Rank: %d | Worker Thread: %s (OS ID: %d)%n",
                                isMaster ? "Master" : "Worker",
                                idx,
                                threadName,
                                osTid
                            );
                        });

            }).join();

        } finally {
            customPool.shutdown();
        }

        System.out.println(
            "Parallel region closed. Execution returned to master thread."
        );
    }
}