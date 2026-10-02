import time
import platform
import numpy as np
import numba
import matplotlib.pyplot as plt
from numba import njit, prange

print("=" * 72)
print("OpenMP / Numba Multi-Core Scaling Lab")
print("=" * 72)
print("CPU / Platform:", platform.processor() or platform.platform())
print("Logical hardware threads detected by Numba:", numba.config.NUMBA_NUM_THREADS)
print("Numba threading layer will be initialized after first parallel call.")
print()

# ----------------------------
# Challenge 1: Monte Carlo Pi
# ----------------------------
@njit(parallel=True)
def monte_carlo_pi(n_samples):
    inside_circle = 0
    for i in prange(n_samples):
        x = np.random.uniform(0.0, 1.0)
        y = np.random.uniform(0.0, 1.0)
        if x * x + y * y <= 1.0:
            inside_circle += 1
    return (4.0 * inside_circle) / n_samples

_ = monte_carlo_pi(10_000)

print("Threading layer:", numba.threading_layer())
print("\nCHALLENGE 1 — MONTE CARLO")
SAMPLES = 120_000_000
thread_counts = [1, 2, 4, 8, numba.config.NUMBA_NUM_THREADS]
thread_counts = sorted(set(t for t in thread_counts if t <= numba.config.NUMBA_NUM_THREADS))

print(f"{'Threads':<10} | {'Time (s)':<12} | {'Speedup':<10} | {'Efficiency (%)':<15} | {'Pi':<12}")
print("-" * 80)

t1_baseline = None
challenge1 = []

for t in thread_counts:
    numba.set_num_threads(t)
    start = time.perf_counter()
    pi_est = monte_carlo_pi(SAMPLES)
    elapsed = time.perf_counter() - start

    if t == 1:
        t1_baseline = elapsed
        speedup = 1.0
        efficiency = 100.0
    else:
        speedup = t1_baseline / elapsed
        efficiency = (speedup / t) * 100.0

    challenge1.append((t, elapsed, speedup, efficiency, pi_est))
    print(f"{t:<10} | {elapsed:<12.4f} | {speedup:<10.2f}x | {efficiency:<15.1f} | {pi_est:<12.8f}")

# Reset to max threads for Challenges 2 and 3
numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

# ----------------------------
# Challenge 2: Mandelbrot
# ----------------------------
@njit(parallel=True)
def render_mandelbrot_rows(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):
        cy = -1.2 + (r / h) * 2.4
        for c in range(w):
            cx = -2.0 + (c / w) * 2.5
            z_real, z_imag = 0.0, 0.0
            it = 0
            while (z_real * z_real + z_imag * z_imag <= 4.0) and (it < max_iter):
                next_real = z_real * z_real - z_imag * z_imag + cx
                z_imag = 2.0 * z_real * z_imag + cy
                z_real = next_real
                it += 1
            img[r, c] = it
    return img

@njit(parallel=True)
def render_mandelbrot_cols(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for c in prange(w):
        cx = -2.0 + (c / w) * 2.5
        for r in range(h):
            cy = -1.2 + (r / h) * 2.4
            z_real, z_imag = 0.0, 0.0
            it = 0
            while (z_real * z_real + z_imag * z_imag <= 4.0) and (it < max_iter):
                next_real = z_real * z_real - z_imag * z_imag + cx
                z_imag = 2.0 * z_real * z_imag + cy
                z_real = next_real
                it += 1
            img[r, c] = it
    return img

_ = render_mandelbrot_rows(100, 100, 50)
_ = render_mandelbrot_cols(100, 100, 50)

H, W, MAX_IT = 2500, 2500, 1000

print("\nCHALLENGE 2 — MANDELBROT")
t0 = time.perf_counter()
grid_rows = render_mandelbrot_rows(H, W, MAX_IT)
t_rows = time.perf_counter() - t0

t1 = time.perf_counter()
grid_cols = render_mandelbrot_cols(H, W, MAX_IT)
t_cols = time.perf_counter() - t1

print(f"Row-Parallel Render Time:    {t_rows:.4f} s")
print(f"Column-Parallel Render Time: {t_cols:.4f} s")

plt.figure(figsize=(8, 8))
plt.imshow(grid_rows, cmap='magma', extent=[-2.0, 0.5, -1.2, 1.2])
plt.title(f"Mandelbrot {H}x{W} (Render: {t_rows:.2f}s)")
plt.axis('off')
plt.savefig('mandelbrot_output.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved image: mandelbrot_output.png")

# ----------------------------
# Challenge 3: Heat Stencil
# ----------------------------
@njit(parallel=True)
def heat_step(u, u_next, alpha=0.20):
    rows, cols = u.shape
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            u_next[i, j] = u[i, j] + alpha * (
                u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4.0 * u[i, j]
            )

def run_heat(dtype):
    GRID_SIZE = 1500
    STEPS = 300
    u = np.zeros((GRID_SIZE, GRID_SIZE), dtype=dtype)
    u_next = np.zeros_like(u)
    u[0, :] = 100.0
    u[:, 0] = 100.0
    u_next[0, :] = 100.0
    u_next[:, 0] = 100.0

    # Warmup for this dtype specialization
    heat_step(u, u_next)

    start = time.perf_counter()
    for step in range(STEPS):
        heat_step(u, u_next)
        u, u_next = u_next, u
    elapsed = time.perf_counter() - start
    cells_per_sec = (GRID_SIZE * GRID_SIZE * STEPS) / elapsed / 1e6
    return elapsed, cells_per_sec

print("\nCHALLENGE 3 — HEAT STENCIL")
t64, mc64 = run_heat(np.float64)
print(f"float64 runtime: {t64:.4f} s")
print(f"float64 throughput: {mc64:.2f} Megacells/sec")

t32, mc32 = run_heat(np.float32)
print(f"float32 runtime: {t32:.4f} s")
print(f"float32 throughput: {mc32:.2f} Megacells/sec")
print(f"Runtime decrease factor (float64 / float32): {t64/t32:.3f}x")

# ----------------------------
# Summary
# ----------------------------
print("\n" + "=" * 72)
print("FINAL SCORECARD VALUES")
print("=" * 72)
for t, elapsed, speedup, efficiency, pi_est in challenge1:
    print(f"Monte Carlo | {t} thread(s) | {elapsed:.4f} s | {speedup:.2f}x | {efficiency:.1f}%")
print(f"Mandelbrot Rows | max threads | {t_rows:.4f} s")
print(f"Mandelbrot Cols | max threads | {t_cols:.4f} s")
print(f"Heat Stencil float64 | max threads | {t64:.4f} s | {mc64:.2f} Megacells/sec")
print(f"Heat Stencil float32 | max threads | {t32:.4f} s | {mc32:.2f} Megacells/sec | decrease factor {t64/t32:.3f}x")
print("=" * 72)
