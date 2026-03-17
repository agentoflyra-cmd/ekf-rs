# Matmul Optimization Notes

## Target Machine

Source: local `lscpu` output on the current development machine.

- CPU: 13th Gen Intel(R) Core(TM) i9-13900HX
- Architecture: `x86_64`
- Logical CPUs: `32`
- Threads per core: `2`
- Caches reported by `lscpu`:
  - L1d: `896 KiB (24 instances)`
  - L1i: `1.3 MiB (24 instances)`
  - L2: `32 MiB (12 instances)`
  - L3: `36 MiB (1 instance)`

## Cache-Fit Heuristic

For square matmul with `m = k = n`, the three active matrices are:

- `lhs`: `m * k`
- `rhs`: `k * n`
- `out`: `m * n`

For `f32`, each element is `4` bytes, so the total working-set size is:

```text
4 * (m*n + n*k + m*k) = 12m^2 bytes
```

### L1-based estimate

`lscpu` reports `896 KiB / 24 ~= 37.3 KiB` L1d per instance, but for planning we use the more conservative `32 KiB` per worker.

If only about `50%` of L1 can be treated as usable hot data for the kernel:

```text
usable_l1 = 32 KiB / 2 = 16 KiB = 16384 bytes
12m^2 <= 16384
m^2 <= 1365.33
m <= 36.95
```

Conservative conclusion:

- `m <= 36` is the safe L1-fit region
- `m = 37` is the upper edge and may still work depending on surrounding pressure

For reference, if the full `32 KiB` could be used:

```text
12m^2 <= 32768
m <= 52.25
```

That is too optimistic for real code once loop state, stack, instruction footprint, and other data compete for cache.

### L2 note

`lscpu` reports `32 MiB (12 instances)` for L2, so L2 is not a simple per-logical-core private cache number. For this machine, L2 should be treated as the next blocking layer, not as the first target.

The practical implication is:

- first optimize for L1-resident micro-kernels
- then use L2-aware tiling only if EKF workloads actually reach those sizes

## Implication For EKF-Specific Matmul

EKF workloads usually do not look like large dense GEMM. In practice, they are often dominated by small matrices such as:

- state covariance `P`
- process noise `Q`
- measurement noise `R`
- Jacobians `F`, `H`
- innovation covariance and gain-related products

That means optimization priorities should be:

1. Optimize very small matrices first, not large-matrix peak FLOPS.
2. Prefer kernels that are cheap on setup overhead.
3. Consider fixed-size or shape-specialized paths for common EKF dimensions.
4. Benchmark the actual matrix sizes used by this project before adding more complex blocking.

## Immediate Guidance

- For `m <= 36`, a carefully written single-thread small-matrix kernel should have a good chance to stay in the L1-friendly regime.
- For EKF, the real win is likely from specialization for small shapes rather than from copying large BLAS-style blocking strategies directly.
- BLAS comparison is still useful as an upper bound and as a regression reference, but it should not be assumed that generic BLAS is optimal for tiny EKF matrices.

## EKF-Specific Guidance

For this codebase, the most important matrix optimization is not SIMD first. It is choosing the right algebra:

1. Prefer `solve()` over explicit `inverse()` in the EKF update path.
2. Prefer `A * B^T` specialized kernels for Jacobian-related products such as:
   - `F P F^T`
   - `H P H^T`
   - `P H^T`
3. Treat covariance-like matrices as a separate category for future SPD-specific solvers.

This means the practical optimization order should be:

1. remove unnecessary inverse formation
2. use `matmul_transposed_rhs()` where the math naturally has a right transpose
3. add SPD solves for innovation covariance
4. only then evaluate whether hand-written SIMD is still worth it

For production EKF workloads, this sequence usually gives more benefit than micro-optimizing generic dense GEMM first.

## Follow-up Benchmark Plan

When extending `ekf-matmul`, benchmark at least these groups separately:

- tiny: `2..=8`
- small: `9..=16`
- medium-small: `17..=36`
- beyond L1 heuristic: `48`, `64`, `96`

And compare:

- current naive kernel
- transposed-`rhs` kernel
- manually blocked kernel
- BLAS baseline

The EKF path should ultimately choose based on actual shape distribution, not only on theoretical cache limits.
