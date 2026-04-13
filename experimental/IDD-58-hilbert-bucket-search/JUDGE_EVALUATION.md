# Blind Judge Evaluation: Hilbert-Bucket Search Improvements

## Evaluation Methodology

Each hypothesis is scored on 5 criteria (1-5 scale, 5=best):
1. **Throughput** — Raw speed improvement over baseline
2. **Correctness** — Recall vs exhaustive baseline (5=100%, 1=<80%)
3. **Scalability** — Performance scaling across resolutions/box sizes
4. **Integration Cost** — Ease of integrating into production pipeline (5=drop-in)
5. **Memory Overhead** — Additional memory required (5=negligible, 1=significant)

## Benchmark Data Summary

### Baseline (existing Rust 3D spatial_bounds_to_buckets)

| Scenario | p=4 | p=6 | p=8 |
|----------|-----|-----|-----|
| narrow (0.4-0.6) | 4.5 us | 228 us | 19.5 ms |
| medium (0.3-0.7) | 23.4 us | 2.0 ms | 149 ms |
| wide (0.2-0.8) | 40.7 us | 7.4 ms | 517 ms |

### Hypothesis A: Rust 4D Serial/Parallel Enumeration

**Serial 4D:**
| Scenario | p=4 | p=6 |
|----------|-----|-----|
| narrow+narrow_t | 34 us | 7.95 ms |
| narrow+full_t | 106 us | 30.1 ms |
| medium+narrow_t | 501 us | 88.0 ms |
| wide_all | 1.21 ms | 479 ms |

**Parallel 4D (rayon):**
| Scenario | p=4 | p=6 |
|----------|-----|-----|
| narrow+narrow_t | 173 us | 6.09 ms |
| narrow+full_t | 297 us | 21.6 ms |
| medium+narrow_t | 619 us | 58.1 ms |
| wide_all | 1.22 ms | 306 ms |

**Speedup (parallel vs serial):** 1.0-1.6x for small workloads (overhead dominates), 
1.3-1.6x for large workloads. The 4D serial at p=4 is only ~8x slower than 3D baseline, 
which is expected since 4D adds an extra dimension of enumeration.

**Key comparison to Python:** Python itertools at p=4 takes ~5ms (LUT) or ~47ms (no LUT).
Rust serial at p=4 takes 34-501 us — a **10-150x speedup over Python** depending on scenario.

### Hypothesis B: Range Clustering

**Clustering overhead (post-processing only):**
| Input size | Time |
|-----------|------|
| 1,296 IDs (p=4 narrow) | 744 ns |
| 3,456 IDs (p=4 narrow+full_t) | 1.62 us |
| 10,000 IDs (p=4 medium) | 3.78 us |
| 104,976 IDs (p=6 narrow) | 33.4 us |
| 1,048,576 IDs (p=6 medium) | 267 us |

**Compression ratios:**
| Scenario | Bucket IDs | Ranges | Ratio |
|----------|-----------|--------|-------|
| narrow_p4 | 1,296 | 446 | 2.9x |
| medium_p4 | 6,000 | 1,294 | 4.6x |
| wide_p4 | 10,368 | 842 | 12.3x |
| narrow_p6 | 104,976 | 11,726 | 9.0x |
| medium_p6 | 589,824 | 15,744 | 37.5x |
| narrow_p8 | 16,777,216 | 14 | 1,198,373x |

**End-to-end (enumerate + cluster):** Negligible overhead vs serial enumeration alone.

### Hypothesis C: Hierarchical Coarse-to-Fine

**Performance:**
| Scenario | Direct fine | Hierarchical | Notes |
|----------|-----------|-------------|-------|
| narrow c4->f6 | 228 us | 2.57 ms | 11x SLOWER |
| medium c4->f6 | 2.0 ms | 12.6 ms | 6.3x SLOWER |
| wide c4->f6 | 7.4 ms | 21.8 ms | 2.9x SLOWER |
| narrow c4->f8 | 19.5 ms | 95.7 ms | 4.9x SLOWER |
| medium c4->f8 | 149 ms | 462 ms | 3.1x SLOWER |
| wide c4->f8 | 517 ms | 793 ms | 1.5x SLOWER |

**Recall: 100%** (always a superset of direct, but with 21-70% extra false-positive buckets)

### Hypothesis D: Sampling-Based Approximation

**Performance (all at p=4):**
| N samples | narrow | medium | wide |
|-----------|--------|--------|------|
| 100 | 2.0 us | 2.1 us | 2.1 us |
| 500 | 9.8 us | 11.4 us | 11.3 us |
| 1,000 | 18.8 us | 22.0 us | 23.3 us |
| 5,000 | 93.5 us | 108 us | 131 us |

**Recall:**
| Scenario | N=100 | N=500 | N=1000 | N=5000 |
|----------|-------|-------|--------|--------|
| narrow_p4 (216 total) | 36.6% | 92.1% | 99.1% | 100% |
| medium_p4 (1000 total) | 9.5% | 40.2% | 61.9% | 99.6% |
| wide_p4 (1728 total) | 5.6% | 24.6% | 43.8% | 94.7% |
| narrow_p6 (5832 total) | 1.7% | 8.2% | 15.8% | 57.4% |
| medium_p6 (32768 total) | 0.3% | 1.5% | 3.0% | 14.2% |

---

## Scoring

### Hypothesis A: Rust 4D Parallel Bucket Enumeration

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Throughput | **4** | 10-150x faster than Python, ~8x slower than 3D (expected). Parallel adds 1.3-1.6x for large workloads. |
| Correctness | **5** | 100% recall — same algorithm, deterministic, parallel exactly matches serial. |
| Scalability | **3** | Same O((range)^4) complexity as Python, just with lower constant. Still intractable at p>=8 for wide bounds. |
| Integration Cost | **5** | Drop-in replacement for Python `query_buckets`. Add PyO3 binding and call from `HilbertIndex`. |
| Memory Overhead | **5** | Same BTreeSet approach, no additional persistent state. |
| **Total** | **22/25** | |

### Hypothesis B: Range Clustering

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Throughput | **5** | Clustering step itself is <1ms even for millions of IDs. Negligible overhead. |
| Correctness | **5** | Lossless transformation — same bucket set, just represented differently. |
| Scalability | **5** | Compression ratio *increases* with resolution: 2.9x at p=4 to 1.2M:1 at p=8. The higher the resolution, the more contiguous the Hilbert mapping. |
| Integration Cost | **3** | Requires Qdrant filter change: switch from MatchAny to multiple Range filters. Need to verify Qdrant supports OR-combined Range filters efficiently. |
| Memory Overhead | **5** | Reduces memory: fewer range objects than individual IDs. |
| **Total** | **23/25** | |

### Hypothesis C: Hierarchical Coarse-to-Fine

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Throughput | **1** | 1.5-11x SLOWER than direct enumeration in all tested scenarios. The sub-bucket fan-out per coarse cell creates more work than it saves through pruning. |
| Correctness | **5** | 100% recall (superset of baseline), but with 21-70% extra false-positive buckets. |
| Scalability | **1** | Gets worse at higher resolution: the fan-out factor (16^3=4096 fine cells per coarse cell at c4->f8) overwhelms any pruning benefit. |
| Integration Cost | **4** | Same interface, returns bucket IDs. But extra buckets increase Qdrant filter load. |
| Memory Overhead | **3** | Generates more buckets than direct approach due to coarse-cell boundary overlap. |
| **Total** | **14/25** | |

### Hypothesis D: Sampling-Based Approximation

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Throughput | **5** | O(N) regardless of bounds volume. 2us for N=100, 23us for N=1000. Fastest approach at any resolution. |
| Correctness | **2** | Poor recall for production use. At p=4: needs N>=5000 for >99% on narrow bounds, impossible for medium+. At p=6: <60% recall even with N=5000. |
| Scalability | **2** | Recall degrades rapidly as total bucket count increases. Fundamentally, N must scale with bucket count for high recall, negating the speed benefit. |
| Integration Cost | **5** | Same interface, returns bucket IDs. Easy to swap in. |
| Memory Overhead | **5** | Minimal — only stores the sampled set. |
| **Total** | **19/25** | |

---

## Final Rankings

| Rank | Hypothesis | Score | Verdict |
|------|-----------|-------|---------|
| 1 | **B: Range Clustering** | 23/25 | **RECOMMENDED for production** — lossless, fast, superb compression at high resolutions. Integration requires Qdrant filter adaptation. |
| 2 | **A: Rust 4D Parallel** | 22/25 | **RECOMMENDED for production** — essential infrastructure improvement. 10-150x faster than Python. Should be the default 4D bucket enumeration path. |
| 3 | **D: Sampling** | 19/25 | **REJECTED for production** — recall too low for correctness guarantees. Useful only as approximate fallback when exact enumeration is intractable (p>=8, wide bounds). |
| 4 | **C: Hierarchical** | 14/25 | **REJECTED** — slower than direct enumeration in all scenarios. The coarse-to-fine approach doesn't amortize the sub-bucket fan-out cost. Theoretical benefit doesn't materialize in practice. |

## Recommended Implementation Plan

### Phase 1 (Immediate): Hypothesis A
- Add `spatial_bounds_to_buckets_4d()` to loci_core PyO3 bindings
- Replace Python `itertools.product` path in `HilbertIndex.query_buckets()`
- Expected impact: 10-150x speedup for all 4D queries

### Phase 2 (Short-term): Hypothesis B
- Add `cluster_into_ranges()` to loci_core PyO3 bindings
- Modify Qdrant filter construction to use Range filters for large bucket sets
- Prototype: if num_ranges < num_ids / 4, use ranges; else use MatchAny
- Expected impact: Enable higher-resolution queries (p=8) that are currently blocked by the 10K bucket limit

### Phase 3 (Exploration): Combined A+B
- Use Rust 4D enumeration (A) feeding into range clustering (B)
- End-to-end pipeline: bounds -> Rust 4D buckets -> range clustering -> Qdrant Range filters
- Expected impact: Full 4D spatial+temporal queries at p=6-8 with manageable filter cardinality
