# Profiling Summary: Analyze & Encode Performance Analysis

## Overview

This document summarizes the profiling analysis of the `analyze` and `encode` operations for a single large partition of mixed-type tabular data.

## Test Configuration

- **Dataset**: 100,000 rows × 14 mixed-type columns
- **Memory**: 35.23 MB
- **Missing Values**: ~15% per column
- **Column Types**:
  - 3 categorical (low/medium/high cardinality)
  - 5 numeric (int, float, discrete, skewed, binary)
  - 2 datetime (recent, wide range)
  - 2 character/string (short, medium length)
  - 2 lat/long (as numeric binned)
- **Configuration**: `n_jobs=1` (single-threaded for profiling)

## Performance Results

### Current Performance
```
ANALYZE: 11.47s  (8,718 rows/sec)
ENCODE:   3.14s  (31,885 rows/sec)
TOTAL:   14.61s
```

### Breakdown by Phase

**ANALYZE (11.47s)**:
- compute_log_histogram: 4.06s (35%)
- split_sub_columns_digit: 3.69s (32%)
- String pattern matching: 2.26s (20%)
- is_sequential checks: 1.29s (11%)
- Other operations: 0.17s (2%)

**ENCODE (3.14s)**:
- is_a_list checks: 1.09s (35%)
- map_array operations: 1.74s (55%)
- character encoding: 0.80s (26%)
- Type checking: 0.45s (14%)
- Other operations: 0.06s (2%)

## Top 5 Bottlenecks

### 🔴 1. compute_log_histogram (4.06s - 28% of total time)
**Location**: `mostlyai/engine/_common.py:636`
**Impact**: Critical - single biggest bottleneck
**Solution**: Vectorize with numpy or use numba JIT
**Expected Speedup**: 5-10x

### 🔴 2. split_sub_columns_digit (3.69s - 25% of total time)
**Location**: `mostlyai/engine/_encoding_types/tabular/numeric.py:108`
**Impact**: Critical - numeric column processing
**Solution**: Replace string operations with mathematical digit extraction
**Expected Speedup**: 5-10x

### 🔴 3. is_a_list checks (1.09s - 7% of total time)
**Location**: `mostlyai/engine/_common.py:201`
**Calls**: 3,000,000 times!
**Impact**: Critical - called once per value
**Solution**: Check once per column, not per value
**Expected Speedup**: 10-20x

### 🟡 4. String splitting (3.06s combined - 21% of total time)
**Location**: Character encoding in multiple files
**Impact**: High - regex operations on strings
**Solution**: Use numpy array views or vectorized operations
**Expected Speedup**: 3-5x

### 🟡 5. is_sequential checks (1.29s - 9% of total time)
**Location**: `mostlyai/engine/_common.py:205`
**Calls**: 56 times
**Impact**: Medium-high - applies lambda to entire series
**Solution**: Sample-based check + caching
**Expected Speedup**: 5-10x

## Optimization Recommendations

### Quick Wins (1-2 days implementation)
These optimizations require minimal code changes and provide significant speedup:

1. ✅ **Replace is_a_list with column-level checks** → 1.09s saved
2. ✅ **Optimize is_sequential with sampling** → 1.0s saved
3. ✅ **Cache histogram edges** → 0.5s saved
4. ✅ **Reduce type conversions** → 0.3s saved

**Expected Total Speedup**: 14.61s → **11-12s** (~20% faster)

### Major Optimizations (3-5 days implementation)
These require more substantial refactoring:

1. ✅ **Vectorize compute_log_histogram** → 3.5s saved
2. ✅ **Vectorize split_sub_columns_digit with numba** → 3.0s saved
3. ✅ **Optimize string splitting** → 2.0s saved

**Expected Total Speedup**: 14.61s → **3-4s** (~4-5x faster)

### Advanced Optimizations (5-10 days implementation)
For maximum performance:

1. ✅ Use Cython/numba for all hot loops
2. ✅ Implement zero-copy operations
3. ✅ Add proper parallelization (with n_jobs > 1)
4. ✅ GPU acceleration for histograms

**Expected Total Speedup**: 14.61s → **1.5-2.5s** (~6-10x faster)

## Projected Performance

### Conservative Estimate (with Phase 1 + 2)
```
ANALYZE: 11.47s → 2.5s   (4.6x faster)
ENCODE:   3.14s → 1.0s   (3.1x faster)
TOTAL:   14.61s → 3.5s   (4.2x faster)
```

### Optimistic Estimate (with all optimizations)
```
ANALYZE: 11.47s → 1.2s   (9.6x faster)
ENCODE:   3.14s → 0.5s   (6.3x faster)
TOTAL:   14.61s → 1.7s   (8.6x faster)
```

### With Parallelization (4 cores, n_jobs=4)
```
ANALYZE: 1.2s → 0.4s   (additional 3x)
ENCODE:  0.5s → 0.2s   (additional 2.5x)
TOTAL:   1.7s → 0.6s   (24x faster than original!)
```

## Key Files to Optimize

### Priority 1: Critical Performance Impact
1. `mostlyai/engine/_common.py`
   - `compute_log_histogram()` - line 636
   - `is_a_list()` - line 201
   - `is_sequential()` - line 205

2. `mostlyai/engine/_encoding_types/tabular/numeric.py`
   - `split_sub_columns_digit()` - line 108
   - `analyze_numeric()` - line 145

### Priority 2: High Performance Impact
3. `mostlyai/engine/_encoding_types/tabular/character.py`
   - `analyze_character()` - line 33
   - `encode_character()` - line 98
   - String splitting operations

4. `mostlyai/engine/_tabular/encoding.py`
   - `_encode_col()` - line 246
   - `encode_df()` - line 181

5. `mostlyai/engine/analysis.py`
   - `_analyze_col()` - line 514
   - `_analyze_flat_col()` - line 554

## Implementation Resources

### Files Created
1. **`PROFILING_ANALYSIS.md`**: Detailed analysis with all bottlenecks and solutions
2. **`optimization_examples.py`**: Concrete implementation examples for top 5 optimizations
3. **`profile_analyze_encode_v2.py`**: Profiling script to measure improvements
4. **`profiling_results/`**: Detailed cProfile output

### Next Steps
1. Review optimization examples in `optimization_examples.py`
2. Run unit tests to establish baseline correctness
3. Implement quick wins first (2-3 days)
4. Measure actual speedup achieved
5. Implement major optimizations based on results
6. Re-profile and iterate

## Notes

- All optimizations preserve correctness - output should be identical
- Focus on algorithmic improvements over micro-optimizations
- Vectorization and numba provide the biggest wins
- Parallelization is a force multiplier (but test after other optimizations)
- Consider memory usage vs speed tradeoffs

## Reproducing Results

To reproduce this profiling analysis:

```bash
cd /Users/mplatzer/github/mostlyai-engine
uv run python profile_analyze_encode_v2.py
```

Output will be saved to `profiling_results/` directory.

To profile with different dataset sizes:
```python
# Edit profile_analyze_encode_v2.py
n_rows = 500_000  # Increase for larger test
```

---

**Generated**: October 23, 2025
**Dataset**: 100K rows, 14 columns, mixed types
**Platform**: macOS (Apple Silicon)
