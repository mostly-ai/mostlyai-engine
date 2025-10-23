# Profiling Analysis & Optimization Recommendations

## Executive Summary

**Dataset**: 100,000 rows × 14 columns (mixed types), 35.23 MB
**ANALYZE**: 11.47s (8,718 rows/sec)
**ENCODE**: 3.14s (31,885 rows/sec)
**TOTAL**: 14.61s

## Key Bottlenecks Identified

### ANALYZE Operation (11.47s total)

#### 1. **compute_log_histogram: 4.06s (35% of time)**
   - **Location**: `_common.py:636`
   - **Calls**: 16 times (once per numeric/datetime column)
   - **Self time**: 3.79s
   - **Issue**: Computing histograms for numeric and datetime columns
   - **Impact**: 🔴 **CRITICAL** - Single biggest bottleneck

#### 2. **String Pattern Matching: 2.26s (20% of time)**
   - **Location**: regex `split()` operations
   - **Calls**: 710,226 calls
   - **Self time**: 1.10s (split) + 1.16s (lambda overhead)
   - **Issue**: Character encoding splits each string into individual characters
   - **Impact**: 🟡 **HIGH**

#### 3. **split_sub_columns_digit: 3.69s (32% of time)**
   - **Location**: `numeric.py:108`
   - **Calls**: 12 times (numeric columns)
   - **Issue**: Splitting numeric values into digit sub-columns
   - **Impact**: 🟡 **HIGH**

#### 4. **is_sequential checks: 1.29s (11% of time)**
   - **Location**: `_common.py:205`
   - **Calls**: 56 times
   - **Issue**: Checking if each column is sequential (applies lambda to entire series)
   - **Impact**: 🟠 **MEDIUM**

#### 5. **Type conversions: ~0.8s (7% of time)**
   - **Location**: Various `astype()`, `to_numpy()` calls
   - **Issue**: String/Arrow conversions, dtype changes
   - **Impact**: 🟢 **LOW-MEDIUM**

### ENCODE Operation (3.14s total)

#### 1. **is_a_list checks: 1.09s (35% of time)**
   - **Location**: `_common.py:201`
   - **Calls**: 3,000,000 times (!)
   - **Issue**: Checking every value to see if it's a list (for sequential data)
   - **Impact**: 🔴 **CRITICAL**

#### 2. **map_array operations: 1.74s (55% of time)**
   - **Location**: pandas `algorithms.py:1667`
   - **Calls**: 60 times
   - **Self time**: 0.35s
   - **Issue**: Applying transformations to entire arrays
   - **Impact**: 🟡 **HIGH**

#### 3. **Character encoding splits: 0.80s (26% of time)**
   - **Location**: `character.py:98` and split operations
   - **Issue**: Similar to ANALYZE - splitting strings into characters
   - **Impact**: 🟡 **HIGH**

#### 4. **Type checking overhead: 0.45s (14% of time)**
   - **Location**: `isinstance()` and ABC checks
   - **Calls**: 3,916,990 times
   - **Impact**: 🟢 **LOW-MEDIUM**

---

## Optimization Recommendations

### Priority 1: CRITICAL Optimizations (Expected Speedup: 3-5x)

#### 1.1 Optimize `compute_log_histogram` ✅ **HIGH IMPACT**

**Current Issue**:
- Uses Python loops to compute histogram edges
- Called 16 times, taking 4.06s total

**Proposed Solutions**:

**A. Vectorize histogram computation:**
```python
# In _common.py:636
def compute_log_histogram(values: np.ndarray, n_bins: int = 100) -> dict:
    # Current: iterative edge computation
    # Proposed: Use numpy's efficient histogram functions

    # Option 1: Direct numpy histogram (fastest)
    if len(values) > 0:
        counts, edges = np.histogram(np.log10(np.abs(values) + 1e-10), bins=n_bins)
        return {"counts": counts.tolist(), "edges": edges.tolist()}

    # Option 2: Use scipy.stats for more advanced binning
    # Option 3: Pre-compute bins and reuse
```

**Expected Speedup**: 5-10x faster (from 4.06s to 0.4-0.8s)

**B. Cache histogram edges:**
```python
# Compute edges once per data type, reuse for all columns
_HISTOGRAM_EDGES_CACHE = {}

def compute_log_histogram(values, n_bins=100):
    dtype_key = (values.dtype, n_bins)
    if dtype_key not in _HISTOGRAM_EDGES_CACHE:
        _HISTOGRAM_EDGES_CACHE[dtype_key] = _compute_edges(n_bins)
    edges = _HISTOGRAM_EDGES_CACHE[dtype_key]
    # ... use cached edges
```

#### 1.2 Optimize `is_a_list` checks ✅ **HIGH IMPACT**

**Current Issue**:
- Called 3,000,000 times in ENCODE (once per value!)
- Each call does `isinstance(v, (list, tuple, np.ndarray))`
- Takes 1.09s (35% of encode time)

**Root Cause**: Checking every single value to determine if data is sequential

**Proposed Solutions**:

**A. Column-level sequential flag (BEST):**
```python
# In encoding.py or analysis.py
def encode_df(df, stats, ...):
    # Pre-determine which columns are sequential
    sequential_columns = {
        col: _is_column_sequential(df[col])  # Check once per column
        for col in df.columns
    }

    # Pass flag down instead of checking every value
    for column in stats["columns"].keys():
        is_seq = sequential_columns.get(column, False)
        # Skip per-value checks, use flag directly
```

**B. Optimize the check itself:**
```python
# Current: _common.py:201
def is_a_list(v):
    return isinstance(v, (list, tuple, np.ndarray))

# Optimized version 1: Type caching
_LIST_TYPES = (list, tuple, np.ndarray)
def is_a_list(v):
    return isinstance(v, _LIST_TYPES)

# Optimized version 2: Early returns
def is_a_list(v):
    t = type(v)
    return t is list or t is tuple or t is np.ndarray

# Optimized version 3: For flat data, skip entirely
def _encode_flat_col(...):
    # We know it's flat, don't check is_a_list at all
    # Process directly as flat array
```

**Expected Speedup**: 10-20x faster (from 1.09s to 0.05-0.1s)

#### 1.3 Optimize `split_sub_columns_digit` ✅ **HIGH IMPACT**

**Current Issue**:
- Takes 3.69s (32% of analyze time)
- Converts numbers to strings, splits into digits

**Proposed Solutions**:

**A. Vectorized digit extraction:**
```python
# Current approach: string-based splitting
def split_sub_columns_digit(values):
    str_values = values.astype(str)
    # ... split each string

# Proposed: Mathematical digit extraction
def split_sub_columns_digit(values):
    # For integers: use modulo/division
    # Much faster than string operations
    values = np.abs(values.astype(np.int64))

    digits = []
    for position in range(max_digits):
        digit = (values // 10**position) % 10
        digits.append(digit)

    return pd.DataFrame(digits).T
```

**B. Use numba for JIT compilation:**
```python
import numba

@numba.jit(nopython=True)
def extract_digits_fast(values, n_digits):
    result = np.zeros((len(values), n_digits), dtype=np.int8)
    for i in range(len(values)):
        val = abs(values[i])
        for j in range(n_digits):
            result[i, j] = (val // 10**j) % 10
    return result
```

**Expected Speedup**: 5-10x faster (from 3.69s to 0.4-0.7s)

---

### Priority 2: HIGH Impact Optimizations (Expected Speedup: 1.5-2x)

#### 2.1 Reduce String Split Operations

**Current Issue**:
- Character encoding uses regex `split()` 710,226 times
- Takes 2.26s in ANALYZE, 0.80s in ENCODE

**Proposed Solutions**:

**A. Vectorize character splitting:**
```python
# Current: iterate and split each string
for s in strings:
    chars = list(s)  # or regex split

# Proposed: Use numpy array operations
def split_strings_to_chars(series):
    # Convert to numpy array view (zero-copy)
    arr = np.array(series.values, dtype='U')
    max_len = series.str.len().max()

    # Use numpy's view to split into characters
    char_array = arr.view(f'U1').reshape(len(arr), -1)[:, :max_len]
    return char_array
```

**B. Consider alternative encoding:**
```python
# Instead of splitting into individual characters, use:
# 1. Byte/UTF-8 encoding (faster)
# 2. Character n-grams (if suitable)
# 3. Hash-based encoding for long strings
```

**Expected Speedup**: 3-5x faster (from 2.26s to 0.5-0.8s)

#### 2.2 Optimize `is_sequential` Checks

**Current Issue**:
- Called 56 times, takes 1.29s in ANALYZE
- Uses `series.apply(lambda ...)` which is slow

**Proposed Solutions**:

**A. Quick check on first few values:**
```python
def is_sequential(series, sample_size=100):
    # Check just the first N non-null values
    sample = series.dropna().iloc[:sample_size]
    if len(sample) == 0:
        return False

    # Fast check: if first value is list-like, assume all are
    first_val = sample.iloc[0]
    return isinstance(first_val, (list, tuple, np.ndarray))
```

**B. Cache results:**
```python
# Store is_sequential result in column metadata
# Check once, reuse everywhere
```

**Expected Speedup**: 5-10x faster (from 1.29s to 0.15-0.25s)

#### 2.3 Reduce Type Conversions

**Current Issue**:
- Multiple `astype()`, `to_numpy()`, `_from_sequence()` calls
- Takes ~0.8s in ANALYZE, ~0.3s in ENCODE

**Proposed Solutions**:

**A. Minimize conversions:**
```python
# Keep data in optimal format (Arrow/numpy) longer
# Batch conversions together
# Avoid back-and-forth conversions

# Example: Keep as numpy array until final output
def analyze_categorical(values: pd.Series, ...):
    # Current: converts to various types multiple times
    # Proposed: work with numpy array directly
    arr = values.to_numpy()  # Convert once
    # ... do all analysis on arr
    # Convert back only at the end
```

**B. Use zero-copy views where possible:**
```python
# Instead of: df.copy()
# Use: df.view() or pass references
```

**Expected Speedup**: 2-3x faster (from 0.8s to 0.3-0.4s)

---

### Priority 3: MEDIUM Impact Optimizations (Expected Speedup: 1.2-1.5x)

#### 3.1 Parallel Processing Improvements

**Current State**: Using `n_jobs=1` (serial processing)

**Proposed**:
```python
# 1. Enable parallelization at column level (already exists, but optimize)
n_jobs = min(cpu_count() - 1, len(columns))

# 2. Use shared memory for large DataFrames
from multiprocessing import shared_memory

# 3. Batch small operations together to reduce overhead
# Don't parallelize if < 10 columns or < 10k rows
```

**Expected Speedup**: 2-4x with proper parallelization (4 cores)

#### 3.2 Reduce Memory Allocations

**Proposed**:
```python
# 1. Pre-allocate arrays where size is known
result = np.empty((n_rows, n_cols), dtype=dtype)

# 2. Reuse temporary arrays
# 3. Use generator expressions instead of list comprehensions for large data
# 4. Clear intermediate results explicitly
```

**Expected Speedup**: 1.2-1.3x

#### 3.3 Optimize JSON Operations

**Issue**: JSON encoding/decoding in stats files

**Proposed**:
```python
# Use faster JSON library
import orjson  # or ujson

# Or use pickle/parquet for intermediate stats
# (if stats don't need to be human-readable)
```

**Expected Speedup**: 1.1-1.2x

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days) - **3-4x speedup**
1. ✅ Optimize `is_a_list` - column-level sequential detection
2. ✅ Optimize `is_sequential` - sample-based check + caching
3. ✅ Cache histogram edges where possible
4. ✅ Reduce unnecessary type conversions

### Phase 2: Major Refactors (3-5 days) - **2-3x additional speedup**
1. ✅ Vectorize `compute_log_histogram`
2. ✅ Vectorize `split_sub_columns_digit` with numba
3. ✅ Optimize character splitting operations
4. ✅ Improve parallelization strategy

### Phase 3: Advanced Optimizations (5-10 days) - **1.5-2x additional speedup**
1. Use Cython/numba for hot loops
2. Implement zero-copy operations
3. Add GPU support for histogram computation
4. Optimize memory layout and access patterns

---

## Expected Overall Speedup

**Conservative Estimate**:
- ANALYZE: 11.47s → **2-3s** (4-6x faster)
- ENCODE: 3.14s → **0.8-1.2s** (3-4x faster)
- TOTAL: 14.61s → **2.8-4.2s** (3.5-5x faster)

**Optimistic Estimate** (with all optimizations):
- ANALYZE: 11.47s → **1-1.5s** (8-10x faster)
- ENCODE: 3.14s → **0.4-0.6s** (5-8x faster)
- TOTAL: 14.61s → **1.4-2.1s** (7-10x faster)

---

## Specific Code Locations to Optimize

### Most Critical Files:
1. ✅ `mostlyai/engine/_common.py`
   - `compute_log_histogram()` (line 636)
   - `is_a_list()` (line 201)
   - `is_sequential()` (line 205)

2. ✅ `mostlyai/engine/_encoding_types/tabular/numeric.py`
   - `split_sub_columns_digit()` (line 108)
   - `analyze_numeric()` (line 145)

3. ✅ `mostlyai/engine/_encoding_types/tabular/character.py`
   - `analyze_character()` (line 33)
   - `encode_character()` (line 98)

4. ✅ `mostlyai/engine/_tabular/encoding.py`
   - `_encode_col()` (line 246)
   - `encode_df()` (line 181)

5. ✅ `mostlyai/engine/analysis.py`
   - `_analyze_col()` (line 514)
   - `_analyze_flat_col()` (line 554)

---

## Testing Strategy

1. **Unit Tests**: Test each optimized function independently
2. **Integration Tests**: Ensure end-to-end correctness
3. **Performance Tests**: Measure before/after for each optimization
4. **Regression Tests**: Ensure output statistics are identical
5. **Edge Cases**: Empty data, single row, all nulls, sequential data

---

## Monitoring & Profiling

```python
# Add lightweight timing to production code
import time
from functools import wraps

def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        if elapsed > 0.1:  # Log if > 100ms
            _LOG.debug(f"{func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper
```

---

## Next Steps

1. **Review** this analysis with the team
2. **Prioritize** which optimizations to implement first
3. **Implement** Phase 1 quick wins
4. **Measure** actual speedups achieved
5. **Iterate** based on results

---

## Appendix: Detailed Profiling Data

See detailed profiling output files:
- `profiling_results/analyze_profile.txt`
- `profiling_results/encode_profile.txt`

Generated by: `profile_analyze_encode_v2.py`
