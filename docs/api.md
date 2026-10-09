---
hide:
  - navigation
---

## Engine Reference

::: mostlyai.engine
    options:
        members:
            - split
            - analyze
            - encode
            - train
            - generate

## Schema Reference

::: mostlyai.engine.domain
    options:
        filters:
            - "!^CustomBaseModel"

## Large datasets and parallel processing

Analysis and tabular encoding process one partition at a time. For large datasets,
use `split(..., n_partitions=...)` to reduce the size of each partition, and release
input DataFrames when they are no longer needed before calling `analyze` or `encode`.
Sequential records belonging to the same context stay together in a partition.

The default parallel backend remains Joblib's process-based `loky`. To avoid copying
input columns into worker processes, select threads using Joblib's configuration:

```python
from joblib import parallel_config
from mostlyai import engine

with parallel_config(backend="threading"):
    engine.analyze(workspace_dir="engine-ws")
    engine.encode(workspace_dir="engine-ws")
```

Threads share input data, but operations still allocate intermediate and encoded
arrays. Partitioning is therefore useful with either backend. Python-heavy work may
run slower with threads, and stochastic encoding results can depend on thread
scheduling. Language encoding does not use Joblib parallel workers.
