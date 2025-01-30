# Synthetic Data Engine 💎
[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai-engine/) ![license](https://img.shields.io/github/license/mostly-ai/mostlyai-engine) ![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai-engine) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mostlyai-engine) [![stats](https://pepy.tech/badge/mostlyai-engine)](https://pypi.org/project/mostlyai-engine/)

[Package Documentation](https://mostly-ai.github.io/mostlyai-engine/) | [Platform Documentation](https://mostly.ai/docs)

Create high-fidelity privacy-safe synthetic data:

1. prepare, analyze, and encode original data
2. train a generative model on the encoded data
3. generate synthetic data samples to your needs:
    * up-sample / down-sample
    * conditionally generate
    * rebalance categories
    * impute missings
    * incorporate fairness
    * adjust sampling temperature

...all within your safe compute environment, all with a few lines of Python code 💥.

Note: This library is the underlying model engine of the [Synthetic Data SDK ✨](https://github.com/mostly-ai/mostlyai). Please refer to the latter, for an easy-to-use, higher-level software toolkit.


## (Experimental) Installation of development version for AMD GPUs

**Prerequisites: install [uv](https://docs.astral.sh/uv/) package manager**

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

**Create the project environment**

```shell
git clone https://github.com/mostly-ai/mostlyai-engine.git
cd mostlyai-engine
git checkout rocm-support
uv sync --frozen
# build bitsandbytes and vllm from source as there're no pre-built ROCm wheels
source install_bnb_vllm_rocm.sh
```

## Quick start

### Tabular Model: flat data, without context

```python
from pathlib import Path
import pandas as pd
from mostlyai import engine

# set up workspace and default logging
ws = Path("ws-tabular-flat")
engine.init_logging()

# load original data
url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/census"
trn_df = pd.read_csv(f"{url}/census.csv.gz")

# execute the engine steps
engine.split(                         # split data as PQT files for `trn` + `val` to `{ws}/OriginalData/tgt-data`
  workspace_dir=ws,
  tgt_data=trn_df,
  model_type="TABULAR",
)
engine.analyze(workspace_dir=ws)      # generate column-level statistics to `{ws}/ModelData/tgt-stats/stats.json`
engine.encode(workspace_dir=ws)       # encode training data to `{ws}/OriginalData/encoded-data`
engine.train(                         # train model and store to `{ws}/ModelStore/model-data`
    workspace_dir=ws,
    max_training_time=1,              # limit TRAIN to 1 minute for demo purposes
)
engine.generate(workspace_dir=ws)     # use model to generate synthetic samples to `{ws}/SyntheticData`
pd.read_parquet(ws / "SyntheticData") # load synthetic data
```

### Tabular Model: sequential data, with context

```python
from pathlib import Path
import pandas as pd
from mostlyai import engine

engine.init_logging()

# set up workspace and default logging
ws = Path("ws-tabular-sequential")
engine.init_logging()

# load original data
url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/baseball"
trn_ctx_df = pd.read_csv(f"{url}/players.csv.gz")  # context data
trn_tgt_df = pd.read_csv(f"{url}/batting.csv.gz")  # target data

# execute the engine steps
engine.split(                         # split data as PQT files for `trn` + `val` to `{ws}/OriginalData/(tgt|ctx)-data`
  workspace_dir=ws,
  tgt_data=trn_tgt_df,
  ctx_data=trn_ctx_df,
  tgt_context_key="players_id",
  ctx_primary_key="id",
  model_type="TABULAR",
)
engine.analyze(workspace_dir=ws)      # generate column-level statistics to `{ws}/ModelStore/(tgt|ctx)-data/stats.json`
engine.encode(workspace_dir=ws)       # encode training data to `{ws}/OriginalData/encoded-data`
engine.train(                         # train model and store to `{ws}/ModelStore/model-data`
    workspace_dir=ws,
    max_training_time=1,              # limit TRAIN to 1 minute for demo purposes
)
engine.generate(workspace_dir=ws)     # use model to generate synthetic samples to `{ws}/SyntheticData`
pd.read_parquet(ws / "SyntheticData") # load synthetic data
```

### Language Model: flat data, without context

```python
from pathlib import Path
import pandas as pd
from mostlyai import engine

# init workspace and logging
ws = Path("ws-language-flat")
engine.init_logging()

# load original data
trn_df = pd.read_parquet("https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/headlines/headlines.parquet")
trn_df = trn_df.sample(n=10_000, random_state=42)[['category', 'headline']]

# execute the engine steps
engine.split(                         # split data as PQT files for `trn` + `val` to `{ws}/OriginalData/tgt-data`
    workspace_dir=ws,
    tgt_data=trn_df,
    model_type="LANGUAGE",
)
engine.analyze(workspace_dir=ws)      # generate column-level statistics to `{ws}/ModelStore/tgt-stats/stats.json`
engine.encode(workspace_dir=ws)       # encode training data to `{ws}/OriginalData/encoded-data`
engine.train(                         # train model and store to `{ws}/ModelStore/model-data`
    workspace_dir=ws,
    max_training_time=2,                   # limit TRAIN to 2 minute for demo purposes
    model="MOSTLY_AI/LSTMFromScratch-3m",  # use a light-weight LSTM model, trained from scratch (GPU recommended)
    # model="microsoft/phi-1.5",           # alternatively use a pre-trained HF-hosted LLM model (GPU required)
)
engine.generate(                      # use model to generate synthetic samples to `{ws}/SyntheticData`
    workspace_dir=ws,
    sample_size=10,
)
```
