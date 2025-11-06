# Synthetic Data Engine 💎

![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai-engine)
[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai-engine/)
[![stats](https://pepy.tech/badge/mostlyai-engine)](https://pypi.org/project/mostlyai-engine/)
![license](https://img.shields.io/github/license/mostly-ai/mostlyai-engine)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mostlyai-engine)

[Documentation](https://mostly-ai.github.io/mostlyai-engine/) | [Technical Paper](https://arxiv.org/abs/2501.12012) | [Free Cloud Service](https://app.mostly.ai/)

Create high-fidelity privacy-safe synthetic data:

1. train a generative model
    * fit to flat or sequential data
    * optionally provide context data
    * control training time & parameters
    * monitor training progress
    * ensure differential privacy
2. generate synthetic data samples to your needs:
    * up-sample / down-sample
    * conditionally generate
    * rebalance categories
    * impute missings
    * incorporate fairness
    * adjust sampling temperature

...all within your safe compute environment, all with a few lines of Python code 💥.

Note: This library is the underlying model engine of the [Synthetic Data SDK](https://github.com/mostly-ai/mostlyai). Please refer to the latter, for an easy-to-use, higher-level software toolkit.


## Installation

It is highly recommended to install the package within a dedicated virtual environment using [uv](https://docs.astral.sh/uv/).

The latest release of `mostlyai-engine` can be installed via uv:

```bash
uv pip install -U mostlyai-engine
```

or alternatively for a GPU setup (needed for LLM finetuning and inference):
```bash
uv pip install -U 'mostlyai-engine[gpu]'
```

On Linux, one can explicitly install the CPU-only variant of torch together with `mostlyai-engine`:

```bash
uv pip install -U torch==2.8.0+cpu torchvision==0.23.0+cpu mostlyai-engine --extra-index-url https://download.pytorch.org/whl/cpu
```

## Basic Usage of TabularARGN

The `TabularARGN` class provides a scikit-learn-compatible interface for working with tabular data. It can be used for synthetic data generation, classification, regression, density estimation, and imputation.

**Key advantage**: Train the model once, then flexibly reuse it for any downstream task—whether that's regression, classification, imputation, or conditional sampling—without needing to retrain.

### Flat Tables

Load your data and train the model:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from mostlyai.engine import TabularARGN

# load original data
url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/census"
data = pd.read_csv(f"{url}/census.csv.gz")
trn_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

# fit the model
argn = TabularARGN(max_training_time=1, random_state=42)
argn.fit(trn_data)
```

#### Synthetic Data Generation

Generate new synthetic samples, either unconditionally or conditionally:

```python
# unconditional sampling
syn_data = argn.sample(n_samples=1000)

# conditional sampling with seed values
syn_data = argn.sample(seed=pd.DataFrame({"age": [25, 50], "education": ["Bachelors", "HS-grad"]}))
```

#### Classification

Use the trained model for classification tasks:

```python
from sklearn.metrics import accuracy_score, roc_auc_score
from mostlyai.engine import TabularARGNClassifier

clf = TabularARGNClassifier(argn, target="income", n_draws=100)
preds = clf.predict(val_data)
probs = clf.predict_proba(val_data)

accuracy = accuracy_score(val_data["income"], preds)
auc = roc_auc_score(val_data["income"], probs[:, 1])
print(f"Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
```

#### Regression

Use the trained model for regression tasks:

```python
from sklearn.metrics import mean_absolute_error
from mostlyai.engine import TabularARGNRegressor

reg = TabularARGNRegressor(argn, target="age", n_draws=100)
preds = reg.predict(val_data)

mae = mean_absolute_error(val_data["age"], preds)
print(f"MAE: {mae:.1f} years")
```

#### Density Estimation

Compute log probabilities to detect outliers:

```python
log_probs = argn.log_prob(data)
idx_outlier = log_probs.argmin()
most_unusual = data.iloc[idx_outlier]
```

#### Imputation

Fill in missing values using the trained model:

```python
from mostlyai.engine import TabularARGNImputer

# with pre-trained model
imputer = TabularARGNImputer(argn)
imputed_data = imputer.transform(data_with_missings)

# or fit and impute in one go
imputer = TabularARGNImputer(max_training_time=1)
imputed_data = imputer.fit_transform(data_with_missings)
```

### Sequential Tables

For sequential data (e.g., time series or event logs), specify the context key:

```python
import pandas as pd
from mostlyai.engine import TabularARGN

# load sequential data
url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/baseball"
trn_df = pd.read_csv(f"{url}/batting.csv.gz")

# fit model with context key
argn = TabularARGN(tgt_context_key="players_id", max_training_time=1, random_state=42)
argn.fit(trn_df)

# generate synthetic sequences
syn_df = argn.sample(n_samples=100)

# compute log probabilities for sequences
log_probs = argn.log_prob(trn_df.head(100))
```

## Advanced Usage for TABULAR and LANGUAGE data

For more fine-grained control over the training and generation process, you can use the lower-level engine functions. This approach is useful when you need to customize the workflow, inspect intermediate results, or work with language models.

### Tabular Model: flat data

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

### Tabular Model: sequential data with context

```python
from pathlib import Path
import pandas as pd
from mostlyai import engine

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

### Language Model

For text data, you can leverage pre-trained language models or train lightweight LSTM models from scratch:

```python
from pathlib import Path
import pandas as pd
from mostlyai import engine

# init workspace and logging
ws = Path("ws-language-flat")
engine.init_logging()

# load original data
trn_df = pd.read_parquet("https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/headlines/headlines.parquet")
trn_df = trn_df.sample(n=10_000, random_state=42)

# execute the engine steps
engine.split(                         # split data as PQT files for `trn` + `val` to `{ws}/OriginalData/tgt-data`
    workspace_dir=ws,
    tgt_data=trn_df,
    tgt_encoding_types={
        'category': 'LANGUAGE_CATEGORICAL',
        'date': 'LANGUAGE_DATETIME',
        'headline': 'LANGUAGE_TEXT',
    }
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
pd.read_parquet(ws / "SyntheticData") # load synthetic data
```
