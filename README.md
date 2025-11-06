# Synthetic Data Engine 💎

![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai-engine)
[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai-engine/)
[![stats](https://pepy.tech/badge/mostlyai-engine)](https://pypi.org/project/mostlyai-engine/)
![license](https://img.shields.io/github/license/mostly-ai/mostlyai-engine)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mostlyai-engine)

[Documentation](https://mostly-ai.github.io/mostlyai-engine/) | [Technical Paper](https://arxiv.org/abs/2501.12012) | [Free Cloud Service](https://app.mostly.ai/)

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

Note: This library is the underlying model engine of the [Synthetic Data SDK](https://github.com/mostly-ai/mostlyai). Please refer to the latter, for an easy-to-use, higher-level software toolkit.


## Installation

The latest release of `mostlyai-engine` can be installed via pip:

```bash
pip install -U mostlyai-engine
```

or alternatively for a GPU setup (needed for LLM finetuning and inference):
```bash
pip install -U 'mostlyai-engine[gpu]'
```

On Linux, one can explicitly install the CPU-only variant of torch together with `mostlyai-engine`:

```bash
pip install -U torch==2.8.0+cpu torchvision==0.23.0+cpu mostlyai-engine --extra-index-url https://download.pytorch.org/whl/cpu
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

## Scikit-Learn Interface

The engine provides a scikit-learn compatible interface for common machine learning tasks. This allows you to use generative models for classification, regression, imputation, and density estimation with familiar sklearn APIs.

### Classification

Train a generative classifier and predict class labels:

```python
import pandas as pd
from mostlyai.engine import TabularARGNClassifier
from sklearn.model_selection import train_test_split

# load census data
url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/census"
df = pd.read_csv(f"{url}/census.csv.gz")

# split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# prepare features and target
X_train = train_df.drop(columns=['income'])
y_train = train_df['income']
X_test = test_df.drop(columns=['income'])
y_test = test_df['income']

# train classifier
clf = TabularARGNClassifier(
    target='income',
    max_training_time=1,  # 1 minute for demo
    verbose=1
)
clf.fit(pd.concat([X_train, y_train], axis=1))

# make predictions
y_pred = clf.predict(X_test, n_draws=10)
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")

# get prediction probabilities
y_proba = clf.predict_proba(X_test, n_draws=10)
print(f"Prediction probabilities shape: {y_proba.shape}")
```

### Regression

Train a generative regressor for continuous target prediction:

```python
import pandas as pd
from mostlyai.engine import TabularARGNRegressor

# load census data
url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/census"
df = pd.read_csv(f"{url}/census.csv.gz")

# prepare data (predicting age)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_df.drop(columns=['age'])
y_train = train_df['age']
X_test = test_df.drop(columns=['age'])
y_test = test_df['age']

# train regressor
reg = TabularARGNRegressor(
    target='age',
    max_training_time=1,
    verbose=1
)
reg.fit(pd.concat([X_train, y_train], axis=1))

# make predictions
y_pred = reg.predict(X_test, n_draws=10)
r2_score = reg.score(X_test, y_test)
print(f"R² Score: {r2_score:.3f}")
```

### Imputation

Impute missing values using a generative model:

```python
import pandas as pd
import numpy as np
from mostlyai.engine import TabularARGNImputer

# load census data
url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/census"
df = pd.read_csv(f"{url}/census.csv.gz")

# introduce some missing values
df_with_missing = df.copy()
df_with_missing.loc[df_with_missing.sample(frac=0.1, random_state=42).index, 'age'] = np.nan
df_with_missing.loc[df_with_missing.sample(frac=0.1, random_state=43).index, 'education'] = np.nan

# train imputer on complete data
imp = TabularARGNImputer(
    max_training_time=1,
    verbose=1
)
imp.fit(df)

# impute missing values
df_imputed = imp.transform(df_with_missing)
print(f"Missing values before: {df_with_missing.isnull().sum().sum()}")
print(f"Missing values after: {df_imputed.isnull().sum().sum()}")
```

### Density Estimation

Estimate the log-likelihood of data samples (tabular models only):

```python
import pandas as pd
from mostlyai.engine import TabularARGN

# load census data
url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/census"
df = pd.read_csv(f"{url}/census.csv.gz")

# split data
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# train model
model = TabularARGN(
    max_training_time=1,
    verbose=1
)
model.fit(train_df)

# compute log-likelihood for each sample
log_likelihood = model.log_prob(test_df)
print(f"Average log-likelihood: {log_likelihood.mean():.2f}")
```

### Unconditional Sampling

Generate synthetic data without conditioning:

```python
import pandas as pd
from mostlyai.engine import TabularARGN

# load census data
url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/census"
df = pd.read_csv(f"{url}/census.csv.gz")

# train generative model
model = TabularARGN(
    max_training_time=1,
    verbose=1
)
model.fit(df)

# generate synthetic samples
synthetic_df = model.sample(n_samples=1000)
print(f"Generated {len(synthetic_df)} synthetic samples")
print(synthetic_df.head())
```

### Conditional Sampling

Generate synthetic data conditioned on specific features:

```python
import pandas as pd
from mostlyai.engine import TabularARGN

# load census data
url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/census"
df = pd.read_csv(f"{url}/census.csv.gz")

# train model
model = TabularARGN(max_training_time=1, verbose=1)
model.fit(df)

# create seed data with partial information
seed_df = pd.DataFrame({
    'age': [25, 35, 45],
    'education': ['Bachelors', 'Masters', 'Doctorate']
})

# generate complete samples conditioned on seed
synthetic_df = model.sample(seed=seed_df)
print("Conditioned synthetic samples:")
print(synthetic_df)
```

### Advanced: Reusing Fitted Models

You can reuse a fitted `TabularARGN` model for multiple downstream tasks:

```python
import pandas as pd
from mostlyai.engine import TabularARGN, TabularARGNClassifier, TabularARGNRegressor

# load census data
url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/census"
df = pd.read_csv(f"{url}/census.csv.gz")

# train base model once
base_model = TabularARGN(max_training_time=2, verbose=1)
base_model.fit(df)

# reuse for classification
clf = TabularARGNClassifier(X=base_model, target='income')
# clf is already fitted, can directly predict

# reuse for regression
reg = TabularARGNRegressor(X=base_model, target='age')
# reg is already fitted, can directly predict

# reuse for unconditional sampling
synthetic_samples = base_model.sample(n_samples=500)
```
