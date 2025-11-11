# Synthetic Data Engine 💎

![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai-engine)
[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai-engine/)
[![stats](https://pepy.tech/badge/mostlyai-engine)](https://pypi.org/project/mostlyai-engine/)
![license](https://img.shields.io/github/license/mostly-ai/mostlyai-engine)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mostlyai-engine)

[Documentation](https://mostly-ai.github.io/mostlyai-engine/) | [Technical Paper](https://arxiv.org/abs/2501.12012) | [Free Cloud Service](https://app.mostly.ai/)

Create high-fidelity privacy-safe synthetic data:

1. train a generative model once
    * fit flat or sequential data
    * control training time & parameters
    * monitor training progress
    * optionally enable differential privacy
    * optionally provide context data
2. generate synthetic data samples to your needs:
    * up-sample / down-sample
    * conditionally generate
    * rebalance categories
    * impute missing values
    * incorporate fairness
    * adjust sampling temperature
    * predict / classify / regress
    * detect outliers / anomalies
    * and more

...all within your own compute environment, all with a few lines of Python code 💥.

Models only need to be trained once and can then be flexibly reused for various downstream tasks — such as regression, classification, imputation, or sampling — without the need for retraining.

Two models are available:

1. `TabularARGN`: For structured, flat or sequential tabular data.
2. `LanguageModel`: For semi-structured, flat textual tabular data.

This library serves as the core model engine for the [Synthetic Data SDK](https://github.com/mostly-ai/mostlyai). For an easy-to-use, higher-level toolkit, please refer to the SDK.


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

The `TabularARGN` class provides a scikit-learn-compatible interface for working with structured tabular data. It can be used for synthetic data generation, classification, regression, density estimation, and imputation.

### Flat Tables

Load your data and train the model:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from mostlyai.engine import TabularARGN

# prepare data
df = pd.read_csv("https://github.com/user-attachments/files/23478827/census10k.csv.gz")
df_train, df_test = train_test_split(df, test_size=0.2)

# fit TabularARGN
argn = TabularARGN()
argn.fit(df_train)
```

#### Sampling / Synthetic Data Generation

Generate new synthetic samples:

```python
# unconditional sampling
argn.sample(n_samples=1000)
```

Generate new synthetic samples conditionally:

```python
# prepare seed
df_seed = pd.DataFrame({
    "age": [25, 50],
    "education": ["Bachelors", "HS-grad"]
})

# conditional sampling
argn.sample(seed_data=df_seed)
```

#### Density Estimation / Log Likelihood

Compute log probabilities to detect outliers:

```python
# calculate log likelihoods
log_probs = argn.log_prob(df)

# determine biggest outlier
df.iloc[log_probs.argmin()]
```

#### Imputation / Filling Gaps

Fill in missing values:

```python
# prepare demo data with missings
df_with_missings = df_test.head(300)
df_with_missings.loc[0:299, "age"] = pd.NA
df_with_missings.loc[0:199, "race"] = pd.NA
df_with_missings.loc[100:299, "income"] = pd.NA

# impute missing values
argn.impute(df_with_missings)
```

#### Predictions / Classification

Predict any categorical target column:

```python
from sklearn.metrics import accuracy_score, roc_auc_score

# predict class labels for a categorical
preds = argn.predict(df_test, target="income", n_draws=10, agg_fn="mode")

# predict class probabilities for a categorical
probs = argn.predict_proba(df_test, target="income", n_draws=10, agg_fn="mode")

# evaluate performance
accuracy = accuracy_score(df_test["income"], preds)
auc = roc_auc_score(df_test["income"], probs[:, 1])
print(f"Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
```

#### Predictions / Regression

Predict any numerical target column:

```python
from sklearn.metrics import mean_absolute_error

# predict target values
preds = argn.predict(df_test, target="age", n_draws=10, agg_fn="mean")

# evaluate performance
mae = mean_absolute_error(df_test["age"], preds)
print(f"MAE: {mae:.1f} years")
```

### Sequential Tables

For sequential data (e.g., time series or event logs), specify the context key:

```python
import pandas as pd
from mostlyai.engine import TabularARGN

# load sequential data
df = pd.read_csv("https://github.com/user-attachments/files/23479267/batting.csv.gz")

# fit TabularARGN with a context key column
argn = TabularARGN(
    tgt_context_key="players_id",
    max_training_time=1,
    random_state=42,
)
argn.fit(df)
```

Use the trained model to generate new samples:
```python
# unconditional sampling
argn.sample(n_samples=100)
```

## Basic Usage of LanguageModel

The `LanguageModel` class provides a scikit-learn-compatible interface for working with semi-structured textual data. It leverages pre-trained language models or trains lightweight LSTM models from scratch to generate synthetic text data.

### Text Data

Load your data and train the model:

```python
import pandas as pd
from mostlyai.engine import LanguageModel

# load data
df = pd.read_csv("https://github.com/user-attachments/files/23479325/news10k.csv.gz")

# fit LanguageModel
lm = LanguageModel(
    model="MOSTLY_AI/LSTMFromScratch-3m",
    tgt_encoding_types={
        'category': 'LANGUAGE_CATEGORICAL',
        'date': 'LANGUAGE_DATETIME',
        'headline': 'LANGUAGE_TEXT',
    },
    max_training_time=2,
    random_state=42,
)
lm.fit(df)
```

#### Synthetic Text Generation

Generate new synthetic samples using the trained language model:

```python
# unconditional sampling
lm.sample(
    n_samples=100,
    sampling_temperature=0.5,
)
```

```python
# prepare seed
df_seed = pd.DataFrame({"category": ["business", "tech"]})

# conditional sampling with seed values
syn_data = lm.sample(seed_data=df_seed, sampling_temperature=0.5)
```

**Note**: The default model is `"MOSTLY_AI/LSTMFromScratch-3m"`, a lightweight LSTM model trained from scratch (GPU recommended). You can also use pre-trained HuggingFace models by setting e.g. `model="microsoft/phi-1.5"` (GPU required).
