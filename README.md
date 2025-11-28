# Synthetic Data Engine 💎

![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai-engine)
[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai-engine/)
[![stats](https://pepy.tech/badge/mostlyai-engine)](https://pypi.org/project/mostlyai-engine/)
![license](https://img.shields.io/github/license/mostly-ai/mostlyai-engine)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mostlyai-engine)

[Documentation](https://mostly-ai.github.io/mostlyai-engine/) | [Technical Paper](https://arxiv.org/abs/2501.12012) | [Free Cloud Service](https://app.mostly.ai/)

Create high-fidelity privacy-safe synthetic data:

1. train a generative model once:
    * train on flat or sequential data
    * control training time & params
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

Note: Models only need to be trained once and can then be flexibly reused for various downstream tasks — such as regression, classification, imputation, or sampling — without the need for retraining.

Two model classes with these methods are available:

1. `TabularARGN()`: For structured, flat or sequential tabular data.
   * `argn.fit(data)`: Train a TabularARGN model
   * `argn.sample(n_samples)`: Generate samples
   * `argn.predict(target, n_draws, agg_fn)`: Predict a feature
   * `argn.predict_proba(target)`: Estimate probabilities
   * `argn.log_prob(data)`: Compute log likelihood
   * `argn.impute(data)`: Fill missing values
2. `LanguageModel()`: For semi-structured, flat textual tabular data.
   * `.fit(data)`: Train a Language model
   * `.sample(n_samples)`: Generate samples

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

## TabularARGN for Flat Data

The `TabularARGN` class provides a scikit-learn-compatible interface for working with structured tabular data. It can be used for synthetic data generation, classification, regression, and imputation.

### Model Training

Load your data and train the model:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from mostlyai.engine import TabularARGN

# prepare data
data = pd.read_csv("https://github.com/user-attachments/files/23480587/census10k.csv.gz")
data_train, data_test = train_test_split(data, test_size=0.2)

# fit TabularARGN
argn = TabularARGN()
argn.fit(data_train)
```

### Sampling / Synthetic Data Generation

Generate new synthetic samples:

```python
# unconditional sampling
argn.sample(n_samples=1000)
```

Generate new synthetic samples conditionally:

```python
# prepare seed
seed_data = pd.DataFrame({
    "age": [25, 50],
    "education": ["Bachelors", "HS-grad"]
})

# conditional sampling
argn.sample(seed_data=seed_data)
```

### Imputation / Filling Gaps

Fill in missing values:

```python
# prepare demo data with missings
data_with_missings = data_test.head(300).reset_index(drop=True)
data_with_missings.loc[0:299, "age"] = pd.NA
data_with_missings.loc[0:199, "race"] = pd.NA
data_with_missings.loc[100:299, "income"] = pd.NA

# impute missing values each with a random sample
data_imputed = argn.impute(data_with_missings)

# impute missing values each with their point estimates
data_imputed = argn.impute(data_with_missings, n_draws=100)

```

### Predictions / Classification

Predict any categorical target column:

```python
from sklearn.metrics import accuracy_score, roc_auc_score

# predict class labels for a categorical
predictions = argn.predict(data_test, target="income", n_draws=100, agg_fn="mode")

# evaluate performance
accuracy = accuracy_score(data_test["income"], predictions)
auc = roc_auc_score(data_test["income"], probabilities[:, 1])
print(f"Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
```

### Predictions / Regression

Predict any numerical target column:

```python
from sklearn.metrics import mean_absolute_error

# predict target values
predictions = argn.predict(data_test, target="age", n_draws=10, agg_fn="mean")

# evaluate performance
mae = mean_absolute_error(data_test["age"], predictions)
print(f"MAE: {mae:.1f} years")
```

### Conditional Probabilities

Assess any marginal conditional probability, for one or more target columns:

```python
# extract class probabilities for a categorical
argn.predict_proba(
    X=pd.DataFrame({
        "age": [25, 30, 35],
        "sex": ["Male", "Female", "Male"],
    }),
    target="income"
)

# extract bin probabilities for a numerical
argn.predict_proba(
    X=pd.DataFrame({
        # "age": [25, 30, 35],
        "sex": ["Male", "Female", "Male"],
        "occupation": ["Craft-repair", "Craft-repair", "Craft-repair"]
    }),
    target="capital_gain"
)

# extract two-way marginals
argn.predict_proba(
    X=data_test[["age", "race"]],
    target=["sex", "income"]
)
```

### Log Probability

Compute log likelihood of observations:

```python
# compute log probability for each observation
log_probs = argn.log_prob(data_test)

# list top 10 outliers
data_test.iloc[log_probs.argsort()[:10]]
```

## TabularARGN for Sequential Data

For sequential data (e.g., time series or event logs), specify the context key:

### Model Training - With Context Data

```python
import pandas as pd
from mostlyai.engine import TabularARGN

# load sequential data
tgt_data = pd.read_csv("https://github.com/user-attachments/files/23480787/batting.csv.gz")
ctx_data = pd.read_csv("https://github.com/user-attachments/files/23480786/players.csv.gz")

# fit TabularARGN with a context key column
argn = TabularARGN(
    tgt_context_key="players_id",
    ctx_primary_key="id",
    ctx_data=ctx_data,
    max_training_time=2,  # 2 minutes
    verbose=0,
)
argn.fit(tgt_data)
```

### Sampling / Synthetic Data Generation

Generate new synthetic samples (using existing context):
```python
argn.sample(n_samples=5)
```

Generate new synthetic samples conditionally (using custom context and seed):

```python
ctx_data = pd.DataFrame({
    "id": ["Player1", "Player2"],
    "weight": [170, 160],
    "height": [70, 68],
    "bats": ["R", "L"],
    "throws": ["R", "L"],
})
argn.sample(ctx_data=ctx_data)
```

## Basic Usage of LanguageModel

The `LanguageModel` class provides a scikit-learn-compatible interface for working with semi-structured textual data. It leverages pre-trained language models or trains lightweight LSTM models from scratch to generate synthetic text data.

**Note**: The default model is `MOSTLY_AI/LSTMFromScratch-3m`, a lightweight LSTM model trained from scratch (GPU strongly recommended). You can also use pre-trained HuggingFace models by setting model to e.g. `microsoft/phi-1.5` (GPU required).

### Model Training

Load your data and train the model:

```python
import pandas as pd
from mostlyai.engine import LanguageModel

# load data
data = pd.read_csv("https://github.com/user-attachments/files/23486562/airbnb20k.csv.gz")

# fit LanguageModel
lm = LanguageModel(
    model="MOSTLY_AI/LSTMFromScratch-3m",
    tgt_encoding_types={
        'neighbourhood': 'LANGUAGE_CATEGORICAL',
        'title': 'LANGUAGE_TEXT',
    },
    max_training_time=10,  # 10 minutes
    verbose=1,
)
lm.fit(data)
```

### Sampling / Synthetic Text Generation

Generate new synthetic samples using the trained language model:

```python
# unconditional sampling
lm.sample(
    n_samples=100,
    sampling_temperature=0.8,
)
```

```python
# prepare seed
seed_data = pd.DataFrame({
    "neighbourhood": ["Westminster", "Hackney"],
})

# conditional sampling with seed values
lm.sample(
    seed_data=seed_data,
    sampling_temperature=0.8,
)
```

## Further Examples

Example notebooks demonstrating various use cases are available in the `examples` directory:
- TabularARGN for flat tabular data [![Run on Colab](https://img.shields.io/badge/Open%20in-Colab-blue?logo=google-colab)](https://colab.research.google.com/github/mostly-ai/mostlyai-engine/blob/main/examples/flat.ipynb)
- TabularARGN for sequential data [![Run on Colab](https://img.shields.io/badge/Open%20in-Colab-blue?logo=google-colab)](https://colab.research.google.com/github/mostly-ai/mostlyai-engine/blob/main/examples/language.ipynb)
- LanguageModel for textual data [![Run on Colab](https://img.shields.io/badge/Open%20in-Colab-blue?logo=google-colab)](https://colab.research.google.com/github/mostly-ai/mostlyai-engine/blob/main/examples/sequential.ipynb)
