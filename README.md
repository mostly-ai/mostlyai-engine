# Synthetic Data Engine 💎

![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai-engine)
[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai-engine/)
[![stats](https://pepy.tech/badge/mostlyai-engine)](https://pypi.org/project/mostlyai-engine/)
![license](https://img.shields.io/github/license/mostly-ai/mostlyai-engine)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mostlyai-engine)

[Documentation](https://mostly-ai.github.io/mostlyai-engine/) | [Technical Paper](https://arxiv.org/abs/2501.12012) | [Free Cloud Service](https://app.mostly.ai/)

Create high-fidelity privacy-safe synthetic data:

1. train a generative model once
    * fit to flat or sequential data
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

Two model classes are provided:

1. **TabularARGN**: For structured, flat or sequential, tabular data.
2. **LanguageModel**: For semi-structured flat textual tabular data.

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

The `TabularARGN` class provides a scikit-learn-compatible interface for working with structured tabular data. It can be used for synthetic data generation, classification, regression, density estimation, and imputation.

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

## Basic Usage of LanguageModel

The `LanguageModel` class provides a scikit-learn-compatible interface for working with text data. It leverages pre-trained language models or trains lightweight LSTM models from scratch to generate synthetic text data.

### Text Data

Load your data and train the model:

```python
import pandas as pd
from mostlyai.engine import LanguageModel

# load original data
trn_df = pd.read_parquet("https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/headlines/headlines.parquet")
trn_df = trn_df.sample(n=10_000, random_state=42)

# fit the model
lm = LanguageModel(
    tgt_encoding_types={
        'category': 'LANGUAGE_CATEGORICAL',
        'date': 'LANGUAGE_DATETIME',
        'headline': 'LANGUAGE_TEXT',
    },
    max_training_time=2,
    random_state=42,
)
lm.fit(trn_df)
```

#### Synthetic Text Generation

Generate new synthetic samples:

```python
# unconditional sampling
syn_data = lm.sample(n_samples=100)

# conditional sampling with seed values
syn_data = lm.sample(seed_data=pd.DataFrame({"category": ["business", "tech"]}))
```

**Note**: The default model is `"MOSTLY_AI/LSTMFromScratch-3m"`, a lightweight LSTM model trained from scratch (GPU recommended). You can also use pre-trained HuggingFace models by setting `model="microsoft/phi-1.5"` (GPU required).
