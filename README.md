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

Two models with these classes are available:

1. `TabularARGN`: For structured, flat or sequential tabular data.
   * `TabularARGNSampler`
   * `TabularARGNDensity`
   * `TabularARGNImputer`
   * `TabularARGNClassifier`
   * `TabularARGNRegressor`
2. `LanguageModel`: For semi-structured, flat textual tabular data.
   * `LanguageModelSampler`

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

# load original data
url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/census"
data = pd.read_csv(f"{url}/census.csv.gz")
trn_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

# fit the model
argn = TabularARGN(max_training_time=1, random_state=42)
argn.fit(trn_data)
```

#### Synthetic Data Generation

Use the trained model to generate new synthetic samples:

```python
from mostlyai.engine import TabularARGNSampler

# create sampler from trained model
sampler = TabularARGNSampler(argn, sampling_temperature=1.0)
```

Generate new (representative) synthetic samples:

```python
# unconditional sampling
sampler.sample(n_samples=1000)
```

Generate new synthetic samples conditionally:

```python
# conditional sampling with seed values
sampler.sample(seed_data=pd.DataFrame({
    "age": [25, 50],
    "education": ["Bachelors", "HS-grad"]
}))
```


#### Classification

Use the trained model for classification tasks:

```python
from sklearn.metrics import accuracy_score, roc_auc_score
from mostlyai.engine import TabularARGNClassifier

# create classifier from trained model
clf = TabularARGNClassifier(argn, target="income", n_draws=100)

# sample predictions
preds = clf.predict(val_data)
probs = clf.predict_proba(val_data)

# evaluate performance
accuracy = accuracy_score(val_data["income"], preds)
auc = roc_auc_score(val_data["income"], probs[:, 1])
print(f"Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
```

#### Regression

Use the trained model for regression tasks:

```python
from sklearn.metrics import mean_absolute_error
from mostlyai.engine import TabularARGNRegressor

# create regressor from trained model
reg = TabularARGNRegressor(argn, target="age", n_draws=100)

# sample predictions
preds = reg.predict(val_data)

# evaluate performance
mae = mean_absolute_error(val_data["age"], preds)
print(f"MAE: {mae:.1f} years")
```

#### Density Estimation

Compute log probabilities to detect outliers:

```python
from mostlyai.engine import TabularARGNDensity

# create classifier from trained model
density = TabularARGNDensity(argn)

# calculate log likelihood
log_probs = density.score_samples(data)

# determine biggest outlier
idx_outlier = log_probs.argmin()
data.iloc[idx_outlier]
```

#### Imputation

Fill in missing values using the trained model:

```python
from mostlyai.engine import TabularARGNImputer

# create imputer from trained model
imputer = TabularARGNImputer(argn)

# sample imputed data
imputed_data = imputer.transform(data_with_missings)

# OR: fit and impute in one go
imputer = TabularARGNImputer(max_training_time=1)
imputed_data = imputer.fit_transform(data_with_missings)
```

### Sequential Tables

For sequential data (e.g., time series or event logs), specify the context key:

```python
import pandas as pd
from mostlyai.engine import TabularARGN, TabularARGNSampler

# load sequential data
url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/baseball"
trn_df = pd.read_csv(f"{url}/batting.csv.gz")

# train model with context key
argn = TabularARGN(
    tgt_context_key="players_id",
    max_training_time=1,
    random_state=42,
)
argn.fit(trn_df)
```

Use the trained model to generate new samples:
```python
# unconditional sampling
sampler = TabularARGNSampler(argn)
sampler.sample(n_samples=100)
```

## Basic Usage of LanguageModel

The `LanguageModel` class provides a scikit-learn-compatible interface for working with semi-structured textual data. It leverages pre-trained language models or trains lightweight LSTM models from scratch to generate synthetic text data.

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
    model="MOSTLY_AI/LSTMFromScratch-3m",
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

Generate new synthetic samples using the trained language model:

```python
from mostlyai.engine import LanguageSampler

# create sampler from trained model
sampler = LanguageSampler(lm, sampling_temperature=0.5)
```

Generate new synthetic samples:

```python
# unconditional sampling
sampler.sample(n_samples=100)
```

```python
# conditional sampling with seed values
syn_data = sampler.sample(seed_data=pd.DataFrame({"category": ["business", "tech"]}))
```

**Note**: The default model is `"MOSTLY_AI/LSTMFromScratch-3m"`, a lightweight LSTM model trained from scratch (GPU recommended). You can also use pre-trained HuggingFace models by setting e.g. `model="microsoft/phi-1.5"` (GPU required).
