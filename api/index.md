## Engine Reference

### mostlyai.engine.split

```python
split(
    tgt_data,
    *,
    ctx_data=None,
    tgt_primary_key=None,
    ctx_primary_key=None,
    tgt_context_key=None,
    model_type=None,
    tgt_encoding_types=None,
    ctx_encoding_types=None,
    n_partitions=1,
    trn_val_split=0.8,
    workspace_dir="engine-ws",
    update_progress=None
)
```

Splits the provided original data into training and validation sets, and stores these as partitioned Parquet files. This is a simplified version of `mostlyai-data`, tailored towards single- and two-table use cases, while requiring all data to be passed as DataFrames in memory.

Creates the following folder structure within the `workspace_dir`:

- `OriginalData/tgt-data`: Partitioned target data files.
- `OriginalData/tgt-meta`: Metadata files for target data.
- `OriginalData/ctx-data`: Partitioned context data files (if context is provided).
- `OriginalData/ctx-meta`: Metadata files for context data (if context is provided).

Parameters:

| Name                 | Type               | Description                                   | Default                                                                                                                              |
| -------------------- | ------------------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `tgt_data`           | `DataFrame`        | DataFrame containing the target data.         | *required*                                                                                                                           |
| `ctx_data`           | \`DataFrame        | None\`                                        | DataFrame containing the context data.                                                                                               |
| `tgt_primary_key`    | \`str              | None\`                                        | Primary key column name in the target data.                                                                                          |
| `ctx_primary_key`    | \`str              | None\`                                        | Primary key column name in the context data.                                                                                         |
| `tgt_context_key`    | \`str              | None\`                                        | Context key column name in the target data.                                                                                          |
| `model_type`         | \`str              | ModelType                                     | None\`                                                                                                                               |
| `tgt_encoding_types` | \`dict\[str, str   | ModelEncodingType\]                           | None\`                                                                                                                               |
| `ctx_encoding_types` | \`dict\[str, str   | ModelEncodingType\]                           | None\`                                                                                                                               |
| `n_partitions`       | `int`              | Number of partitions to split the data into.  | `1`                                                                                                                                  |
| `trn_val_split`      | \`float            | Callable\[[Series], tuple[Series, Series]\]\` | Fraction of data to use for training (0 < value < 1), or a callable that takes keys as input and returns (trn_keys, val_keys) tuple. |
| `workspace_dir`      | \`str              | Path\`                                        | Path to the workspace directory where files will be created.                                                                         |
| `update_progress`    | \`ProgressCallback | None\`                                        | A custom progress callback.                                                                                                          |

### mostlyai.engine.analyze

```python
analyze(
    *,
    value_protection=True,
    differential_privacy=None,
    workspace_dir="engine-ws",
    update_progress=None
)
```

Generates (privacy-safe) column-level statistics of the original data, that has been `split` into the workspace. This information is required for encoding the original as well as for decoding the generating data.

Creates the following folder structure within the `workspace_dir`:

- `ModelStore/tgt-stats/stats.json`: Column-level statistics for target data
- `ModelStore/ctx-stats/stats.json`: Column-level statistics for context data (if context is provided).

Parameters:

| Name               | Type               | Description                                         | Default                                                  |
| ------------------ | ------------------ | --------------------------------------------------- | -------------------------------------------------------- |
| `value_protection` | `bool`             | Whether to enable value protection for rare values. | `True`                                                   |
| `workspace_dir`    | \`str              | Path\`                                              | Path to workspace directory containing partitioned data. |
| `update_progress`  | \`ProgressCallback | None\`                                              | Optional callback to update progress during analysis.    |

### mostlyai.engine.encode

```python
encode(*, workspace_dir='engine-ws', update_progress=None)
```

Encodes data in the workspace that has already been split and analyzed.

Creates the following folder structure within the `workspace_dir`:

- `OriginalData/encoded-data`: Encoded data for training, stored as parquet files.

Parameters:

| Name              | Type               | Description | Default                        |
| ----------------- | ------------------ | ----------- | ------------------------------ |
| `workspace_dir`   | \`str              | Path\`      | Directory path for workspace.  |
| `update_progress` | \`ProgressCallback | None\`      | Callback for progress updates. |

### mostlyai.engine.train

```python
train(
    *,
    model=None,
    max_training_time=14400.0,
    max_epochs=100.0,
    batch_size=None,
    gradient_accumulation_steps=None,
    enable_flexible_generation=True,
    max_sequence_window=None,
    differential_privacy=None,
    model_state_strategy=ModelStateStrategy.reset,
    device=None,
    workspace_dir="engine-ws",
    update_progress=None,
    upload_model_data_callback=None
)
```

Trains a model with optional early stopping and differential privacy.

Creates the following folder structure within the `workspace_dir`:

- `ModelStore`: Trained model checkpoints and logs.

Parameters:

| Name                          | Type                        | Description                                                      | Default                                                                                                                                |
| ----------------------------- | --------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `model`                       | \`str                       | None\`                                                           | The identifier of the model to train. If tabular, defaults to MOSTLY_AI/Medium. If language, defaults to MOSTLY_AI/LSTMFromScratch-3m. |
| `max_training_time`           | \`float                     | None\`                                                           | Maximum training time in minutes. If None, defaults to 10 days.                                                                        |
| `max_epochs`                  | \`float                     | None\`                                                           | Maximum number of training epochs. If None, defaults to 100 epochs.                                                                    |
| `batch_size`                  | \`int                       | None\`                                                           | Per-device batch size for training and validation. If None, determined automatically.                                                  |
| `gradient_accumulation_steps` | \`int                       | None\`                                                           | Number of steps to accumulate gradients. If None, determined automatically.                                                            |
| `enable_flexible_generation`  | `bool`                      | Whether to enable flexible order generation. Defaults to True.   | `True`                                                                                                                                 |
| `max_sequence_window`         | \`int                       | None\`                                                           | Maximum sequence window for tabular sequential models. Only applicable for tabular models.                                             |
| `differential_privacy`        | \`DifferentialPrivacyConfig | dict                                                             | None\`                                                                                                                                 |
| `model_state_strategy`        | `ModelStateStrategy`        | Strategy for handling existing model state (reset/resume/reuse). | `reset`                                                                                                                                |
| `device`                      | \`device                    | str                                                              | None\`                                                                                                                                 |
| `workspace_dir`               | \`str                       | Path\`                                                           | Directory path for workspace. Training outputs are stored in ModelStore subdirectory.                                                  |
| `update_progress`             | \`ProgressCallback          | None\`                                                           | Callback function to report training progress.                                                                                         |
| `upload_model_data_callback`  | \`Callable                  | None\`                                                           | Callback function to upload model data during training.                                                                                |

### mostlyai.engine.generate

```python
generate(
    *,
    ctx_data=None,
    seed_data=None,
    sample_size=None,
    batch_size=None,
    sampling_temperature=1.0,
    sampling_top_p=1.0,
    device=None,
    rare_category_replacement_method=RareCategoryReplacementMethod.constant,
    rebalancing=None,
    imputation=None,
    fairness=None,
    workspace_dir="engine-ws",
    update_progress=None
)
```

Generates synthetic data from a trained model.

Creates the following folder structure within the `workspace_dir`:

- `SyntheticData`: Generated synthetic data, stored as parquet files.

Parameters:

| Name                               | Type                            | Description                                              | Default                                                                                     |
| ---------------------------------- | ------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `ctx_data`                         | \`DataFrame                     | None\`                                                   | Context data to be used for generation.                                                     |
| `seed_data`                        | \`DataFrame                     | None\`                                                   | Seed data to condition generation on fixed target columns.                                  |
| `sample_size`                      | \`int                           | None\`                                                   | Number of samples to generate. Defaults to number of original samples.                      |
| `batch_size`                       | \`int                           | None\`                                                   | Batch size for generation. If None, determined automatically.                               |
| `sampling_temperature`             | `float`                         | Sampling temperature. Higher values increase randomness. | `1.0`                                                                                       |
| `sampling_top_p`                   | `float`                         | Nucleus sampling probability threshold.                  | `1.0`                                                                                       |
| `device`                           | \`str                           | None\`                                                   | Device to run generation on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'. |
| `rare_category_replacement_method` | \`RareCategoryReplacementMethod | str\`                                                    | Method for handling rare categories. Only applicable for tabular models.                    |
| `rebalancing`                      | \`RebalancingConfig             | dict                                                     | None\`                                                                                      |
| `imputation`                       | \`ImputationConfig              | dict                                                     | None\`                                                                                      |
| `fairness`                         | \`FairnessConfig                | dict                                                     | None\`                                                                                      |
| `workspace_dir`                    | \`str                           | Path\`                                                   | Directory path for workspace.                                                               |
| `update_progress`                  | \`ProgressCallback              | None\`                                                   | Callback for progress updates.                                                              |

## Schema Reference

### mostlyai.engine.domain.DifferentialPrivacyConfig

The differential privacy configuration for training the model. If not provided, then no differential privacy will be applied.

Parameters:

| Name                       | Type    | Description                                                                                                                                                                                                                                                                      | Default                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_epsilon`              | \`float | None\`                                                                                                                                                                                                                                                                           | Specifies the maximum allowable epsilon value. If the training process exceeds this threshold, it will be terminated early. Only model checkpoints with epsilon values below this limit will be retained. If not provided, the training will proceed without early termination based on epsilon constraints.                                                                                                  |
| `delta`                    | `float` | The delta value for differential privacy. It is the probability of the privacy guarantee not holding. The smaller the delta, the more confident you can be that the privacy guarantee holds. This delta will be equally distributed between the analysis and the training phase. | `1e-05`                                                                                                                                                                                                                                                                                                                                                                                                       |
| `noise_multiplier`         | `float` | Determines how much noise while training the model with differential privacy. This is the ratio of the standard deviation of the Gaussian noise to the L2-sensitivity of the function to which the noise is added.                                                               | `1.5`                                                                                                                                                                                                                                                                                                                                                                                                         |
| `max_grad_norm`            | `float` | Determines the maximum impact of a single sample on updating the model weights during training with differential privacy. This is the maximum norm of the per-sample gradients.                                                                                                  | `1.0`                                                                                                                                                                                                                                                                                                                                                                                                         |
| `value_protection_epsilon` | \`float | None\`                                                                                                                                                                                                                                                                           | The DP epsilon of the privacy budget for determining the value ranges, which are gathered prior to the model training during the analysis step. Only applicable if value protection is True. Privacy budget will be equally distributed between the columns. For categorical we calculate noisy histograms and use a noisy threshold. For numeric and datetime we calculate bounds based on noisy histograms. |

### mostlyai.engine.domain.FairnessConfig

Configure a fairness objective for the table.

The generated synthetic data will maintain robust statistical parity between the target column and the specified sensitive columns. All these columns must be categorical.

Parameters:

| Name                | Type        | Description | Default    |
| ------------------- | ----------- | ----------- | ---------- |
| `target_column`     | `str`       |             | *required* |
| `sensitive_columns` | `list[str]` |             | *required* |

### mostlyai.engine.domain.ImputationConfig

Configure imputation. Imputed columns will suppress the sampling of NULL values.

Parameters:

| Name      | Type        | Description                             | Default    |
| --------- | ----------- | --------------------------------------- | ---------- |
| `columns` | `list[str]` | The names of the columns to be imputed. | *required* |

### mostlyai.engine.domain.ModelEncodingType

The encoding type used for model training and data generation.

- `AUTO`: Model chooses among available encoding types based on the column's data type.
- `TABULAR_CATEGORICAL`: Model samples from existing (non-rare) categories.
- `TABULAR_NUMERIC_AUTO`: Model chooses among 3 numeric encoding types based on the values.
- `TABULAR_NUMERIC_DISCRETE`: Model samples from existing discrete numerical values.
- `TABULAR_NUMERIC_BINNED`: Model samples from binned buckets, to then sample randomly within a bucket.
- `TABULAR_NUMERIC_DIGIT`: Model samples each digit of a numerical value.
- `TABULAR_CHARACTER`: Model samples each character of a string value.
- `TABULAR_DATETIME`: Model samples each part of a datetime value.
- `TABULAR_DATETIME_RELATIVE`: Model samples the relative difference between datetimes within a sequence.
- `TABULAR_LAT_LONG`: Model samples a latitude-longitude column. The format is "latitude,longitude".
- `LANGUAGE_TEXT`: Model will sample free text, using a LANGUAGE model.
- `LANGUAGE_CATEGORICAL`: Model samples from existing (non-rare) categories, using a LANGUAGE model.
- `LANGUAGE_NUMERIC`: Model samples from the valid numeric value range, using a LANGUAGE model.
- `LANGUAGE_DATETIME`: Model samples from the valid datetime value range, using a LANGUAGE model.

### mostlyai.engine.domain.ModelStateStrategy

The strategy of how any existing model states and training progress are to be handled.

- `RESET`: Start training from scratch. Overwrite any existing model states and training progress.
- `REUSE`: Reuse any existing model states, but start progress from scratch. Used for fine-tuning existing models.
- `RESUME`: Reuse any existing model states and progress. Used for continuing an aborted training.

### mostlyai.engine.domain.ModelType

The type of model.

- `TABULAR`: A generative AI model tailored towards tabular data, trained from scratch.
- `LANGUAGE`: A generative AI model build upon a (pre-trained) language model.

### mostlyai.engine.domain.RareCategoryReplacementMethod

Specifies how rare categories will be sampled. Only applicable if value protection has been enabled.

- `CONSTANT`: Replace rare categories by a constant `_RARE_` token.
- `SAMPLE`: Replace rare categories by a sample from non-rare categories.

### mostlyai.engine.domain.RebalancingConfig

Configure rebalancing.

Parameters:

| Name            | Type               | Description                                                                                                           | Default    |
| --------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------- | ---------- |
| `column`        | `str`              | The name of the column to be rebalanced.                                                                              | *required* |
| `probabilities` | `dict[str, float]` | The target distribution of samples values. The keys are the categorical values, and the values are the probabilities. | *required* |
