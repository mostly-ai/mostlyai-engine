# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import shutil
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.spatial.distance import euclidean

from mostlyai import engine

results_dir = Path("_SIMULATIONS")
workspace_dir = Path("ws-tabular-sequential")
engine.init_logging()

parser = argparse.ArgumentParser()
parser.add_argument("--max_epochs", type=int, default=5)
parser.add_argument("--branch", type=str, default="sequence-continuation")
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--dataset", type=str, default="toy")
parser.add_argument("--seed_data", type=str, default="first_step")
args = parser.parse_args()
branch = args.branch
max_epochs = args.max_epochs
random_state = args.random_state
dataset = args.dataset
seed_data = args.seed_data
experiment_name = (
    f"branch={branch}_max_epochs={max_epochs}_random_state={random_state}_dataset={dataset}_seed_data={seed_data}"
)
engine.set_random_state(random_state)

match dataset:
    case "baseball":
        url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/baseball"
        ctx = pd.read_csv(f"{url}/players.csv.gz")
        tgt = pd.read_csv(f"{url}/batting.csv.gz")
        tgt_context_key = "players_id"
        ctx_primary_key = "id"
    case "physio":
        url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/physio"
        ctx = pd.read_csv(f"{url}/patients.csv.gz")[:10000]
        tgt = pd.read_csv(f"{url}/measures.csv.gz")[["patients_id", "Temp", "Urine"]]
        tgt_context_key = "patients_id"
        ctx_primary_key = "id"
        tgt = tgt[tgt[tgt_context_key].isin(ctx[ctx_primary_key])]
    case "toy":
        tgt = pd.read_parquet("toy.parquet")
        ctx_primary_key = tgt_context_key = "id"
        ctx = tgt[[tgt_context_key]].drop_duplicates()
    case "toy-0-seqlens":
        tgt = pd.read_parquet("toy.parquet")
        ctx_primary_key = tgt_context_key = "id"
        ctx = tgt[[tgt_context_key]].drop_duplicates()
        n_0_seqlens = int(0.1 * len(ctx))  # 10% of the subjects have 0-length sequences
        ctx_0_seqlens = pd.DataFrame({ctx_primary_key: [str(uuid.uuid4()) for _ in range(n_0_seqlens)]})
        ctx = pd.concat([ctx, ctx_0_seqlens], ignore_index=True).sample(frac=1).reset_index(drop=True)
    case _:
        raise ValueError(f"Invalid dataset: {dataset}")

match (dataset, seed_data):
    case (dataset, "first_step"):
        seed_data = tgt.groupby(tgt_context_key, as_index=False).first()
    case _:
        seed_data = None

epoch_val_loss = {}


def update_progress(*args, **kwargs):
    message = kwargs.get("message") or {}
    val_loss = message.get("val_loss")
    if val_loss is not None:
        epoch = message.get("epoch")
        epoch_val_loss[epoch] = val_loss


engine.split(
    workspace_dir=workspace_dir,
    tgt_data=tgt,
    ctx_data=ctx,
    tgt_context_key=tgt_context_key,
    ctx_primary_key=ctx_primary_key,
    model_type="TABULAR",
)
engine.analyze(workspace_dir=workspace_dir, value_protection=False)
engine.encode(workspace_dir=workspace_dir)
engine.train(
    workspace_dir=workspace_dir,
    max_epochs=max_epochs,
    update_progress=update_progress,
)
engine.generate(workspace_dir=workspace_dir, seed_data=seed_data)
syn_data_path = workspace_dir / "SyntheticData"
syn = pd.read_parquet(syn_data_path)
ctx_ids = ctx[ctx_primary_key].unique()
tgt_seq_lens = pd.Series(0, index=ctx_ids)
tgt_seq_lens.update(tgt.groupby(tgt_context_key).size())
syn_seq_lens = pd.Series(0, index=ctx_ids)
syn_seq_lens.update(syn.groupby(tgt_context_key).size())
tgt_seq_lens_mean = tgt_seq_lens.mean()
syn_seq_lens_mean = syn_seq_lens.mean()
tgt_seq_lens_std = tgt_seq_lens.std()
syn_seq_lens_std = syn_seq_lens.std()
tgt_seq_lens_quantiles = np.quantile(tgt_seq_lens, np.arange(0, 1.1, 0.1), method="inverted_cdf")
syn_seq_lens_quantiles = np.quantile(syn_seq_lens, np.arange(0, 1.1, 0.1), method="inverted_cdf")
tgt_counts = tgt_seq_lens.value_counts().sort_index()
syn_counts = syn_seq_lens.value_counts().sort_index()
tgt_perc = tgt_counts / tgt_counts.sum() * 100
syn_perc = syn_counts / syn_counts.sum() * 100
fig, ax = plt.subplots(figsize=(7, 5))
width = 0.4
indices = sorted(set(tgt_perc.index).union(syn_perc.index))
indices = list(indices)
x = np.arange(len(indices))
tgt_vals = [tgt_perc.get(i, 0) for i in indices]
syn_vals = [syn_perc.get(i, 0) for i in indices]
ax.bar(x - width / 2, tgt_vals, width=width, color="skyblue", label="Original")
ax.bar(x + width / 2, syn_vals, width=width, color="salmon", label="Synthetic")
ax.set_xticks(x)
ax.set_xticklabels(indices)
ax.set_xlabel("Sequence Length")
ax.set_ylabel("Percentage")
# ax.set_xlim(-1, 32)
# ax.set_ylim(0, 30)
ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
ax.set_title("Sequence Length Distribution (%)")
ax.legend()
plt.tight_layout()
all_indices = sorted(set(tgt_counts.index).union(syn_counts.index))
tgt_aligned = np.array([tgt_counts.get(i, 0) for i in all_indices])
syn_aligned = np.array([syn_counts.get(i, 0) for i in all_indices])
distance = euclidean(tgt_aligned, syn_aligned)
experiment_dir = results_dir / experiment_name
shutil.rmtree(experiment_dir, ignore_errors=True)
experiment_dir.mkdir(parents=True, exist_ok=True)
syn.to_parquet(experiment_dir / "synthetic_data.parquet")
fig.savefig(experiment_dir / "sequence_length_distribution.png")
(experiment_dir / "sequence_length_distribution.json").write_text(
    json.dumps(
        {
            "distance": distance,
            "tgt_counts": tgt_counts.to_dict(),
            "syn_counts": syn_counts.to_dict(),
            "epoch_val_loss": epoch_val_loss,
            "tgt_seq_lens_mean": tgt_seq_lens_mean,
            "syn_seq_lens_mean": syn_seq_lens_mean,
            "tgt_seq_lens_std": tgt_seq_lens_std,
            "syn_seq_lens_std": syn_seq_lens_std,
            "tgt_seq_lens_quantiles": tgt_seq_lens_quantiles.tolist(),
            "syn_seq_lens_quantiles": syn_seq_lens_quantiles.tolist(),
        }
    )
)
print("Training Sequence Lengths (avg, std): ", tgt_seq_lens_mean, tgt_seq_lens_std)
print("Synthetic Sequence Lengths (avg, std): ", syn_seq_lens_mean, syn_seq_lens_std)
print("Training Sequence Lengths (quantiles): ", tgt_seq_lens_quantiles)
print("Synthetic Sequence Lengths (quantiles): ", syn_seq_lens_quantiles)
print(f"Distance: {distance:.3f}")
print(f"Epoch to Validation Loss: {epoch_val_loss}")
