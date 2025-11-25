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

"""
TabularARGN

This module provides two models for tabular data:

- FlatModel: A model that learns to generate tabular data, one row at a time.
- SequentialModel: A model that learns to generate tabular data, one coherent sequence at a time.

Each model can learn to generate data given some context data, which can either consist of scalar or sequential data.
"""

import logging
from enum import Enum
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
from opacus.layers import DPLSTM
from torch import nn

from mostlyai.engine._common import (
    CTXFLT,
    CTXSEQ,
    RIDX_SUB_COLUMN_PREFIX,
    SLEN_SUB_COLUMN_PREFIX,
    get_columns_from_cardinalities,
    get_sub_columns_from_cardinalities,
    get_sub_columns_lookup,
    get_sub_columns_nested_from_cardinalities,
)
from mostlyai.engine._tabular.fairness import apply_fairness_transforms

_LOG = logging.getLogger(__name__)


class ModelSize(str, Enum):
    S = "S"
    M = "M"
    L = "L"


ModelSizeOrUnits = ModelSize | dict[str, list[int]]


def get_no_of_model_parameters(model) -> int:
    trainable_count = sum([np.prod(p.shape) for p in model.parameters()])
    return trainable_count


def get_model_units(model: nn.Module) -> dict[str, Any]:
    embedder_units = {
        name[name.find("embedder@") :]: module.embedding_dim
        for name, module in model.named_modules()
        if ".embedder@" in name
    }
    column_embedder_units = {
        name[name.find("column_embedder@") :]: module.out_features
        for name, module in model.named_modules()
        if ".column_embedder@" in name and not isinstance(module, nn.Identity)  # columns are compressed optionally
    }
    flat_context_units = {
        name[name.find("flat_context@") :]: [layer.out_features for layer in module]
        for name, module in model.named_modules()
        if ".flat_context@" in name
        if isinstance(module, nn.ModuleList)
    }
    sequential_context_units = {
        name[name.find("sequential_context@") :]: [module.hidden_size] * module.num_layers
        for name, module in model.named_modules()
        if ".sequential_context@" in name
        if isinstance(module, nn.LSTM) or isinstance(module, DPLSTM)
    }
    history_units = {
        name[name.find("history@") :]: [module.hidden_size] * module.num_layers
        for name, module in model.named_modules()
        if ".history@" in name
        if isinstance(module, nn.LSTM) or isinstance(module, DPLSTM)
    }
    regressor_units = {
        name[name.find("regressor@") :]: [layer.out_features for layer in module]
        for name, module in model.named_modules()
        if ".regressor@" in name
        if isinstance(module, nn.ModuleList)
    }
    heuristics_dict = (
        embedder_units
        | column_embedder_units
        | flat_context_units
        | sequential_context_units
        | history_units
        | regressor_units
    )
    return heuristics_dict


####################
###  HEURISTICS  ###
####################


def _embedding_heuristic(id: str, model_size: ModelSizeOrUnits, dim_input: int) -> int:
    if isinstance(model_size, dict):
        return model_size[id]
    model_size_output_dim = dict(
        S=max(10, int(2 * np.ceil(dim_input**0.15))),
        M=max(10, int(3 * np.ceil(dim_input**0.25))),
        L=max(10, int(4 * np.ceil(dim_input**0.33))),
    )
    return min(dim_input, model_size_output_dim[model_size])


def _column_embedding_heuristic(
    id: str,
    model_size: ModelSizeOrUnits,
    dim_input: int,
    n_sub_cols: int,
    compress_enabled: bool,
) -> int:
    if isinstance(model_size, dict):
        return model_size.get(id, dim_input)
    model_size_output_dim = dict(
        S=int(4 + n_sub_cols),
        M=int(10 + n_sub_cols),
        L=int(16 + n_sub_cols),
    )
    compress = compress_enabled and n_sub_cols > 2
    dim_output = model_size_output_dim[model_size] if compress else dim_input
    # dim_output should always be at most dim_input
    return min(dim_input, dim_output)


def _regressor_heuristic(id: str, model_size: ModelSizeOrUnits, dim_input: int, cardinality: int) -> list[int]:
    if isinstance(model_size, dict):
        return model_size[id]
    model_size_layers = dict(S=[4], M=[16], L=[16, 16])
    dims = [dim_input]
    layers = model_size_layers[model_size]
    # first layers depend on input dimension
    for idx, unit in enumerate(layers[:-1]):
        coefficient = round(np.log(max(dims[idx], np.e)))
        dims.append(unit * coefficient)
    # last layer depends on cardinality
    unit = layers[-1]
    coefficient = round(np.log(max(cardinality, np.e)))
    dims.append(unit * coefficient)
    return dims[1:]


def _flat_context_heuristic(id: str, model_size: ModelSizeOrUnits, dim_input: int) -> list[int]:
    if isinstance(model_size, dict):
        return model_size[id]
    model_size_layers = dict(S=[8], M=[64], L=[128])
    layers = model_size_layers[model_size]
    coefficient = round(np.log(max(dim_input, np.e)))
    dims = [unit * coefficient for unit in layers]
    return dims


def _sequential_context_heuristic(
    id: str, model_size: ModelSizeOrUnits, dim_input: int, seq_len_median: int
) -> list[int]:
    if isinstance(model_size, dict):
        return model_size[id]
    model_size_layers = dict(S=[8], M=[32], L=[64, 64])
    layers = model_size_layers[model_size]
    coefficient = round(np.log(max(dim_input * seq_len_median, np.e)))
    dims = [unit * coefficient for unit in layers]
    return dims


def _history_heuristic(id: str, model_size: ModelSizeOrUnits, dim_input: int, seq_len_median: int) -> list[int]:
    if isinstance(model_size, dict):
        return model_size[id]
    model_size_layers = dict(S=[16], M=[64], L=[128, 128])
    layers = model_size_layers[model_size]
    coefficient = round(np.log(max(dim_input * seq_len_median, np.e)))
    dims = [unit * coefficient for unit in layers]
    return dims


#######################
### BUILDING BLOCKS ###
#######################


class Embedders(nn.Module):
    def __init__(
        self,
        model_size: ModelSizeOrUnits,
        cardinalities: dict[str, int],
        device: torch.device,
    ):
        super().__init__()

        self.model_size = model_size
        self.cardinalities = cardinalities
        self.device = device

        self.dims = []

        self.embedders = nn.ModuleDict()

        # embedding layers for each sub column defined in cardinalities
        has_ridx = any(sub_col.startswith(RIDX_SUB_COLUMN_PREFIX) for sub_col in self.cardinalities)
        last_slen_sub_col = next(
            (
                sub_col
                for sub_col in reversed(self.cardinalities)
                if sub_col.startswith(SLEN_SUB_COLUMN_PREFIX)
                if has_ridx  # last SLEN sub column is dangling for model with RIDX only
            ),
            None,
        )
        for sub_col, dim_input in self.cardinalities.items():
            dim_output = _embedding_heuristic(id=self.id(sub_col), model_size=model_size, dim_input=dim_input)
            embedder = nn.Embedding(num_embeddings=dim_input, embedding_dim=dim_output, device=device)
            # the embeddings of the last SLEN sub column are never used
            # so we explicitly freeze them to make opacus not complain about "per sample gradient is not initialized"
            if sub_col == last_slen_sub_col:
                embedder.weight.requires_grad = False
            self.add(sub_column=sub_col, embedder=embedder)
            self.dims.append(dim_output)

    def __bool__(self):
        return bool(self.embedders)

    @staticmethod
    def id(sub_column: str) -> str:
        return f"embedder@{sub_column}"

    def get(self, sub_column: str) -> nn.Module:
        return self.embedders[self.id(sub_column)]

    def add(self, sub_column: str, embedder: nn.Module) -> None:
        self.embedders[self.id(sub_column)] = embedder

    def forward(self, x) -> dict[str, torch.Tensor]:
        # pass through sub column embedders
        embeddings = {}
        for sub_col in self.cardinalities:
            xs = torch.as_tensor(x[sub_col], device=self.device)
            if xs.is_nested:  # account for nested tensors
                xs = torch.nested.to_padded_tensor(xs, 0)
            xs = self.get(sub_col)(xs)
            xs = torch.squeeze(xs, -2)
            embeddings[sub_col] = xs
        return embeddings

    def zero_mask(self, *first_dims: int) -> dict[str, torch.Tensor]:
        # zero mask with shapes like embeddings
        embeddings = {}
        for idx, sub_col in enumerate(self.cardinalities):
            last_dim = (self.dims[idx],)
            shape = first_dims + last_dim
            embeddings[sub_col] = torch.zeros(shape, device=self.device)
        return embeddings


class SequentialContextEmbedders(Embedders):
    def __init__(
        self,
        model_size: ModelSizeOrUnits,
        cardinalities: dict[str, int],
        device: torch.device,
    ):
        super().__init__(model_size, cardinalities, device)

    def forward(self, x) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        # pass through sub column embedders
        embeddings = {}
        mask = None
        for sub_col in self.cardinalities:
            xs = torch.as_tensor(x[sub_col], device=self.device)
            xs = torch.nested.to_padded_tensor(xs, padding=-1)
            mask = (xs != -1).squeeze(-1)
            xs = torch.where(xs == -1, torch.tensor(0), xs)
            xs = self.get(sub_col)(xs)
            xs = torch.squeeze(xs, -2)
            embeddings[sub_col] = xs
        assert mask is not None
        return embeddings, mask


class ColumnEmbedders(nn.Module):
    def __init__(
        self,
        model_size: ModelSizeOrUnits,
        cardinalities: dict[str, int],
        embedding_dims: list[int],
        device: torch.device,
    ):
        super().__init__()

        self.model_size = model_size
        self.cardinalities = cardinalities
        self.embedding_dims = embedding_dims
        self.device = device

        self.column_sub_columns = get_sub_columns_nested_from_cardinalities(cardinalities, groupby="columns")

        self.dims = []

        self.column_embedders = nn.ModuleDict()

        # column embeddings for each column defined in cardinalities
        # only consider adding compressor layer if there are more than 50 sub columns
        compressor_enabled = len(self.cardinalities) > 50
        offset = 0
        for column, sub_cols in self.column_sub_columns.items():
            n_sub_cols = len(sub_cols)
            sub_cols_dims = self.embedding_dims[offset : offset + n_sub_cols]
            dim_input = sum(sub_cols_dims)
            dim_output = _column_embedding_heuristic(
                id=self.id(column),
                model_size=self.model_size,
                dim_input=dim_input,
                n_sub_cols=n_sub_cols,
                compress_enabled=compressor_enabled,
            )
            column_embedder = (
                nn.Linear(in_features=dim_input, out_features=dim_output, device=self.device)
                # apply compression only if dim_output < dim_input
                if dim_output < dim_input
                else nn.Identity()
            )
            self.add(column=column, column_embedder=column_embedder)
            self.dims.append(dim_output)
            offset += n_sub_cols

    @staticmethod
    def id(column: str) -> str:
        return f"column_embedder@{column}"

    def get(self, column: str) -> nn.Module:
        return self.column_embedders[self.id(column)]

    def add(self, column: str, column_embedder: nn.Module) -> None:
        self.column_embedders[self.id(column)] = column_embedder

    def forward(self, x) -> dict[str, torch.Tensor]:
        # pass through column embedders
        column_embeddings = {}
        for column, sub_cols in self.column_sub_columns.items():
            sub_cols_embeddings = torch.cat([x[sub_col] for sub_col in sub_cols], dim=-1)
            column_embeddings[column] = self.get(column)(sub_cols_embeddings)

        return column_embeddings

    def zero_mask(self, *first_dims: int) -> dict[str, torch.Tensor]:
        # zero mask with shapes like column embeddings
        column_embeddings = {}
        for idx, column in enumerate(self.column_sub_columns):
            last_dim = (self.dims[idx],)
            shape = first_dims + last_dim
            column_embeddings[column] = torch.zeros(shape, device=self.device)
        return column_embeddings


class FlatContextCompressor(nn.Module):
    def __init__(
        self,
        model_size: ModelSizeOrUnits,
        ctxflt_cardinalities: dict[str, int],
        device: torch.device,
    ):
        super().__init__()

        self.model_size = model_size
        self.ctxflt_cardinalities = ctxflt_cardinalities
        self.device = device

        self.compressor_layers = nn.ModuleDict()
        self.dropout = nn.Dropout(p=0.25)

        # flat context embedding layers
        self.embedders = Embedders(
            model_size=self.model_size,
            cardinalities=self.ctxflt_cardinalities,
            device=self.device,
        )

        # flat context compressor layers
        self.dim_output = 0
        if self.embedders:
            dim_input = sum(self.embedders.dims)
            dims = _flat_context_heuristic(id=self.id(), model_size=model_size, dim_input=dim_input)
            compressor_layers = nn.ModuleList()
            for dim_in, dim_out in zip([dim_input] + dims[:-1], dims):
                compressor_layers.append(nn.Linear(in_features=dim_in, out_features=dim_out, device=device))
            self.set(compressor_layers)
            self.dim_output = dims[-1]

    @staticmethod
    def id() -> str:
        return "flat_context@"

    def get(self) -> nn.ModuleList:
        return self.compressor_layers[self.id()]

    def set(self, compressor: nn.ModuleList) -> None:
        self.compressor_layers[self.id()] = compressor

    def forward(self, x) -> list[torch.Tensor]:
        # pass through flat context embeddings
        embeddings = self.embedders(x)

        # pass through multi-layer flat context compressor
        flat_context = []
        if embeddings:
            xs = torch.cat(list(embeddings.values()), dim=-1)
            for compressor_layer in self.get():
                xs = compressor_layer(xs)
                xs = self.dropout(xs)
                flat_context = [xs]
        return flat_context


class SequentialContextCompressor(nn.Module):
    def __init__(
        self,
        model_size: ModelSizeOrUnits,
        ctxseq_cardinalities: dict[str, int],
        ctxseq_len_median: dict[str, int],
        device: torch.device,
        with_dp: bool = False,
    ):
        super().__init__()

        self.model_size = model_size
        self.ctxseq_cardinalities = ctxseq_cardinalities
        self.ctxseq_len_median = ctxseq_len_median
        self.device = device
        dropout_rate = 0.25

        self.ctxseq_table_sub_columns = get_sub_columns_nested_from_cardinalities(
            self.ctxseq_cardinalities, groupby="tables"
        )

        self.embedders = nn.ModuleDict()
        self.compressor_layers = nn.ModuleDict()
        self.dropout = nn.Dropout(p=dropout_rate)

        self.dim_outputs = []
        for table, sub_columns in self.ctxseq_table_sub_columns.items():
            # sequential context table embedding layers
            table_cardinalities = {
                sub_col: card for sub_col, card in self.ctxseq_cardinalities.items() if sub_col in sub_columns
            }
            table_embedders = SequentialContextEmbedders(
                model_size=self.model_size,
                cardinalities=table_cardinalities,
                device=self.device,
            )
            self.embedders[table] = table_embedders

            # sequential context table compressor layers
            dim_input = sum(table_embedders.dims)
            dims = _sequential_context_heuristic(
                id=self.id(table),
                model_size=self.model_size,
                dim_input=dim_input,
                seq_len_median=ctxseq_len_median.get(table, 1),
            )
            hidden_size = dims[-1]
            num_layers = len(dims)
            bidirectional = True
            lstm_cls = DPLSTM if with_dp else partial(nn.LSTM, device=self.device)
            table_compressor_layer = lstm_cls(
                input_size=dim_input,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout_rate if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.add(table=table, compressor_layer=table_compressor_layer)
            self.dim_outputs.append(hidden_size * (2 if bidirectional else 1))

    @staticmethod
    def id(table: str) -> str:
        return f"sequential_context@{table}"

    def get(self, table: str) -> nn.Module:
        return self.compressor_layers[self.id(table)]

    def add(self, table: str, compressor_layer: nn.Module) -> None:
        self.compressor_layers[self.id(table)] = compressor_layer

    def forward(self, x) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        sequential_contexts = []
        sequential_context_masks = []
        for table, _ in self.ctxseq_table_sub_columns.items():
            # pass through sequential context table embeddings
            table_embeddings, mask = self.embedders[table](x)
            sequential_context_masks.append(mask)
            # pass through multi-layer sequential context table compressor
            xs = torch.cat(list(table_embeddings.values()), dim=-1)
            compressor_layer = self.get(table)
            xs = self.dropout(xs)
            lengths = torch.sum(mask, dim=-1)
            packed_xs = nn.utils.rnn.pack_padded_sequence(xs, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_xs, _ = compressor_layer(packed_xs)
            xs, _ = nn.utils.rnn.pad_packed_sequence(packed_xs, batch_first=True)

            # we return the full sequence of hidden states
            sequential_contexts.append(xs)

        return sequential_contexts, sequential_context_masks


class ContextCompressor(nn.Module):
    def __init__(
        self,
        model_size: ModelSizeOrUnits,
        ctx_cardinalities: dict[str, int],
        ctxseq_len_median: dict[str, int],
        device: torch.device,
        with_dp: bool = False,
    ):
        super().__init__()
        self.model_size = model_size
        self.ctxflt_cardinalities = {k: v for k, v in ctx_cardinalities.items() if k.startswith(CTXFLT)}
        self.ctxseq_cardinalities = {k: v for k, v in ctx_cardinalities.items() if k.startswith(CTXSEQ)}
        self.ctxseq_len_median = ctxseq_len_median
        self.device = device

        # flat context
        self.flat_context_compressor = FlatContextCompressor(
            model_size=self.model_size,
            ctxflt_cardinalities=self.ctxflt_cardinalities,
            device=device,
        )

        # sequential context(s)
        self.sequential_context_compressor = SequentialContextCompressor(
            model_size=self.model_size,
            ctxseq_cardinalities=self.ctxseq_cardinalities,
            ctxseq_len_median=self.ctxseq_len_median,
            device=device,
            with_dp=with_dp,
        )

        # size of context output
        self.dim_output = self.flat_context_compressor.dim_output + sum(self.sequential_context_compressor.dim_outputs)

    def forward(self, x) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        flat_context = self.flat_context_compressor(x)
        sequential_contexts, sequential_context_masks = self.sequential_context_compressor(x)
        return flat_context, sequential_contexts, sequential_context_masks


class HistoryCompressor(nn.Module):
    def __init__(
        self,
        model_size: ModelSizeOrUnits,
        embedding_dims: list[int],
        seq_len_median: int,
        device: torch.device,
        with_dp: bool = False,
    ):
        super().__init__()

        self.model_size = model_size
        self.embedding_dims = embedding_dims
        self.seq_len_median = seq_len_median
        self.device = device
        dropout_rate = 0.25

        self.compressor_layers = nn.ModuleDict()
        self.dropout = nn.Dropout(p=dropout_rate)

        dim_input = sum(embedding_dims)
        dims = _history_heuristic(
            id=self.id(),
            model_size=self.model_size,
            dim_input=dim_input,
            seq_len_median=self.seq_len_median,
        )

        hidden_size = dims[-1]
        num_layers = len(dims)
        lstm_cls = DPLSTM if with_dp else nn.LSTM
        compressor_layer = lstm_cls(
            input_size=dim_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.set(compressor_layer)
        self.dim_output = hidden_size

    @staticmethod
    def id() -> str:
        return "history@"

    def get(self) -> nn.Module:
        return self.compressor_layers[self.id()]

    def set(self, compressor_layer: nn.Module) -> None:
        self.compressor_layers[self.id()] = compressor_layer

    def forward(self, x, history_state=None) -> tuple[torch.Tensor, torch.Tensor]:
        compressor_layer = self.get()
        x = self.dropout(x)
        x, history_state = compressor_layer(x, history_state)
        return x, history_state


class Regressors(nn.Module):
    def __init__(
        self,
        *,
        model_size: ModelSizeOrUnits,
        cardinalities: dict[str, int],
        context_dim: int,
        history_dim: int | None = None,
        embedding_dims: list[int],
        column_embedding_dims: list[int],
        device: torch.device,
    ):
        super().__init__()

        self.model_size = model_size
        self.cardinalities = cardinalities
        self.context_dim = context_dim
        self.history_dim = history_dim
        self.column_embedding_dims = column_embedding_dims
        self.embedding_dims = embedding_dims
        self.device = device

        self.column_sub_columns = get_sub_columns_nested_from_cardinalities(self.cardinalities, groupby="columns")
        self.sub_columns_lookup = get_sub_columns_lookup(self.column_sub_columns)

        self.dims_output = {}

        self.regressors = nn.ModuleDict()
        self.dropout = nn.Dropout(p=0.25)

        for sub_column, lookup in self.sub_columns_lookup.items():
            # collect previous sub column embedding dims for current column
            prev_embedding_dims = self.embedding_dims[lookup.sub_col_offset : lookup.sub_col_cum]

            dim_input = (
                self.context_dim
                + (self.history_dim or 0)  # history is not present in FlatModel
                + sum(self.column_embedding_dims)
                + sum(prev_embedding_dims)
            )

            # multi-layer sub column regressor
            dims = _regressor_heuristic(
                id=self.id(sub_column),
                model_size=self.model_size,
                dim_input=dim_input,
                cardinality=self.cardinalities[sub_column],
            )
            regressor_layers = nn.ModuleList()
            for dim_in, dim_out in zip([dim_input] + dims[:-1], dims):
                regressor_layers.append(nn.Linear(in_features=dim_in, out_features=dim_out, device=self.device))
            self.add(sub_column=sub_column, regressor=regressor_layers)
            self.dims_output[sub_column] = dims[-1]

    @staticmethod
    def id(sub_column: str) -> str:
        return f"regressor@{sub_column}"

    def get(self, sub_column: str) -> nn.ModuleList:
        return self.regressors[self.id(sub_column)]

    def add(self, sub_column: str, regressor: nn.ModuleList) -> None:
        self.regressors[self.id(sub_column)] = regressor

    def forward(self, regressor_in: list[torch.Tensor], sub_col: str) -> torch.Tensor:
        x = torch.cat(regressor_in, dim=-1)
        # pass through multi-layer sub column regressor
        for regressor_layer in self.get(sub_col):
            x = self.dropout(x)
            x = regressor_layer(x)
        x = nn.ReLU()(x)
        return x


class Predictors(nn.Module):
    def __init__(
        self,
        cardinalities: dict[str, int],
        regressors_dims: dict[str, int],
        device: torch.device,
        empirical_probs: dict[str, np.ndarray] | None = None,
    ):
        super().__init__()

        self.cardinalities = cardinalities
        self.regressors_dims = regressors_dims
        self.device = device

        self.predictors = nn.ModuleDict()
        empirical_probs = empirical_probs or {}
        if empirical_probs:
            _LOG.info("initializing predictor bias with empirical log probabilities")

        for sub_col, dim_output in self.cardinalities.items():
            dim_input = self.regressors_dims[sub_col]
            self.predictors[sub_col] = nn.Linear(in_features=dim_input, out_features=dim_output, device=self.device)
            if empirical_probs:
                nn.init.xavier_uniform_(self.predictors[sub_col].weight)
                with torch.no_grad():
                    self.predictors[sub_col].bias.copy_(
                        torch.as_tensor(
                            np.log(empirical_probs[sub_col]),
                            dtype=self.predictors[sub_col].bias.dtype,
                            device=device,
                        )
                    )

    def forward(self, x: torch.Tensor, sub_col: str) -> torch.Tensor:
        return self.predictors[sub_col](x)


def _make_permutation_mask(
    col_embedding_dims: list[int],
    columns: list[str],
    column_order: list[str] | None,
    is_sequential: bool,
    device: torch.device,
) -> torch.Tensor:
    n_cols = len(columns)
    if column_order is not None:
        # create mask in provided order
        order = torch.tensor([columns.index(c) for c in column_order], dtype=torch.int32)
    elif is_sequential and n_cols >= 1:
        # create mask in random order, but keep positional columns at first position
        order = torch.randperm(n_cols - 1) + 1
        order = torch.cat((torch.zeros(1, dtype=torch.int32), order), dim=0)
    else:
        # create mask in random order
        order = torch.randperm(n_cols)

    # convert order into a binary mask consisting of 0s and 1s
    idx = torch.argsort(order)
    ones = torch.ones(n_cols, n_cols, dtype=torch.int32, device=device)
    mask = torch.tril(ones, diagonal=-1)  # strict lower triangular matrix
    mask = mask[idx, :][:, idx].bool()  # re-order rows and columns
    reps = torch.as_tensor(col_embedding_dims, dtype=torch.int32, device=device)
    mask = torch.repeat_interleave(mask, repeats=reps, dim=1)  # expand columns
    return mask


def _sampling_temperature(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = torch.max(torch.tensor(temperature, device=probs.device), torch.tensor(1e-3, device=probs.device))
    # compute softmax with temperature scaling
    scaled_logits = torch.divide(torch.log(probs), temperature)
    probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
    return probs


def _sampling_nucleus(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Implements nucleus sampling (https://arxiv.org/abs/1904.09751)
    Parameters:
        probs: probabilities of tokens to sample from
        top_p: cumulative probability of top p tokens to sample from
    """
    # sort probabilities in descending order and get their indices
    sorted_indices = torch.argsort(input=probs, descending=True, dim=-1)
    sorted_probs = torch.gather(input=probs, dim=-1, index=sorted_indices)

    # compute the cumulative sum of the sorted probabilities
    cumulative_probs = torch.cumsum(input=sorted_probs, dim=-1)

    # remove tokens whose cumulative probability exceeds the threshold
    sorted_indices_to_remove = torch.greater(cumulative_probs, top_p)

    # shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    # we need at least one token to sample from
    sorted_indices_to_remove[..., 0] = False

    # scatter sorted tensors to original indexing
    _, reversed_indices = torch.sort(input=sorted_indices, dim=-1)
    indices_to_remove = torch.gather(input=sorted_indices_to_remove, dim=-1, index=reversed_indices)

    # zero out probabilities that need to be removed
    probs = probs.masked_fill(mask=indices_to_remove, value=0.0)

    # re-normalize to [0, 1]
    # if all probs are zero (first sorted / biggest is zero), fallback to uniform sampling
    sums = torch.sum(probs, dim=-1, keepdim=True)
    probs = torch.where(sums > 0.0, probs / sums, torch.ones_like(probs) / probs.size(-1))

    return probs


def _sampling_fixed_probs(probs: torch.Tensor, fixed_probs: dict[int, float]) -> torch.Tensor:
    # adding small epsilon avoids edge case where non-fixed values are all zero
    probs = probs + 1.0e-20

    # identify fixed and non-fixed indices
    fixed_indices = torch.as_tensor(list(fixed_probs.keys()), device=probs.device)
    fixed_values = torch.as_tensor(list(fixed_probs.values()), device=probs.device)
    fixed_mask = torch.zeros(probs.size(-1), dtype=torch.bool, device=probs.device)
    fixed_mask[fixed_indices] = True
    non_fixed_mask = ~fixed_mask

    # handle edge case: all indices are fixed to 0.0
    if fixed_mask.all() and fixed_values.sum() == 0.0:
        probs[:] = 1.0 / probs.size(-1)
        return probs

    # overwrite probs under fixed indices
    probs[..., fixed_indices] = fixed_values.unsqueeze(0)

    # normalize non-fixed indices
    remaining_prob = torch.clamp(1.0 - fixed_values.sum(), min=0.0)
    non_fixed_sum = torch.sum(probs[..., non_fixed_mask], dim=-1, keepdim=True)
    probs[..., non_fixed_mask] = probs[..., non_fixed_mask] * (remaining_prob / non_fixed_sum)

    return probs


def _sample(
    probs: torch.Tensor,
    temperature: float | None = None,
    top_p: float | None = None,
    fixed_probs: dict[int, float] | None = None,
) -> torch.Tensor:
    if temperature is not None and temperature != 1.0:
        probs = _sampling_temperature(probs, temperature)

    if top_p is not None and top_p < 1.0:
        probs = _sampling_nucleus(probs, top_p)

    if fixed_probs is not None:
        probs = _sampling_fixed_probs(probs, fixed_probs)

    # ensure that probabilities are valid
    # log cases where this isn't the case to help understand when this can happen
    is_nan = torch.isnan(probs)
    is_neg = probs < 0.0
    is_gt1 = probs > 1.0
    if is_nan.any():
        nan_count = is_nan.sum().item()
        _LOG.warning(f"[ARGN] Set {nan_count} probabilities from NaN to 1 in {probs.shape}")
        probs = torch.nan_to_num(probs, nan=1.0)
    if is_neg.any():
        neg_count = is_neg.sum().item()
        _LOG.warning(f"[ARGN] Clip {neg_count} probabilities below 0 in {probs.shape}")
        probs = torch.clamp(probs, min=0.0)
    if is_gt1.any():
        gt1_count = is_gt1.sum().item()
        _LOG.warning(f"[ARGN] Clip {gt1_count} probabilities above 1 in {probs.shape}")
        probs = torch.clamp(probs, max=1.0)

    probs = torch.multinomial(probs, num_samples=1, replacement=True)
    return probs


##########################
### TabularARGN models ###
##########################


class FlatModel(nn.Module):
    def __init__(
        self,
        tgt_cardinalities: dict[str, int],
        ctx_cardinalities: dict[str, int],
        ctxseq_len_median: dict[str, int],
        model_size: ModelSizeOrUnits,
        column_order: list[str] | None,
        device: torch.device,
        with_dp: bool = False,
        empirical_probs_for_predictor_init: dict[str, np.ndarray] | None = None,
    ):
        super().__init__()

        self.tgt_cardinalities = tgt_cardinalities
        self.tgt_sub_columns = get_sub_columns_from_cardinalities(tgt_cardinalities)
        self.tgt_columns = get_columns_from_cardinalities(tgt_cardinalities)
        self.tgt_column_sub_columns = get_sub_columns_nested_from_cardinalities(tgt_cardinalities, groupby="columns")
        self.last_sub_cols = [v[-1] for v in self.tgt_column_sub_columns.values()]
        self.tgt_sub_columns_lookup = get_sub_columns_lookup(self.tgt_column_sub_columns)
        self.ctx_cardinalities = ctx_cardinalities
        self.ctxseq_len_median = ctxseq_len_median
        self.model_size = model_size
        self.column_order = column_order
        self.device = device or torch.device("cpu")

        # context
        self.context_compressor = ContextCompressor(
            model_size=self.model_size,
            ctx_cardinalities=self.ctx_cardinalities,
            ctxseq_len_median=self.ctxseq_len_median,
            device=device,
            with_dp=with_dp,
        )

        # sub column embeddings
        self.embedders = Embedders(
            model_size=self.model_size,
            cardinalities=self.tgt_cardinalities,
            device=self.device,
        )

        # column embeddings
        self.column_embedders = ColumnEmbedders(
            model_size=self.model_size,
            cardinalities=self.tgt_cardinalities,
            embedding_dims=self.embedders.dims,
            device=self.device,
        )

        # regressors
        self.regressors = Regressors(
            model_size=self.model_size,
            cardinalities=self.tgt_cardinalities,
            context_dim=self.context_compressor.dim_output,
            embedding_dims=self.embedders.dims,
            column_embedding_dims=self.column_embedders.dims,
            device=self.device,
        )

        # predictors
        self.predictors = Predictors(
            cardinalities=self.tgt_cardinalities,
            regressors_dims=self.regressors.dims_output,
            device=device,
            empirical_probs=empirical_probs_for_predictor_init if not with_dp else None,
        )

    def _handle_context(
        self, context: tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
    ) -> list[torch.Tensor]:
        flat_context, sequential_contexts, sequential_context_masks = context

        last_state_seq_ctxs = []
        for seq_ctx, mask in zip(sequential_contexts, sequential_context_masks):
            lengths = torch.sum(mask, dim=1)

            hidden_dim = seq_ctx.size(-1) // 2
            forward_states = seq_ctx[..., :hidden_dim]
            backward_states = seq_ctx[..., hidden_dim:]

            # Take last valid step from forward direction and first valid step from backward direction
            forward_final = torch.stack([forward_states[i, length - 1 : length, :] for i, length in enumerate(lengths)])
            backward_final = backward_states[:, 0:1, :]  # first timestep is last state of backward direction

            xs = torch.cat([forward_final, backward_final], dim=-1)
            xs = torch.squeeze(xs, dim=1)
            last_state_seq_ctxs.append(xs)

        context = flat_context + last_state_seq_ctxs

        if len(context) > 0:
            context = [torch.cat(context, dim=-1)]

        return context

    def forward(
        self,
        x,
        mode: Literal["trn", "gen", "probs"],
        batch_size: int | None = None,
        fixed_probs=None,
        fixed_values=None,
        temperature: float | None = None,
        top_p: float | None = None,
        fairness_transforms: dict[str, Any] | None = None,
        column_order: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the autoregressive model with three operational modes.

        Args:
            x: Input tensor or dict of tensors containing context features.
            mode: Operational mode determining behavior:
                - "trn": Training mode with parallel processing and teacher forcing.
                        Processes all columns simultaneously using ground truth values.
                        Returns logits for each sub-column.
                - "gen": Generation mode with sequential autoregressive sampling.
                        Processes columns one-by-one, sampling values from probability
                        distributions. Returns sampled integer codes for each sub-column.
                - "probs": Probability computation mode without sampling.
                        Processes columns sequentially up to target columns, computing
                        probability distributions without sampling. Returns raw probabilities
                        for target sub-columns specified by fixed_values/column_order.
            batch_size: Number of samples to generate (required for "gen" and "probs" modes).
            fixed_probs: Dict mapping sub-column names to probability distributions to use
                        instead of model predictions. Used for conditional generation.
            fixed_values: Dict mapping sub-column names to fixed integer values. These
                         columns will use the provided values instead of sampling. In "probs"
                         mode, these are typically seed/conditioning features.
            temperature: Sampling temperature for generation (>1.0 = more random, <1.0 = more
                        deterministic). Only used in "gen" mode.
            top_p: Nucleus sampling parameter (0.0-1.0). Samples from smallest set of tokens
                  whose cumulative probability exceeds p. Only used in "gen" mode.
            fairness_transforms: Dict of fairness adjustment functions to apply during
                               generation. Only used in "gen" mode.
            column_order: Custom ordering of columns to process. If None, uses default
                         training order. In "probs" mode, typically [seed_cols, target_col]
                         to only process necessary columns.

        Returns:
            Dict mapping sub-column names to tensors:
            - "trn" mode: Returns logits (pre-softmax values) for loss computation.
            - "gen" mode: Returns sampled integer codes.
            - "probs" mode: Returns probability distributions (post-softmax) for target columns.

        Notes:
            - In "trn" mode, all columns are processed in parallel with teacher forcing.
            - In "gen" mode, columns are processed sequentially with sampling and embedding updates.
            - In "probs" mode, fixed columns update embeddings but target columns only compute
              probabilities without sampling (since no subsequent columns depend on them).
            - Use column_order to optimize "probs" mode by only processing seed + target columns.

        Examples:
            >>> # Training mode
            >>> outputs = model(batch_data, mode="trn")
            >>>
            >>> # Generation mode
            >>> samples = model(
            ...     x=context,
            ...     mode="gen",
            ...     batch_size=100,
            ...     temperature=0.8,
            ...     fixed_values={"seed_col": seed_values}
            ... )
            >>>
            >>> # Probability computation mode
            >>> probs = model(
            ...     x=torch.zeros((5, 0)),
            ...     mode="probs",
            ...     batch_size=5,
            ...     fixed_values={"feature1": features, "feature2": features},
            ...     column_order=["feature1", "feature2", "target"]
            ... )
        """
        fixed_probs = fixed_probs or {}
        fixed_values = fixed_values or {}
        outputs = {}
        probs = {}
        fairness_transforms = fairness_transforms or {}
        effective_column_order = column_order or self.column_order

        if mode == "trn":
            # forward pass through context compressor
            context = self.context_compressor(x)
            context = self._handle_context(context)

            # forward pass through sub column embedders
            tgt_embeds = self.embedders(x)

            # forward pass through column embedders
            tgt_col_embeds = self.column_embedders(tgt_embeds)

            # create batch-wise permutation mask
            col_embeddings = torch.cat(list(tgt_col_embeds.values()), dim=-1)
            col_mask = _make_permutation_mask(
                col_embedding_dims=self.column_embedders.dims,
                columns=self.tgt_columns,
                column_order=effective_column_order,
                is_sequential=False,
                device=self.device,
            )

            for sub_col, lookup in self.tgt_sub_columns_lookup.items():
                # mask concatenated column embeddings
                masked_col_embeds = [torch.mul(col_mask[lookup.col_idx, :].int(), col_embeddings)]

                # collect previous sub column embeddings for current column
                prev_sub_col_embeds = [
                    tgt_embeds[sub_col] for sub_col in self.tgt_sub_columns[lookup.sub_col_offset : lookup.sub_col_cum]
                ]

                # regressor
                regressor_in = context + masked_col_embeds + prev_sub_col_embeds
                xs = self.regressors(regressor_in, sub_col)

                # predictor
                xs = self.predictors(xs, sub_col)

                # update output
                outputs[sub_col] = xs

            return outputs

        elif mode == "gen":
            # forward pass through context compressor
            context = self.context_compressor(x)
            context = self._handle_context(context)

            # initialize sub column embeddings
            tgt_embeds = self.embedders.zero_mask(batch_size)

            # initialize column embeddings
            tgt_col_embeds = self.column_embedders.zero_mask(batch_size)

            # concatenate column embeddings
            col_embeddings = torch.cat(list(tgt_col_embeds.values()), dim=-1)

            # take sub columns in the specified generation order
            column_order = effective_column_order or self.tgt_columns
            sub_column_order = [sub_col for col in column_order for sub_col in self.tgt_column_sub_columns[col]]

            for sub_col in sub_column_order:
                lookup = self.tgt_sub_columns_lookup[sub_col]

                # if sub column is fixed, skip sampling and use that value
                if sub_col in fixed_values:
                    out = fixed_values[sub_col]

                else:  # sample from distribution
                    # collect previous sub column embeddings for current column
                    prev_sub_col_embeds = [
                        tgt_embeds[sub_col]
                        for sub_col in self.tgt_sub_columns[lookup.sub_col_offset : lookup.sub_col_cum]
                    ]

                    # regressor
                    regressor_in = context + [col_embeddings] + prev_sub_col_embeds
                    xs = self.regressors(regressor_in, sub_col)

                    # predictor
                    xs = self.predictors(xs, sub_col)

                    # softmax to probs
                    xs = nn.Softmax(dim=-1)(xs)

                    # apply fairness transform when generating the target sub column
                    if fairness_transforms:
                        xs = apply_fairness_transforms(sub_col, xs, outputs, fairness_transforms)

                    # sample
                    out = torch.squeeze(
                        _sample(
                            probs=xs,
                            temperature=temperature,
                            top_p=top_p,
                            fixed_probs=fixed_probs.get(sub_col),
                        ),
                        dim=-1,
                    )

                # update output
                outputs[sub_col] = out

                # update current sub column embedding
                tgt_embeds[sub_col] = self.embedders.get(sub_col)(out)

                # update current column embedding
                if sub_col in self.last_sub_cols:
                    col_sub_cols = self.tgt_column_sub_columns[lookup.col_name]
                    col_embed_in = torch.cat([tgt_embeds[sc] for sc in col_sub_cols], dim=-1)
                    tgt_col_embeds[lookup.col_name] = self.column_embedders.get(lookup.col_name)(col_embed_in)
                    col_embeddings = torch.cat(list(tgt_col_embeds.values()), dim=-1)

            # order outputs according to tgt_sub_columns
            return {sub_col: outputs[sub_col] for sub_col in self.tgt_sub_columns}

        elif mode == "probs":
            # forward pass through context compressor
            context = self.context_compressor(x)
            context = self._handle_context(context)

            # initialize sub column embeddings
            tgt_embeds = self.embedders.zero_mask(batch_size)

            # initialize column embeddings
            tgt_col_embeds = self.column_embedders.zero_mask(batch_size)

            # concatenate column embeddings
            col_embeddings = torch.cat(list(tgt_col_embeds.values()), dim=-1)

            # take sub columns in the specified generation order
            column_order = effective_column_order or self.tgt_columns
            sub_column_order = [sub_col for col in column_order for sub_col in self.tgt_column_sub_columns[col]]

            for sub_col in sub_column_order:
                lookup = self.tgt_sub_columns_lookup[sub_col]

                # if sub column is fixed, use that value and update embeddings
                if sub_col in fixed_values:
                    out = fixed_values[sub_col]

                    # update current sub column embedding
                    tgt_embeds[sub_col] = self.embedders.get(sub_col)(out)

                    # update current column embedding
                    if sub_col in self.last_sub_cols:
                        col_sub_cols = self.tgt_column_sub_columns[lookup.col_name]
                        col_embed_in = torch.cat([tgt_embeds[sc] for sc in col_sub_cols], dim=-1)
                        tgt_col_embeds[lookup.col_name] = self.column_embedders.get(lookup.col_name)(col_embed_in)
                        col_embeddings = torch.cat(list(tgt_col_embeds.values()), dim=-1)

                else:  # compute probabilities without sampling
                    # collect previous sub column embeddings for current column
                    prev_sub_col_embeds = [
                        tgt_embeds[sub_col]
                        for sub_col in self.tgt_sub_columns[lookup.sub_col_offset : lookup.sub_col_cum]
                    ]

                    # regressor
                    regressor_in = context + [col_embeddings] + prev_sub_col_embeds
                    xs = self.regressors(regressor_in, sub_col)

                    # predictor
                    xs = self.predictors(xs, sub_col)

                    # softmax to probs
                    xs = nn.Softmax(dim=-1)(xs)

                    # store probabilities (no sampling, no embedding updates)
                    probs[sub_col] = xs

            return probs


class AttentionModule(nn.Module):
    def __init__(self, dim_input: int, dim_outputs: list[int], device: torch.device):
        super().__init__()
        self.dim_input = dim_input
        self.dim_outputs = dim_outputs
        self.device = device

        # when using the attention in SCP we need to project the input to have the same dim as the
        # context compressor so that we can compute inner products between the two
        self.q_projs = nn.ModuleList(
            [
                nn.Linear(
                    self.dim_input,
                    dim_output,
                    bias=False,
                    device=self.device,
                )
                for dim_output in self.dim_outputs
            ]
        )

    def forward(
        self,
        history: torch.Tensor,
        seq_ctxs: list[torch.Tensor],
        seq_ctx_masks: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        attn_ctxs = []
        batch_size, history_len, _ = history.shape

        for seq_ctx, mask, q_proj in zip(seq_ctxs, seq_ctx_masks, self.q_projs):
            _, ctx_len = mask.shape
            mask = mask.unsqueeze(1).expand(batch_size, history_len, ctx_len)
            query = q_proj(history)
            attn_ctx = torch.nn.functional.scaled_dot_product_attention(
                query=query,
                key=seq_ctx,
                value=seq_ctx,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            )  # (batch_size, history_len, hidden_dim)
            attn_ctxs.append(attn_ctx)
        if attn_ctxs:
            seq_ctxs = [torch.cat(attn_ctxs, dim=-1)]
        return seq_ctxs


class SequentialModel(nn.Module):
    def __init__(
        self,
        tgt_cardinalities: dict[str, int],
        tgt_seq_len_median: int,
        tgt_seq_len_max: int,
        ctx_cardinalities: dict[str, int],
        ctxseq_len_median: dict[str, int],
        model_size: ModelSizeOrUnits,
        column_order: list[str] | None,
        device: torch.device,
        with_dp: bool = False,
        empirical_probs_for_predictor_init: dict[str, np.ndarray] | None = None,
    ):
        super().__init__()

        self.tgt_cardinalities = tgt_cardinalities
        self.tgt_sub_columns = get_sub_columns_from_cardinalities(tgt_cardinalities)
        self.tgt_columns = get_columns_from_cardinalities(tgt_cardinalities)
        self.tgt_column_sub_columns = get_sub_columns_nested_from_cardinalities(tgt_cardinalities, groupby="columns")
        self.tgt_sub_columns_lookup = get_sub_columns_lookup(self.tgt_column_sub_columns)
        self.tgt_seq_len_median = tgt_seq_len_median
        self.tgt_last_sub_cols = [sub_cols[-1] for sub_cols in self.tgt_column_sub_columns.values()]

        self.model_size = model_size
        self.column_order = column_order
        self.tgt_seq_len_max = tgt_seq_len_max
        self.device = device or torch.device("cpu")
        self.with_dp = with_dp

        # context
        self.context_compressor = ContextCompressor(
            model_size=self.model_size,
            ctx_cardinalities=ctx_cardinalities,
            ctxseq_len_median=ctxseq_len_median,
            device=device,
            with_dp=with_dp,
        )

        # sub column embeddings
        self.embedders = Embedders(
            model_size=self.model_size,
            cardinalities=self.tgt_cardinalities,
            device=self.device,
        )

        # column embeddings
        self.column_embedders = ColumnEmbedders(
            model_size=self.model_size,
            cardinalities=self.tgt_cardinalities,
            embedding_dims=self.embedders.dims,
            device=self.device,
        )

        # history
        self.history_compressor = HistoryCompressor(
            model_size=self.model_size,
            embedding_dims=self.embedders.dims,
            seq_len_median=self.tgt_seq_len_median,
            device=self.device,
            with_dp=with_dp,
        )

        # attention module connecting sequential context(s) to history
        self.attention = AttentionModule(
            dim_input=self.history_compressor.dim_output,
            dim_outputs=self.context_compressor.sequential_context_compressor.dim_outputs,
            device=self.device,
        )

        # regressors
        self.regressors = Regressors(
            model_size=self.model_size,
            cardinalities=self.tgt_cardinalities,
            context_dim=self.context_compressor.dim_output,
            history_dim=self.history_compressor.dim_output,
            embedding_dims=self.embedders.dims,
            column_embedding_dims=self.column_embedders.dims,
            device=self.device,
        )

        # predictors
        self.predictors = Predictors(
            cardinalities=self.tgt_cardinalities,
            regressors_dims=self.regressors.dims_output,
            device=device,
            empirical_probs=empirical_probs_for_predictor_init if not with_dp else None,
        )

    def _repeat_flat_context(self, flat_ctx: list[torch.Tensor], repetition: int) -> list[torch.Tensor]:
        if len(flat_ctx) > 0:
            flat_ctx = flat_ctx[0].unsqueeze(1)
            flat_ctx = flat_ctx.repeat(1, repetition, 1)
            flat_ctx = [flat_ctx]
        return flat_ctx

    def forward(
        self,
        x,
        mode: Literal["trn", "gen"],
        batch_size: int | None = None,
        fixed_probs=None,
        fixed_values=None,
        temperature: float | None = None,
        top_p: float | None = None,
        history=None,
        history_state=None,
        context=None,
        column_order: list[str] | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        fixed_probs = fixed_probs or {}
        fixed_values = fixed_values or {}
        if context is None:
            context = self.context_compressor(x)

        effective_column_order = column_order or self.column_order

        has_ridx = any(sub_col.startswith(RIDX_SUB_COLUMN_PREFIX) for sub_col in self.tgt_cardinalities)

        # SLEN and RIDX are masked for history
        # NOTE: SLEN is not masked for models without RIDX (backwards compatibility)
        history_masked_sub_cols = (
            (SLEN_SUB_COLUMN_PREFIX, RIDX_SUB_COLUMN_PREFIX) if has_ridx else (RIDX_SUB_COLUMN_PREFIX,)
        )
        # SLEN is masked for column embeddings
        # NOTE: SLEN is not masked for models without RIDX (backwards compatibility)
        col_embeddings_masked_sub_cols = (SLEN_SUB_COLUMN_PREFIX,) if has_ridx else ()

        outputs = {}
        if mode == "trn":
            # forward pass through sub column embedders
            tgt_embeds = self.embedders(x)

            # history
            # time shift: remove last time step; add zeros for first time step; add randoms for all others
            tgt_embeds_for_history = {
                k: torch.zeros_like(v) if k.startswith(history_masked_sub_cols) else v for k, v in tgt_embeds.items()
            }
            embeddings = torch.cat(list(tgt_embeds_for_history.values()), dim=-1)
            history_in = embeddings[:, :-1, :]
            history_in = nn.ConstantPad2d((0, 0, 1, 0), 0)(history_in)
            history, _ = self.history_compressor(history_in)

            flat_ctx, seq_ctx, seq_ctx_masks = context
            # attention with history as query
            seq_ctx = self.attention(history, seq_ctx, seq_ctx_masks)
            # repeat flat context to match the length of history
            flat_ctx = self._repeat_flat_context(flat_ctx, history.size(1))
            context_history = [torch.cat(flat_ctx + seq_ctx + [history], -1)]

            # forward pass through column embedders
            tgt_embeds_for_col_embeddings = {
                k: torch.zeros_like(v) if k.startswith(col_embeddings_masked_sub_cols) else v
                for k, v in tgt_embeds.items()
            }
            tgt_col_embeds = self.column_embedders(tgt_embeds_for_col_embeddings)

            # create batch-wise permutation mask
            col_embeddings = torch.cat(list(tgt_col_embeds.values()), dim=-1)
            col_mask = _make_permutation_mask(
                col_embedding_dims=self.column_embedders.dims,
                columns=self.tgt_columns,
                column_order=effective_column_order,
                is_sequential=True,
                device=self.device,
            )

            for sub_col, lookup in self.tgt_sub_columns_lookup.items():
                # mask concatenated column embeddings
                masked_col_embeds = [torch.mul(col_mask[lookup.col_idx, :].int(), col_embeddings)]

                # collect previous sub column embeddings for current column
                prev_sub_col_embeds = {
                    sub_col: tgt_embeds[sub_col]
                    for sub_col in self.tgt_sub_columns[lookup.sub_col_offset : lookup.sub_col_cum]
                }
                if sub_col.startswith(RIDX_SUB_COLUMN_PREFIX):
                    # RIDX sub-columns should not see SLEN sub-columns
                    prev_sub_col_embeds = {
                        k: torch.zeros_like(v) if k.startswith(SLEN_SUB_COLUMN_PREFIX) else v
                        for k, v in prev_sub_col_embeds.items()
                    }
                prev_sub_col_embeds = list(prev_sub_col_embeds.values())

                # regressor
                regressor_in = context_history + masked_col_embeds + prev_sub_col_embeds
                xs = self.regressors(regressor_in, sub_col)

                # predictor
                xs = self.predictors(xs, sub_col)

                # update output
                outputs[sub_col] = xs

            return outputs

        else:  # mode == "gen"
            is_0th_step = False
            if history is None or history_state is None:
                is_0th_step = True
                # initialize history
                history_in = torch.cat(
                    [
                        torch.zeros(
                            (batch_size, 1, self.embedders.dims[i]),
                            device=self.device,
                        )
                        for i in range(len(self.tgt_cardinalities))
                    ],
                    dim=-1,
                )
                # history_state is tuple of current and hidden LSTM state; see nn.LSTM.forward() for further details
                history, history_state = self.history_compressor(history_in)

            flat_ctx, seq_ctx, seq_ctx_masks = context
            # attention between sequential context and history
            seq_ctx = self.attention(history, seq_ctx, seq_ctx_masks)
            # repeat flat context to match the length of history
            flat_ctx = self._repeat_flat_context(flat_ctx, history.size(1))
            context_history = [torch.cat(flat_ctx + seq_ctx + [history], -1)]

            # initialize sub column embeddings
            tgt_embeds = self.embedders.zero_mask(batch_size, 1)

            # initialize column embeddings
            tgt_col_embeds = self.column_embedders.zero_mask(batch_size, 1)

            # concatenate column embeddings
            col_embeddings = torch.cat(list(tgt_col_embeds.values()), dim=-1)

            # take sub columns in the specified generation order
            column_order = effective_column_order or self.tgt_columns
            sub_column_order = [sub_col for col in column_order for sub_col in self.tgt_column_sub_columns[col]]

            for sub_col in sub_column_order:
                lookup = self.tgt_sub_columns_lookup[sub_col]

                # if sub column is fixed, skip sampling and use that value
                if sub_col in fixed_values:
                    out = fixed_values[sub_col]

                else:  # sample from distribution
                    # collect previous sub column embeddings for current column
                    prev_sub_col_embeds = {
                        sub_col: tgt_embeds[sub_col]
                        for sub_col in self.tgt_sub_columns[lookup.sub_col_offset : lookup.sub_col_cum]
                    }
                    if sub_col.startswith(RIDX_SUB_COLUMN_PREFIX):
                        # RIDX sub-columns should not see SLEN sub-columns
                        prev_sub_col_embeds = {
                            k: torch.zeros_like(v) if k.startswith(SLEN_SUB_COLUMN_PREFIX) else v
                            for k, v in prev_sub_col_embeds.items()
                        }
                    prev_sub_col_embeds = list(prev_sub_col_embeds.values())

                    # regressor
                    regressor_in = context_history + [col_embeddings] + prev_sub_col_embeds
                    xs = self.regressors(regressor_in, sub_col)

                    # predictor
                    xs = self.predictors(xs, sub_col)

                    # sample
                    xs = nn.Softmax(dim=-1)(xs)
                    xs = torch.squeeze(xs[:, -1:, :], -2)  # take only last element
                    out = _sample(
                        probs=xs,
                        temperature=temperature,
                        top_p=top_p,
                        fixed_probs=fixed_probs.get(sub_col),
                    )

                # update timestep output
                if is_0th_step and sub_col.startswith(RIDX_SUB_COLUMN_PREFIX):
                    # overwrite output for RIDX sub-columns on 0th step with SLEN sub-columns
                    slen_sub_col = sub_col.replace(RIDX_SUB_COLUMN_PREFIX, SLEN_SUB_COLUMN_PREFIX)
                    out = outputs[slen_sub_col] if slen_sub_col in outputs else out
                outputs[sub_col] = out

                # update current sub column embedding
                tgt_embeds[sub_col] = self.embedders.get(sub_col)(out)
                tgt_embeds_for_history = {
                    k: torch.zeros_like(v) if k.startswith(history_masked_sub_cols) else v
                    for k, v in tgt_embeds.items()
                }
                tgt_embeds_for_col_embeddings = {
                    k: torch.zeros_like(v) if k.startswith(col_embeddings_masked_sub_cols) else v
                    for k, v in tgt_embeds.items()
                }

                # update current column embedding
                if sub_col in self.tgt_last_sub_cols:
                    col_sub_cols = self.tgt_column_sub_columns[lookup.col_name]
                    col_embed_in = torch.cat([tgt_embeds_for_col_embeddings[sc] for sc in col_sub_cols], dim=-1)
                    tgt_col_embeds[lookup.col_name] = self.column_embedders.get(lookup.col_name)(col_embed_in)
                    col_embeddings = torch.cat(list(tgt_col_embeds.values()), dim=-1)

            # update history and hidden state
            history_in = torch.cat(list(tgt_embeds_for_history.values()), dim=-1)
            history, history_state = self.history_compressor(history_in, history_state=history_state)

            # order outputs according to tgt_sub_columns
            outputs = {sub_col: outputs[sub_col] for sub_col in self.tgt_cardinalities}

            # gather output
            return outputs, history, history_state
