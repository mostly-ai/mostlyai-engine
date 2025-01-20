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

import random

import numpy as np
import pytest
import torch

from mostlyai.engine._tabular.argn import (
    _make_permutation_mask,
    _sampling_fixed_probs,
    _sampling_nucleus,
    _sampling_temperature,
    _sample,
)


class TestSamplingFixedProbs:
    test_probs_2d_cases = [
        (
            {0: 0.0, 1: 0.0},
            torch.Tensor(
                [
                    [0.0, 0.0, 0.8, 0.2],
                    [0.0, 0.0, 0.2, 0.8],
                    [0.0, 0.0, 0.5, 0.5],
                ]
            ),
        ),
        ({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}, torch.full((3, 4), 0.25)),
        (
            {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
            torch.Tensor([0.1, 0.2, 0.3, 0.4]).tile((3, 1)),
        ),
        (
            {1: 0.5, 2: 0.2},
            torch.Tensor(
                [
                    [0.15, 0.5, 0.2, 0.15],
                    [0.06, 0.5, 0.2, 0.24],
                    [0.30, 0.5, 0.2, 0.00],
                ]
            ),
        ),
        ({1: 0.7, 2: 0.3}, torch.Tensor([0.0, 0.7, 0.3, 0.0]).tile((3, 1))),
    ]

    @pytest.fixture
    def probs_2d(self):
        return torch.Tensor(
            [
                [0.1, 0.4, 0.4, 0.1],
                [0.1, 0.4, 0.1, 0.4],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )

    @pytest.fixture
    def probs_3d(self, probs_2d):
        return probs_2d.unsqueeze(1)

    @pytest.mark.parametrize("fixed_probs, expected", test_probs_2d_cases)
    def test_probs_2d(self, probs_2d, fixed_probs, expected):
        probs_out = _sampling_fixed_probs(probs_2d, fixed_probs)
        assert probs_2d.shape == probs_out.shape
        assert torch.allclose(probs_out, expected)

    @pytest.mark.parametrize("fixed_probs, expected", test_probs_2d_cases)
    def test_probs_3d(self, probs_3d, fixed_probs, expected):
        expected = expected.unsqueeze(1)
        probs_out = _sampling_fixed_probs(probs_3d, fixed_probs)
        assert probs_3d.shape == probs_out.shape
        assert torch.allclose(probs_out, expected)

    @pytest.mark.parametrize("do_3d_case", [False, True])
    def test_edge_case_single_probability(self, do_3d_case):
        probs = torch.Tensor(
            [
                [0.3],
                [1.0],
            ]
        )
        expected = torch.Tensor(
            [
                [1.0],
                [1.0],
            ]
        )
        if do_3d_case:
            probs = probs.unsqueeze(1)
            expected = expected.unsqueeze(1)
        fixed_probs = {0: 0.0}
        probs_out = _sampling_fixed_probs(probs, fixed_probs)
        assert torch.allclose(probs_out, expected)

    @pytest.mark.parametrize("do_3d_case", [False, True])
    def test_edge_case_many_summing_to_one(self, do_3d_case):
        # ensure that 11-th probability (unfixed) is non-negative
        probs = torch.Tensor([[0.01] * 11])
        if do_3d_case:
            probs = probs.unsqueeze(1)
        fixed_probs = {i: 0.1 for i in range(10)}
        probs_out = _sampling_fixed_probs(probs, fixed_probs)
        assert (probs_out >= 0).all()


class TestSamplingNucleus:
    @pytest.fixture
    def probs_2d(self):
        return torch.Tensor(
            [
                [0.4, 0.4, 0.2],
                [0.4, 0.2, 0.4],
            ]
        )

    @pytest.fixture
    def probs_2d_expected_p_075(self):
        """Expected outcome for nucleus sampling with top_p = 0.75 (2D case)"""
        return torch.Tensor(
            [
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
            ]
        )

    @pytest.mark.parametrize(
        "top_p, expected_output",
        [
            (1.0, "probs_2d"),
            (0.75, "probs_2d_expected_p_075"),
        ],
    )
    def test_probs_2d(self, top_p, expected_output, request):
        probs_2d = request.getfixturevalue("probs_2d")
        expected = request.getfixturevalue(expected_output)

        probs_out = _sampling_nucleus(probs_2d, top_p=top_p)

        assert probs_2d.shape == probs_out.shape
        assert torch.allclose(probs_out, expected)

    @pytest.fixture
    def probs_3d(self):
        return torch.Tensor(
            [
                [
                    [0.2, 0.3, 0.5],
                    [0.1, 0.1, 0.8],
                ],
                [
                    [0.5, 0.3, 0.2],
                    [0.8, 0.1, 0.1],
                ],
            ]
        )

    @pytest.fixture
    def probs_3d_expected_p_075(self):
        """Expected outcome for nucleus sampling with top_p = 0.75 (3D case)"""
        return torch.Tensor(
            [
                [
                    [0.0, 0.375, 0.625],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [0.625, 0.375, 0.0],
                    [1.0, 0.0, 0.0],
                ],
            ]
        )

    @pytest.mark.parametrize(
        "top_p, expected_output",
        [
            (1.0, "probs_3d"),
            (0.75, "probs_3d_expected_p_075"),
        ],
    )
    def test_probs_3d(self, top_p, expected_output, request):
        probs_3d = request.getfixturevalue("probs_3d")
        expected = request.getfixturevalue(expected_output)

        probs_out = _sampling_nucleus(probs_3d, top_p=top_p)

        assert probs_3d.shape == probs_out.shape
        assert torch.allclose(probs_out, expected)


class TestSamplingTemperature:
    @pytest.fixture
    def probs(self):
        return torch.Tensor(
            [
                [0.1, 0.4, 0.4, 0.1],
                [0.1, 0.4, 0.1, 0.4],
            ]
        )

    @pytest.mark.parametrize("do_3d_case", [False, True])
    @pytest.mark.parametrize("do_single_prob_case", [False, True])
    def test(self, probs, do_3d_case, do_single_prob_case):
        if do_single_prob_case:
            probs = probs[:, 0:1]
        if do_3d_case:
            probs = probs.unsqueeze(1)
        probs_out = _sampling_temperature(probs, 0.1)
        assert probs.shape == probs_out.shape
        # check that all probabilities are non-zero
        assert (probs_out > 0).all()
        # check that probabilities add up to 1
        assert (probs_out.sum(dim=-1) == 1.0).all()


class TestMakePermutationMask:
    def test_flat_fixed_order(self):
        col_embedding_dims = [2, 3, 5, 7]
        columns = [f"tgt:t0/c{idx}" for idx in range(len(col_embedding_dims))]
        column_order = random.sample(columns, len(columns))
        mask = _make_permutation_mask(
            col_embedding_dims=col_embedding_dims,
            columns=columns,
            column_order=column_order,
            is_sequential=False,
            device=torch.device("cpu"),
        )
        mask_col_segment_start = [0, *np.cumsum(col_embedding_dims)[:-1]]
        mask_col_segment_end = [start + col_embedding_dims[i] for i, start in enumerate(mask_col_segment_start)]
        for column_idx, column in enumerate(columns):
            prev_columns = column_order[: column_order.index(column)]
            column_mask = mask[column_idx, :]
            for _column_idx, _column in enumerate(columns):
                mask_slice = column_mask[mask_col_segment_start[_column_idx] : mask_col_segment_end[_column_idx]]
                expected_slice = torch.full_like(mask_slice, fill_value=(_column in prev_columns))
                assert mask_slice.equal(expected_slice)

    def test_sequential_any_order(self):
        col_embedding_dims = [3, 5, 7]
        columns = [f"tgt:t0/c{idx}" for idx in range(len(col_embedding_dims))]

        # append tgt:/ column
        slen_sidx_col_dim = 2
        col_embedding_dims += [slen_sidx_col_dim]
        columns += ["tgt:/"]

        mask = _make_permutation_mask(
            col_embedding_dims=col_embedding_dims,
            columns=columns,
            column_order=None,
            is_sequential=True,
            device=torch.device("cpu"),
        )
        # tgt:/ column is shifted to the beginning; all other columns see it
        assert mask[0, :slen_sidx_col_dim].equal(torch.Tensor([False, False]))
        assert mask[1, :slen_sidx_col_dim].equal(torch.Tensor([True, True]))
        assert mask[2, :slen_sidx_col_dim].equal(torch.Tensor([True, True]))
        assert mask[3, :slen_sidx_col_dim].equal(torch.Tensor([True, True]))


class TestSample:
    @pytest.fixture
    def invalid_probs(self):
        return torch.Tensor(
            [
                [0.2, 0.3, -0.5, 1.0],
                [0.1, 0.2, 0.7, float("inf")],
                [0.3, 0.3, 0.4, float("nan")],
                [float("nan"), float("nan"), float("nan"), float("nan")],
            ]
        )

    def test_invalid_probs(self, invalid_probs):
        probs = _sample(invalid_probs)
        assert probs.shape == (4, 1)
