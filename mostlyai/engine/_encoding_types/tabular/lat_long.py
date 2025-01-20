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

from enum import Enum
from itertools import chain

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mostlyai.engine._common import safe_convert_string
from mostlyai.engine._encoding_types.tabular.categorical import (
    CATEGORICAL_UNKNOWN_TOKEN,
    encode_categorical,
)
from mostlyai.engine._encoding_types.tabular.character import (
    decode_character,
    encode_character,
)

MAX_UNIQUE_VALUES_PER_QUAD = 10_000
RARE_CATEGORY_THRESHOLD = 20
MIN_UNIQUE_QUAD_VALUES = 5


class GeoPositionSymbols(Enum):
    A = ("0", "0")  # NORTH-WEST
    B = ("1", "0")  # NORTH-EAST
    C = ("0", "1")  # SOUTH-WEST
    D = ("1", "1")  # SOUTH-EAST


GeoPositionSymbolsReverseLookup = {s.value: s.name for s in GeoPositionSymbols}
GEOPOSITION_PRECISION = 100_000
GEOPOSITION_BINARY_REPRESENTATION_LENGTH = 28
GEOPOSITION_MAX_LATITUDE = 90 * GEOPOSITION_PRECISION
GEOPOSITION_MAX_LONGITUDE = 180 * GEOPOSITION_PRECISION
GEOPOSITION_STEP_SIZE = 2
GEOPOSITION_SIGNS_LENGTH = 2
NUMBER_OF_GEOPOSITION_QUADS = 8
GEOPOSITION_FIRST_QUAD_LENGTH = 12
GEOPOSITION_FIRST_QUAD_ONSET = GEOPOSITION_FIRST_QUAD_LENGTH + GEOPOSITION_SIGNS_LENGTH
GEOPOSITION_QUADTILE_LENGTH = GEOPOSITION_FIRST_QUAD_ONSET + NUMBER_OF_GEOPOSITION_QUADS * GEOPOSITION_STEP_SIZE
NONE_QUADTILE_ENTRY = "++AAAAAAAAAAAAAAAAAAAAAAAAAAAA"
PLACEHOLDER_QUADTILE_ENTRY = "++AAAAAAAAAAAAAAAAAAAAAAAAAAAA"
GEOPOSITION_OUTPUT_FORMAT = "{latitude:.5f}, {longitude:.5f}"
QUADTILE = "QUADTILE"
QUAD_COLUMNS = [
    f"QUAD{GEOPOSITION_FIRST_QUAD_LENGTH + GEOPOSITION_STEP_SIZE * i}" for i in range(NUMBER_OF_GEOPOSITION_QUADS)
]


def _latlong_int_to_binary(latlong: NDArray) -> NDArray:
    """
    Transform an array of int64 tuples representing latlong (with 10^-5 precision) to
    an array of quadtile strings (28 characters for each). Note: excluding the signs.

    For example: [[3306060 3300000], [3312120 3300000]] ->
    ['AAAAAADDAADAADBDCADACBCABBAA' 'AAAAAADDAADABCACDACBDBDBBAAA']

    :param latlong: Numpy array of (lat, long) int64 tuples (non-negative numbers)
    :return: Numpy array of object (quadtile strings)
    """
    str_repr = np.vectorize(np.binary_repr, otypes=[str])(
        latlong[:, :],
        width=GEOPOSITION_BINARY_REPRESENTATION_LENGTH,
    )

    def _latlong_to_binary_single(latitude: str, longitude: str) -> str:
        """
        Transform latitude and longitude (in binary string format) to quadtile
        """
        quad = [GeoPositionSymbolsReverseLookup[(lat, long)] for lat, long in zip(list(latitude), list(longitude))]
        return "".join(quad)

    quads_transform = np.frompyfunc(_latlong_to_binary_single, nin=2, nout=1)
    quads = quads_transform(str_repr[:, 0], str_repr[:, 1])

    return quads


def _latlong_to_binary(geo_positions: NDArray) -> list[str]:
    """
    Transform geoposition values (latitude and longitude) into quadtile string representations.

    For example: [[33.0606  33.     ], [33.12121 33.     ]] ->
    ['++AAAAAADDAADAADBDCADACBCABBAA' '++AAAAAADDAADABCACDACBDBDBBAAA']

    :param geo_positions: Numpy array of (lat, long) tuples
    :return:
    """
    # operate on integer values to get rid of the values below certain precision
    numeric = (geo_positions * GEOPOSITION_PRECISION).astype(int)

    # get the sign and work on the absolute values
    signs = np.array(["-", "+"])[(numeric >= 0).astype(int)]
    numeric = np.abs(numeric)

    # apply ceiling to the values
    numeric[numeric[:, 0] > GEOPOSITION_MAX_LATITUDE, 0] = GEOPOSITION_MAX_LATITUDE
    numeric[numeric[:, 1] > GEOPOSITION_MAX_LONGITUDE, 1] = GEOPOSITION_MAX_LONGITUDE

    # transform numeric to binary (quadtile strings)
    quads = _latlong_int_to_binary(numeric)

    # build final strings, adding signs
    combine_strings = np.frompyfunc(lambda sign1, sign2, quad: sign1 + sign2 + quad, 3, 1)
    quadtiles = combine_strings(signs[:, 0], signs[:, 1], quads)

    return quadtiles


def _binary_to_latlong(data: pd.Series) -> pd.Series:
    """
    Transform quadtile column into latlong format.
    For example ("-+AAACBAACBACBCBCBCBAACBAAAAAA") -> ("-90, 180")

    :param data: Quadtile as pandas series
    """

    def _binary_to_latlong_single(
        quadtile: str,
    ) -> tuple[float | None, float | None]:
        """
        Transform from a quadtile to its (lat, long) float representation
        For example: '-+AAAAAABCCBBBDADBBCCADDCBCCBB' -> (30.87, 133.11331)
        """
        if not quadtile or not len(quadtile) == len(PLACEHOLDER_QUADTILE_ENTRY):
            return None, None

        latitude_sign = 1 if quadtile[0] == "+" else -1
        longitude_sign = 1 if quadtile[1] == "+" else -1

        latitude_bin = []
        longitude_bin = []
        for quadtile_character in quadtile[2:]:
            lat, long = GeoPositionSymbols[quadtile_character].value
            latitude_bin.append(lat)
            longitude_bin.append(long)

        latitude = (int("".join(latitude_bin), 2) * latitude_sign) / GEOPOSITION_PRECISION
        longitude = (int("".join(longitude_bin), 2) * longitude_sign) / GEOPOSITION_PRECISION

        return latitude, longitude

    quads_reverse_transform = np.frompyfunc(_binary_to_latlong_single, nin=1, nout=2)
    latitudes, longitudes = quads_reverse_transform(data.to_numpy())

    latlong_to_string = np.frompyfunc(
        lambda lat, long: GEOPOSITION_OUTPUT_FORMAT.format(latitude=lat, longitude=long),
        nin=2,
        nout=1,
    )
    geo_positions = latlong_to_string(latitudes, longitudes)

    geo_positions_data = pd.Series(data=geo_positions, name=data.name, index=data.index)

    return geo_positions_data


def split_sub_columns_latlong(
    data: pd.Series,
) -> pd.DataFrame:
    latitude_longitude = split_str_to_latlong(data)
    isnan = np.isnan(latitude_longitude)
    latitude_longitude[isnan] = 0.0
    # mark invalid rows
    invalid_entry = np.any(isnan, axis=1).values

    quadtile = _latlong_to_binary(latitude_longitude.to_numpy())

    def _geoposition_split_quads(quadtile: str) -> tuple:
        """
        Split the complete quadtile (e.g. '-+AAAAAABCCBBBDADBBCCADDCBCCBB') into its
        sub-quadtiles: QUAD12, ..., QUAD28
        """
        quads = [
            quadtile[0:x]
            for x in range(
                GEOPOSITION_FIRST_QUAD_ONSET,
                GEOPOSITION_QUADTILE_LENGTH + GEOPOSITION_STEP_SIZE,
                GEOPOSITION_STEP_SIZE,
            )
        ]
        return tuple(quads)

    split_quadtile = np.frompyfunc(_geoposition_split_quads, nin=1, nout=NUMBER_OF_GEOPOSITION_QUADS + 1)
    quads = split_quadtile(quadtile)

    # return a dataframe containing the sub quads as columns
    transformed_data = pd.DataFrame(quads).T
    transformed_data.columns = QUAD_COLUMNS + [
        QUADTILE,
    ]

    # filter (set to empty string) the invalid entries for quads
    transformed_data[invalid_entry] = ""
    # filter invalid entries for quadtile by setting them to special string
    transformed_data.loc[invalid_entry, QUADTILE] = NONE_QUADTILE_ENTRY
    # insert a binary "nan" column at the beginning of the returned DataFrame
    transformed_data.insert(0, "nan", isnan.any(axis=1).astype(int))
    return transformed_data


def split_str_to_latlong(data: pd.Series) -> pd.DataFrame:
    """
    Split the string input into two float columns
    "-13.1345, 89.331" - > | -13.1345 | 89.331 |
    Anything that is not a match for a number is converted into nan

    :param data: stringified lat, long Series

    return: a DataFrame of two (lat, long) columns, accordingly
    """

    latitude_longitude = (
        (
            pd.concat([data, pd.Series(pd.NA)])  # ensure at least one row
            .reset_index(drop=True)
            .fillna("")
            .astype(str)
            + ","  # ensure at least one comma per row
        )
        .str.split(",", expand=True)
        .iloc[:, 0:2]  # select the first two columns: lat, long
        .apply(lambda x: pd.to_numeric(x, downcast="float", errors="coerce"))
        .head(-1)  # drop previously added row
    )
    return latitude_longitude


def analyze_latlong(values: pd.Series, root_keys: pd.Series, _: pd.Series | None = None) -> dict:
    values = safe_convert_string(values)
    df_split = split_sub_columns_latlong(values)
    df = pd.concat([root_keys, df_split], axis=1)

    quad_partials = [
        name for name in list(df.columns) if name not in (root_keys.name, "QUADTILE", "nan")
    ]  # ["QUAD_12", "QUAD_14", ...]
    has_nan = bool(df_split["nan"].any())

    # mapping of values and their counts (per root key) for each of the quads
    quad_codes = {
        quad_partial: df[[quad_partial, root_keys.name]].drop_duplicates()[quad_partial].value_counts().to_dict()
        for quad_partial in quad_partials
    }

    df_non_na_quadtiles = df_split["QUADTILE"][df_split["nan"] == 0]
    if df_non_na_quadtiles.empty:  # in the edge-case if all are NaN
        df_non_na_quadtiles = pd.Series([PLACEHOLDER_QUADTILE_ENTRY])
    chars_df = df_non_na_quadtiles.str.split("", expand=True).drop([0, GEOPOSITION_QUADTILE_LENGTH + 1], axis=1)
    chars_df.columns = [f"P{idx}" for idx in range(GEOPOSITION_QUADTILE_LENGTH)]
    characters = {sub_col: list(chars_df[sub_col].unique()) for sub_col in chars_df.columns}

    stats = {"has_nan": has_nan, "quad_codes": quad_codes, "characters": characters}
    return stats


def analyze_reduce_latlong(stats_list: list[dict], value_protection: bool = True) -> dict:
    # check if there are missing values
    has_nan = any([j["has_nan"] for j in stats_list])
    unk_cat_aliases = [""]  # na / unknown (unseen) category

    quads = list(stats_list[0]["quad_codes"].keys()) if len(stats_list) else []
    quad_codes = {}

    for quad in quads:
        keys = list(
            set(chain.from_iterable([stats["quad_codes"][quad].keys() for stats in stats_list]))
        )  # all the possible values for a given quad
        totals = {
            key: sum([stats["quad_codes"][quad].get(key, 0) for stats in stats_list]) for key in keys if key
        }  # mapped to total counts per each value
        totals = {
            val: cnt for val, cnt in totals.items() if cnt >= RARE_CATEGORY_THRESHOLD
        }  # keep only those, which appear at least {rare_category_threshold} times
        categories = ([CATEGORICAL_UNKNOWN_TOKEN] + [cat for cat in totals.keys() if cat not in unk_cat_aliases])[
            :MAX_UNIQUE_VALUES_PER_QUAD
        ]  # UNK + remaining possible values
        if len(categories) - 1 < MIN_UNIQUE_QUAD_VALUES:  # excluding unknown category
            continue  # do not include quads having unique values less than {min_unique_quad_values}
        quad_codes[quad] = {category: i for i, category in enumerate(categories)}

    # combine all the characters stats
    chars_all: dict[str, list] = {f"P{idx}": [] for idx in range(GEOPOSITION_QUADTILE_LENGTH)}
    for stats in stats_list:
        for k, v in stats["characters"].items():
            chars_all[k] += list(set(v) - set(chars_all[k]))

    # create character stats
    quadtile_characters = {
        "max_string_length": GEOPOSITION_QUADTILE_LENGTH,
        "has_nan": False,
        "codes": {
            f"P{idx}": {char: code for code, char in enumerate(sorted(chars_all[f"P{idx}"]))}
            for idx in range(GEOPOSITION_QUADTILE_LENGTH)
        },
    }

    # determine cardinalities
    cardinalities = {quad: len(quad_codes[quad]) for quad in quads if quad in quad_codes}
    cardinalities.update({k: len(v) for k, v in quadtile_characters["codes"].items()})  # type: ignore
    if has_nan:
        cardinalities["nan"] = 2  # binary

    stats = {
        "has_nan": has_nan,
        "cardinalities": cardinalities,
        "quad_codes": quad_codes,
        "quadtile_characters": quadtile_characters,
    }
    return stats


def encode_latlong(
    values: pd.Series,
    column_stats: dict,
    context_keys: pd.Series | None = None,
) -> pd.DataFrame:
    values = safe_convert_string(values)
    # split to sub_columns
    quads = split_sub_columns_latlong(values)
    encoded_quads = pd.DataFrame()  # empty DF to include all the ModelEncodingType.tabular_categorical quads
    for quad, value_counts in column_stats["quad_codes"].items():
        stats = {"codes": value_counts}
        encoded_quads[quad] = encode_categorical(quads[quad], stats)

    encoded_quadtile = encode_character(quads["QUADTILE"], column_stats["quadtile_characters"])
    if column_stats["has_nan"]:
        encoded_quadtile["nan"] = quads["nan"]
    df = pd.concat([encoded_quads, encoded_quadtile], axis=1)
    return df


def decode_latlong(df_encoded: pd.DataFrame, stats: dict) -> pd.Series:
    redundant_columns = [col for col in df_encoded.columns if not col.startswith("P")]
    df = df_encoded.drop(redundant_columns, axis=1)
    decoded_latlong = decode_character(df, stats["quadtile_characters"])
    decoded_latlong = _binary_to_latlong(decoded_latlong)
    if "nan" in df_encoded.columns:
        decoded_latlong[df_encoded["nan"] == 1] = pd.NA
    return decoded_latlong
