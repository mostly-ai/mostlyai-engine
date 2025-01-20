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

import pandas as pd
import pyarrow as pa


def is_string_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_string(x.dtype.pyarrow_dtype)
    else:
        return pd.api.types.is_string_dtype(x)


def is_integer_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_integer(x.dtype.pyarrow_dtype)
    else:
        return pd.api.types.is_integer_dtype(x)


def is_float_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_floating(x.dtype.pyarrow_dtype)
    else:
        return pd.api.types.is_float_dtype(x)


def is_date_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_date(x.dtype.pyarrow_dtype)
    else:
        return False


def is_timestamp_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_timestamp(x.dtype.pyarrow_dtype)
    else:
        return pd.api.types.is_datetime64_any_dtype(x)


def is_boolean_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_boolean(x.dtype.pyarrow_dtype)
    else:
        return pd.api.types.is_bool_dtype(x)
