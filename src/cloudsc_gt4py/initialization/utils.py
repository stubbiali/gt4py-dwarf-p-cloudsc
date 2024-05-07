# -*- coding: utf-8 -*-
#
# Copyright 2022-2024 ETH Zurich
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

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from ifs_physics_common.utils.numpyx import assign

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ifs_physics_common.utils.typingx import DataArray, NDArrayLike


def initialize_storage_2d(storage: NDArrayLike, buffer: NDArray) -> None:
    ni = storage.shape[0]
    mi = buffer.size
    nb = ni // mi
    for b in range(nb):
        assign(storage[b * mi : (b + 1) * mi, 0:1], buffer[:, np.newaxis])
    assign(storage[nb * mi :, 0:1], buffer[: ni - nb * mi, np.newaxis])


def initialize_storage_3d(storage: NDArrayLike, buffer: NDArray) -> None:
    ni, _, nk = storage.shape
    mi, mk = buffer.shape
    lk = min(nk, mk)
    nb = ni // mi
    for b in range(nb):
        assign(storage[b * mi : (b + 1) * mi, 0:1, :lk], buffer[:, np.newaxis, :lk])
    assign(storage[nb * mi :, 0:1, :lk], buffer[: ni - nb * mi, np.newaxis, :lk])


def initialize_field(field: DataArray, buffer: NDArray) -> None:
    if field.ndim == 2:
        initialize_storage_2d(field.data, buffer)
    elif field.ndim == 3:
        initialize_storage_3d(field.data, buffer)
    else:
        raise ValueError("The field to initialize must be either 2-d or 3-d.")
