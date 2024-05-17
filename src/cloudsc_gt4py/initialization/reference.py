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
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING

from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import initialize_field, zeros

if TYPE_CHECKING:
    from typing import Literal

    from cloudsc_gt4py.utils.iox import HDF5Reader
    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid, DimTuple
    from ifs_physics_common.utils.typingx import DataArray, DataArrayDict


def allocate_tendencies(
    computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig
) -> DataArrayDict:
    def _zeros(units: str = "") -> DataArray:
        return zeros(
            computational_grid, (I, J, K), units, gt4py_config=gt4py_config, dtype_name="float"
        )

    return {
        "time": datetime(year=2022, month=1, day=1),
        "f_a": _zeros(),
        "f_qi": _zeros(),
        "f_ql": _zeros(),
        "f_qr": _zeros(),
        "f_qs": _zeros(),
        "f_qv": _zeros(),
        "f_t": _zeros(),
    }


def initialize_tendencies(tendencies: DataArrayDict, hdf5_reader: HDF5Reader) -> None:
    hdf5_reader_keys = {"f_a": "TENDENCY_LOC_A", "f_qv": "TENDENCY_LOC_Q", "f_t": "TENDENCY_LOC_T"}
    for name, hdf5_reader_key in hdf5_reader_keys.items():
        buffer = hdf5_reader.get_field(hdf5_reader_key)
        initialize_field(tendencies[name], buffer)

    cld = hdf5_reader.get_field("TENDENCY_LOC_CLD")
    for idx, name in enumerate(("f_ql", "f_qi", "f_qr", "f_qs")):
        initialize_field(tendencies[name], cld[..., idx])


def allocate_diagnostics(
    computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig
) -> DataArrayDict:
    def _zeros(
        grid_id: DimTuple, units: str, dtype_name: Literal["bool", "float", "int"]
    ) -> DataArray:
        return zeros(
            computational_grid, grid_id, units, gt4py_config=gt4py_config, dtype_name=dtype_name
        )

    zeros_ijk = partial(_zeros, grid_id=(I, J, K), units="", dtype_name="float")
    zeros_ijk_h = partial(_zeros, grid_id=(I, J, K - 1 / 2), units="", dtype_name="float")
    zeros_ij = partial(_zeros, grid_id=(I, J), units="", dtype_name="float")

    return {
        "time": datetime(year=2022, month=1, day=1),
        "f_covptot": zeros_ijk(),
        "f_fcqlng": zeros_ijk_h(),
        "f_fcqnng": zeros_ijk_h(),
        "f_fcqrng": zeros_ijk_h(),
        "f_fcqsng": zeros_ijk_h(),
        "f_fhpsl": zeros_ijk_h(),
        "f_fhpsn": zeros_ijk_h(),
        "f_fplsl": zeros_ijk_h(),
        "f_fplsn": zeros_ijk_h(),
        "f_fsqif": zeros_ijk_h(),
        "f_fsqitur": zeros_ijk_h(),
        "f_fsqlf": zeros_ijk_h(),
        "f_fsqltur": zeros_ijk_h(),
        "f_fsqrf": zeros_ijk_h(),
        "f_fsqsf": zeros_ijk_h(),
        "f_rainfrac_toprfz": zeros_ij(),
    }


def initialize_diagnostics(diagnostics: DataArrayDict, hdf5_reader: HDF5Reader) -> None:
    hdf5_reader_keys = {name: "P" + name[2:].upper() for name in diagnostics if name != "time"}
    for name, hdf5_reader_key in hdf5_reader_keys.items():
        buffer = hdf5_reader.get_field(hdf5_reader_key)
        initialize_field(diagnostics[name], buffer)


def get_reference_tendencies(
    computational_grid: ComputationalGrid, hdf5_reader: HDF5Reader, *, gt4py_config: GT4PyConfig
) -> DataArrayDict:
    tendencies = allocate_tendencies(computational_grid, gt4py_config=gt4py_config)
    initialize_tendencies(tendencies, hdf5_reader)
    return tendencies


def get_reference_diagnostics(
    computational_grid: ComputationalGrid, hdf5_reader: HDF5Reader, *, gt4py_config: GT4PyConfig
) -> DataArrayDict:
    diagnostics = allocate_diagnostics(computational_grid, gt4py_config=gt4py_config)
    initialize_diagnostics(diagnostics, hdf5_reader)
    return diagnostics
