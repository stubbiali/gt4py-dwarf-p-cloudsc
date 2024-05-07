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

from cloudsc_gt4py.initialization.utils import initialize_field
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array

if TYPE_CHECKING:
    from typing import Literal, Tuple

    from cloudsc_gt4py.utils.iox import HDF5Reader
    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
    from ifs_physics_common.utils.typingx import DataArray, DataArrayDict


def allocate_state(
    computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig
) -> DataArrayDict:
    def _allocate(
        grid_id: Tuple[DimSymbol, ...], units: str, dtype: Literal["bool", "float", "int"]
    ) -> DataArray:
        return allocate_data_array(
            computational_grid, grid_id, units, gt4py_config=gt4py_config, dtype=dtype
        )

    allocate_b_ij = partial(_allocate, grid_id=(I, J), units="", dtype="bool")
    allocate_f = partial(_allocate, grid_id=(I, J, K), units="", dtype="float")
    allocate_f_h = partial(_allocate, grid_id=(I, J, K - 1 / 2), units="", dtype="float")
    allocate_f_ij = partial(_allocate, grid_id=(I, J), units="", dtype="float")
    allocate_i_ij = partial(_allocate, grid_id=(I, J), units="", dtype="int")

    return {
        "time": datetime(year=2022, month=1, day=1),
        "b_convection_on": allocate_b_ij(),
        "f_a": allocate_f(),
        "f_ap": allocate_f(),
        "f_aph": allocate_f_h(),
        "f_ccn": allocate_f(),
        "f_dyni": allocate_f(),
        "f_dynl": allocate_f(),
        "f_hrlw": allocate_f(),
        "f_hrsw": allocate_f(),
        "f_icrit_aer": allocate_f(),
        "f_lcrit_aer": allocate_f(),
        "f_lsm": allocate_f_ij(),
        "f_lu": allocate_f(),
        "f_lude": allocate_f(),
        "f_mfd": allocate_f(),
        "f_mfu": allocate_f(),
        "f_nice": allocate_f(),
        "f_qi": allocate_f(),
        "f_ql": allocate_f(),
        "f_qr": allocate_f(),
        "f_qs": allocate_f(),
        "f_qv": allocate_f(),
        "f_re_ice": allocate_f(),
        "f_snde": allocate_f(),
        "f_supsat": allocate_f(),
        "f_t": allocate_f(),
        "f_tnd_tmp_a": allocate_f(),
        "f_tnd_tmp_qi": allocate_f(),
        "f_tnd_tmp_ql": allocate_f(),
        "f_tnd_tmp_qr": allocate_f(),
        "f_tnd_tmp_qs": allocate_f(),
        "f_tnd_tmp_qv": allocate_f(),
        "f_tnd_tmp_t": allocate_f(),
        "f_vfa": allocate_f(),
        "f_vfi": allocate_f(),
        "f_vfl": allocate_f(),
        "f_w": allocate_f(),
        "i_convection_type": allocate_i_ij(),
    }


def initialize_state(state: DataArrayDict, hdf5_reader: HDF5Reader) -> None:
    hdf5_reader_keys = {
        "b_convection_on": "LDCUM",
        "f_a": "PA",
        "f_ap": "PAP",
        "f_aph": "PAPH",
        "f_ccn": "PCCN",
        "f_dyni": "PDYNI",
        "f_dynl": "PDYNL",
        "f_hrlw": "PHRLW",
        "f_hrsw": "PHRSW",
        "f_icrit_aer": "PICRIT_AER",
        "f_lcrit_aer": "PLCRIT_AER",
        "f_lsm": "PLSM",
        "f_lu": "PLU",
        "f_lude": "PLUDE",
        "f_mfd": "PMFD",
        "f_mfu": "PMFU",
        "f_nice": "PNICE",
        "f_qv": "PQ",
        "f_re_ice": "PRE_ICE",
        "f_snde": "PSNDE",
        "f_supsat": "PSUPSAT",
        "f_t": "PT",
        "f_tnd_tmp_a": "TENDENCY_TMP_A",
        "f_tnd_tmp_qv": "TENDENCY_TMP_Q",
        "f_tnd_tmp_t": "TENDENCY_TMP_T",
        "f_vfa": "PVFA",
        "f_vfi": "PVFI",
        "f_vfl": "PVFL",
        "f_w": "PVERVEL",
        "i_convection_type": "KTYPE",
    }
    for name, hdf5_reader_key in hdf5_reader_keys.items():
        buffer = hdf5_reader.get_field(hdf5_reader_key)
        initialize_field(state[name], buffer)

    clv = hdf5_reader.get_field("PCLV")
    for idx, name in enumerate(("f_ql", "f_qi", "f_qr", "f_qs")):
        initialize_field(state[name], clv[..., idx])

    tnd_tmp_cld = hdf5_reader.get_field("TENDENCY_TMP_CLD")
    for idx, name in enumerate(("f_tnd_tmp_ql", "f_tnd_tmp_qi", "f_tnd_tmp_qr", "f_tnd_tmp_qs")):
        initialize_field(state[name], tnd_tmp_cld[..., idx])


def get_state(
    computational_grid: ComputationalGrid, hdf5_reader: HDF5Reader, *, gt4py_config: GT4PyConfig
) -> DataArrayDict:
    state = allocate_state(computational_grid, gt4py_config=gt4py_config)
    initialize_state(state, hdf5_reader)
    return state
