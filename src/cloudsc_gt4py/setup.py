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
from typing import TYPE_CHECKING

from ifs_physics_common.grid import D5, ExpandedDim, I, IJ, J, K

if TYPE_CHECKING:
    from ifs_physics_common.iox import HDF5GridOperator
    from ifs_physics_common.typingx import DataArrayDict


IJ_ARGS = lambda h5_name, dtype_name="float", units="": {
    "grid_dims": (I, J),
    "dtype_name": dtype_name,
    "units": units,
    "h5_name": h5_name,
    "h5_dims": (IJ,),
    "h5_dims_map": (IJ, ExpandedDim),
}
IJK_ARGS = lambda h5_name, dim_k=K, units="": {
    "grid_dims": (I, J, dim_k),
    "dtype_name": "float",
    "units": units,
    "h5_name": h5_name,
    "h5_dims": (dim_k, IJ),
    "h5_dims_map": (IJ, ExpandedDim, dim_k),
}
IJKD5_ARGS = lambda h5_name, index, units="": {
    "grid_dims": (I, J, K),
    "dtype_name": "float",
    "units": units,
    "h5_name": h5_name,
    "h5_dims": (D5, K, IJ),
    "h5_dims_map": (IJ, ExpandedDim, K, D5[index]),
}
REFERENCE_TIME = datetime(year=1970, month=1, day=1)


def get_state(hdf5_grid_operator: HDF5GridOperator) -> DataArrayDict:
    field_properties = {
        "b_convection_on": IJ_ARGS(h5_name="LDCUM", dtype_name="bool"),
        "f_a": IJK_ARGS(h5_name="PA"),
        "f_ap": IJK_ARGS(h5_name="PAP"),
        "f_aph": IJK_ARGS(h5_name="PAPH", dim_k=K - 1 / 2),
        "f_ccn": IJK_ARGS(h5_name="PCCN"),
        "f_dyni": IJK_ARGS(h5_name="PDYNI"),
        "f_dynl": IJK_ARGS(h5_name="PDYNL"),
        "f_hrlw": IJK_ARGS(h5_name="PHRLW"),
        "f_hrsw": IJK_ARGS(h5_name="PHRSW"),
        "f_icrit_aer": IJK_ARGS(h5_name="PICRIT_AER"),
        "f_lcrit_aer": IJK_ARGS(h5_name="PLCRIT_AER"),
        "f_lsm": IJ_ARGS(h5_name="PLSM"),
        "f_lu": IJK_ARGS(h5_name="PLU"),
        "f_lude": IJK_ARGS(h5_name="PLUDE"),
        "f_mfd": IJK_ARGS(h5_name="PMFD"),
        "f_mfu": IJK_ARGS(h5_name="PMFU"),
        "f_nice": IJK_ARGS(h5_name="PNICE"),
        "f_qi": IJKD5_ARGS(h5_name="PCLV", index=1),
        "f_ql": IJKD5_ARGS(h5_name="PCLV", index=0),
        "f_qr": IJKD5_ARGS(h5_name="PCLV", index=2),
        "f_qs": IJKD5_ARGS(h5_name="PCLV", index=3),
        "f_qv": IJK_ARGS(h5_name="PQ"),
        "f_re_ice": IJK_ARGS(h5_name="PRE_ICE"),
        "f_snde": IJK_ARGS(h5_name="PSNDE"),
        "f_supsat": IJK_ARGS(h5_name="PSUPSAT"),
        "f_t": IJK_ARGS(h5_name="PT"),
        "f_tnd_tmp_a": IJK_ARGS(h5_name="TENDENCY_TMP_A"),
        "f_tnd_tmp_qi": IJKD5_ARGS(h5_name="TENDENCY_TMP_CLD", index=1),
        "f_tnd_tmp_ql": IJKD5_ARGS(h5_name="TENDENCY_TMP_CLD", index=0),
        "f_tnd_tmp_qr": IJKD5_ARGS(h5_name="TENDENCY_TMP_CLD", index=2),
        "f_tnd_tmp_qs": IJKD5_ARGS(h5_name="TENDENCY_TMP_CLD", index=3),
        "f_tnd_tmp_qv": IJK_ARGS(h5_name="TENDENCY_TMP_Q"),
        "f_tnd_tmp_t": IJK_ARGS(h5_name="TENDENCY_TMP_T"),
        "f_vfa": IJK_ARGS(h5_name="PVFA"),
        "f_vfi": IJK_ARGS(h5_name="PVFI"),
        "f_vfl": IJK_ARGS(h5_name="PVFL"),
        "f_w": IJK_ARGS(h5_name="PVERVEL"),
        "i_convection_type": IJ_ARGS(h5_name="KTYPE", dtype_name="int"),
    }
    state = {
        name: hdf5_grid_operator.get_field(**props) for name, props in field_properties.items()
    }
    state["time"] = REFERENCE_TIME
    return state


def get_reference_tendencies(hdf5_grid_operator: HDF5GridOperator) -> DataArrayDict:
    field_properties = {
        "f_a": IJK_ARGS(h5_name="TENDENCY_LOC_A", units="s^-1"),
        "f_qi": IJKD5_ARGS(h5_name="TENDENCY_LOC_CLD", units="s^-1", index=1),
        "f_ql": IJKD5_ARGS(h5_name="TENDENCY_LOC_CLD", units="s^-1", index=0),
        "f_qr": IJKD5_ARGS(h5_name="TENDENCY_LOC_CLD", units="s^-1", index=2),
        "f_qs": IJKD5_ARGS(h5_name="TENDENCY_LOC_CLD", units="s^-1", index=3),
        "f_qv": IJK_ARGS(h5_name="TENDENCY_LOC_Q", units="s^-1"),
        "f_t": IJK_ARGS(h5_name="TENDENCY_LOC_T", units="s^-1"),
    }
    tends = {
        name: hdf5_grid_operator.get_field(**props) for name, props in field_properties.items()
    }
    tends["time"] = REFERENCE_TIME
    return tends


def get_reference_diagnostics(hdf5_grid_operator: HDF5GridOperator) -> DataArrayDict:
    field_properties = {
        "f_covptot": IJK_ARGS(h5_name="PCOVPTOT"),
        "f_fcqlng": IJK_ARGS(h5_name="PFCQLNG", dim_k=K - 1 / 2),
        "f_fcqnng": IJK_ARGS(h5_name="PFCQNNG", dim_k=K - 1 / 2),
        "f_fcqrng": IJK_ARGS(h5_name="PFCQRNG", dim_k=K - 1 / 2),
        "f_fcqsng": IJK_ARGS(h5_name="PFCQSNG", dim_k=K - 1 / 2),
        "f_fhpsl": IJK_ARGS(h5_name="PFHPSL", dim_k=K - 1 / 2),
        "f_fhpsn": IJK_ARGS(h5_name="PFHPSN", dim_k=K - 1 / 2),
        "f_fplsl": IJK_ARGS(h5_name="PFPLSL", dim_k=K - 1 / 2),
        "f_fplsn": IJK_ARGS(h5_name="PFPLSN", dim_k=K - 1 / 2),
        "f_fsqif": IJK_ARGS(h5_name="PFSQIF", dim_k=K - 1 / 2),
        "f_fsqitur": IJK_ARGS(h5_name="PFSQITUR", dim_k=K - 1 / 2),
        "f_fsqlf": IJK_ARGS(h5_name="PFSQLF", dim_k=K - 1 / 2),
        "f_fsqltur": IJK_ARGS(h5_name="PFSQLTUR", dim_k=K - 1 / 2),
        "f_fsqrf": IJK_ARGS(h5_name="PFSQRF", dim_k=K - 1 / 2),
        "f_fsqsf": IJK_ARGS(h5_name="PFSQSF", dim_k=K - 1 / 2),
        "f_rainfrac_toprfz": IJ_ARGS(h5_name="PRAINFRAC_TOPRFZ"),
    }
    diags = {
        name: hdf5_grid_operator.get_field(**props) for name, props in field_properties.items()
    }
    diags["time"] = REFERENCE_TIME
    return diags
