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
from functools import cached_property
from itertools import repeat
import numpy as np
import sys
from typing import TYPE_CHECKING

from ifs_physics_common.components import ImplicitTendencyComponent
from ifs_physics_common.grid import I, J, K
from ifs_physics_common.storage import managed_temporary_storage
from ifs_physics_common.numpyx import assign

if TYPE_CHECKING:
    from datetime import timedelta

    from cloudsc_gt4py.iox import YoecldpParams, YoethfParams, YomcstParams, YrecldpParams
    from ifs_physics_common.config import GT4PyConfig
    from ifs_physics_common.grid import ComputationalGrid
    from ifs_physics_common.typingx import NDArrayLikeDict, PropertyDict


class Cloudsc(ImplicitTendencyComponent):
    def __init__(
        self,
        computational_grid: ComputationalGrid,
        yoecldp_params: YoecldpParams,
        yoethf_params: YoethfParams,
        yomcst_params: YomcstParams,
        yrecldp_params: YrecldpParams,
        *,
        enable_checks: bool = True,
        gt4py_config: GT4PyConfig,
    ) -> None:
        super().__init__(computational_grid, enable_checks=enable_checks, gt4py_config=gt4py_config)

        self.nlev = self.computational_grid.grids[I, J, K].shape[2]
        externals = {}
        externals.update(yoecldp_params.dict())
        externals.update(yoethf_params.dict())
        externals.update(yomcst_params.dict())
        externals.update(yrecldp_params.dict())
        externals.update(
            {
                "DEPICE": 1,
                "EPSEC": 1e-14,
                "EPSILON": 100 * sys.float_info.epsilon,
                "EVAPRAIN": 2,
                "EVAPSNOW": 1,
                "FALLQV": False,
                "FALLQL": False,
                "FALLQI": False,
                "FALLQR": True,
                "FALLQS": True,
                "MELTQV": -99,
                "MELTQL": yoecldp_params.NCLDQI,
                "MELTQI": yoecldp_params.NCLDQR,
                "MELTQR": yoecldp_params.NCLDQS,
                "MELTQS": yoecldp_params.NCLDQR,
                "NLEV": self.nlev,
                "PHASEQV": 0,
                "PHASEQL": 1,
                "PHASEQI": 2,
                "PHASEQR": 1,
                "PHASEQS": 2,
                "RDCP": yomcst_params.RD / yomcst_params.RCPD,
                "RLDCP": 1 / (yoethf_params.RALSDCP - yoethf_params.RALVDCP),
                "TW1": 1329.31,
                "TW2": 0.0074615,
                "TW3": 0.85e5,
                "TW4": 40.637,
                "TW5": 275.0,
                "VQV": 0.0,
                "VQL": 0.0,
                "VQI": yrecldp_params.RVICE,
                "VQR": yrecldp_params.RVRAIN,
                "VQS": yrecldp_params.RVSNOW,
                "WARMRAIN": 2,
            }
        )

        self.cloudsc = self.compile_stencil("cloudsc", externals)

    @cached_property
    def input_grid_properties(self) -> PropertyDict:
        # todo(stubbiali): sort out units
        return {
            "b_convection_on": {"grid_dims": (I, J), "dtype_name": "bool", "units": ""},
            "f_a": {"grid_dims": (I, J, K), "units": ""},
            "f_ap": {"grid_dims": (I, J, K), "units": ""},
            "f_aph": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_ccn": {"grid_dims": (I, J, K), "units": ""},
            "f_hrlw": {"grid_dims": (I, J, K), "units": ""},
            "f_hrsw": {"grid_dims": (I, J, K), "units": ""},
            "f_icrit_aer": {"grid_dims": (I, J, K), "units": ""},
            "f_lcrit_aer": {"grid_dims": (I, J, K), "units": ""},
            "f_lsm": {"grid_dims": (I, J), "units": ""},
            "f_lu": {"grid_dims": (I, J, K), "units": ""},
            "f_lude": {"grid_dims": (I, J, K), "units": ""},
            "f_mfd": {"grid_dims": (I, J, K), "units": ""},
            "f_mfu": {"grid_dims": (I, J, K), "units": ""},
            "f_nice": {"grid_dims": (I, J, K), "units": ""},
            "f_qi": {"grid_dims": (I, J, K), "units": ""},
            "f_ql": {"grid_dims": (I, J, K), "units": ""},
            "f_qr": {"grid_dims": (I, J, K), "units": ""},
            "f_qs": {"grid_dims": (I, J, K), "units": ""},
            "f_qv": {"grid_dims": (I, J, K), "units": ""},
            "f_re_ice": {"grid_dims": (I, J, K), "units": ""},
            "f_snde": {"grid_dims": (I, J, K), "units": ""},
            "f_supsat": {"grid_dims": (I, J, K), "units": ""},
            "f_t": {"grid_dims": (I, J, K), "units": ""},
            "f_tnd_tmp_a": {"grid_dims": (I, J, K), "units": ""},
            "f_tnd_tmp_qi": {"grid_dims": (I, J, K), "units": ""},
            "f_tnd_tmp_ql": {"grid_dims": (I, J, K), "units": ""},
            "f_tnd_tmp_qr": {"grid_dims": (I, J, K), "units": ""},
            "f_tnd_tmp_qs": {"grid_dims": (I, J, K), "units": ""},
            "f_tnd_tmp_qv": {"grid_dims": (I, J, K), "units": ""},
            "f_tnd_tmp_t": {"grid_dims": (I, J, K), "units": ""},
            "f_vfi": {"grid_dims": (I, J, K), "units": ""},
            "f_vfl": {"grid_dims": (I, J, K), "units": ""},
            "f_w": {"grid_dims": (I, J, K), "units": ""},
            "i_convection_type": {"grid_dims": (I, J), "dtype_name": "int", "units": ""},
        }

    @cached_property
    def tendency_grid_properties(self) -> PropertyDict:
        # todo(stubbiali): sort out units
        return {
            "f_a": {"grid_dims": (I, J, K), "units": "s^-1"},
            "f_t": {"grid_dims": (I, J, K), "units": "s^-1"},
            "f_qv": {"grid_dims": (I, J, K), "units": "s^-1"},
            "f_ql": {"grid_dims": (I, J, K), "units": "s^-1"},
            "f_qi": {"grid_dims": (I, J, K), "units": "s^-1"},
            "f_qr": {"grid_dims": (I, J, K), "units": "s^-1"},
            "f_qs": {"grid_dims": (I, J, K), "units": "s^-1"},
        }

    @cached_property
    def diagnostic_grid_properties(self) -> PropertyDict:
        # todo(stubbiali): sort out units
        return {
            "f_covptot": {"grid_dims": (I, J, K), "units": ""},
            "f_fcqlng": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_fcqnng": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_fcqrng": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_fcqsng": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_fhpsl": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_fhpsn": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_fplsl": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_fplsn": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_fsqif": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_fsqitur": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_fsqlf": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_fsqltur": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_fsqrf": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_fsqsf": {"grid_dims": (I, J, K - 1 / 2), "units": ""},
            "f_rainfrac_toprfz": {"grid_dims": (I, J), "units": ""},
        }

    def array_call(
        self,
        state: NDArrayLikeDict,
        timestep: timedelta,
        out_tendencies: NDArrayLikeDict,
        out_diagnostics: NDArrayLikeDict,
        overwrite_tendencies: dict[str, bool],
    ) -> None:
        with managed_temporary_storage(
            self.computational_grid,
            *repeat(((I, J), "float"), 6),
            ((I, J), "bool"),
            ((K - 1 / 2,), "int"),
            gt4py_config=self.gt4py_config,
        ) as (aph_s, cldtopdist, covpmax, covptot, paphd, trpaus, rainliq, klevel):
            inputs = {
                "in_" + name.split("_", maxsplit=1)[1]: state[name]
                for name in self.input_properties
            }
            tendencies = {
                "out_tnd_loc_" + name.split("_", maxsplit=1)[1]: out_tendencies[name]
                for name in self.tendency_properties
            }
            diagnostics = {
                "out_" + name.split("_", maxsplit=1)[1]: out_diagnostics[name]
                for name in self.diagnostic_properties
            }
            temporaries = {
                "tmp_aph_s": aph_s,
                "tmp_cldtopdist": cldtopdist,
                "tmp_covpmax": covpmax,
                "tmp_covptot": covptot,
                "tmp_klevel": klevel,
                "tmp_paphd": paphd,
                "tmp_rainliq": rainliq,
                "tmp_trpaus": trpaus,
            }
            assign(klevel, np.arange(self.nlev + 1))
            self.cloudsc(
                **inputs,
                **tendencies,
                **diagnostics,
                **temporaries,
                dt=self.gt4py_config.dtypes.float(timestep.total_seconds()),
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K - 1 / 2].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )
