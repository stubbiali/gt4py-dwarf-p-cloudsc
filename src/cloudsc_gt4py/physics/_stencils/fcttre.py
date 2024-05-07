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

from gt4py.cartesian import gtscript

from ifs_physics_common.framework.stencil import function_collection


@function_collection("f_foedelta")
@gtscript.function
def f_foedelta(t):
    from __externals__ import RTT

    return 1 if t > RTT else 0


@function_collection("f_foealfa")
@gtscript.function
def f_foealfa(t):
    from __externals__ import RTICE, RTWAT, RTWAT_RTICE_R

    return min(1.0, ((max(RTICE, min(RTWAT, t)) - RTICE) * RTWAT_RTICE_R) ** 2)


@function_collection("f_foeewm")
@gtscript.function
def f_foeewm(t):
    from __externals__ import R2ES, R3IES, R3LES, R4IES, R4LES, RTT

    return R2ES * (
        f_foealfa(t) * exp(R3LES * (t - RTT) / (t - R4LES))
        + (1 - f_foealfa(t)) * (exp(R3IES * (t - RTT) / (t - R4IES)))
    )


@function_collection("f_foedem")
@gtscript.function
def f_foedem(t):
    from __externals__ import R4IES, R4LES, R5ALSCP, R5ALVCP

    return f_foealfa(t) * R5ALVCP * (1 / (t - R4LES) ** 2) + (1 - f_foealfa(t)) * R5ALSCP * (
        1 / (t - R4IES) ** 2
    )


@function_collection("f_foeldcpm")
@gtscript.function
def f_foeldcpm(t):
    from __externals__ import RALSDCP, RALVDCP

    return f_foealfa(t) * RALVDCP + (1 - f_foealfa(t)) * RALSDCP


@function_collection("f_foeeliq")
@gtscript.function
def f_foeeliq(t):
    from __externals__ import R2ES, R3LES, R4LES, RTT

    return R2ES * exp(R3LES * (t - RTT) / (t - R4LES))


@function_collection("f_foeeice")
@gtscript.function
def f_foeeice(t):
    from __externals__ import R2ES, R3IES, R4IES, RTT

    return R2ES * exp(R3IES * (t - RTT) / (t - R4IES))
