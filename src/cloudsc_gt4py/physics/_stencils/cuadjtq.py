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

from cloudsc_gt4py.physics._stencils.fcttre import f_foedem, f_foeewm, f_foeldcpm
from ifs_physics_common.framework.stencil import function_collection


@function_collection("f_cuadjtq_5")
@gtscript.function
def f_cuadjtq_5(qp, qsmix, t):
    from __externals__ import RETV

    qsat = min(f_foeewm(t) * qp, 0.5)
    cor = 1 / (1 - RETV * qsat)
    qsat *= cor
    cond = (qsmix - qsat) / (1 + qsat * cor * f_foedem(t))
    t += f_foeldcpm(t) * cond
    qsmix -= cond
    return qsmix, t


@function_collection("f_cuadjtq")
@gtscript.function
def f_cuadjtq(ap, qsmix, t):
    qp = 1 / ap
    qsmix, t = f_cuadjtq_5(qp, qsmix, t)
    qsmix, t = f_cuadjtq_5(qp, qsmix, t)
    return qsmix, t
