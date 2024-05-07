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

from cloudsc_gt4py.physics._stencils.fcttre import f_foeeice, f_foeeliq
from ifs_physics_common.framework.stencil import function_collection
from ifs_physics_common.utils.f2py import ported_function


@ported_function(from_file="common/include/fccld.func.h", from_line=26, to_line=27)
@function_collection("f_fokoop")
@gtscript.function
def f_fokoop(t):
    from __externals__ import RKOOP1, RKOOP2

    return min(RKOOP1 - RKOOP2 * t, f_foeeliq(t) / f_foeeice(t))
