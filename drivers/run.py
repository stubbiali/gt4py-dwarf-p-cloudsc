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
import click
from typing import TYPE_CHECKING

from cloudsc_gt4py.physics.cloudsc import Cloudsc
from cloudsc_gt4py.physics.cloudsc_split import CloudscSplit
from cloudsc_gt4py.initialization.reference import (
    get_reference_tendencies,
    get_reference_diagnostics,
)
from cloudsc_gt4py.initialization.state import get_state
from cloudsc_gt4py.utils.iox import HDF5Reader
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.utils.output import (
    print_performance,
    write_performance_to_csv,
    write_stencils_performance_to_csv,
)
from ifs_physics_common.utils.timing import timing
from ifs_physics_common.utils.validation import validate

if TYPE_CHECKING:
    from typing import Literal, Optional, Type

    from ifs_physics_common.framework.config import IOConfig

    from .config import Config, DEFAULT_CONFIG, DEFAULT_IO_CONFIG
else:
    from config import Config, DEFAULT_CONFIG, DEFAULT_IO_CONFIG


def core(config: Config, io_config: IOConfig, cloudsc_cls: Type) -> None:
    hdf5_reader = HDF5Reader(config.input_file, config.data_types)

    nx = config.num_cols or hdf5_reader.get_nlon()
    nz = hdf5_reader.get_nlev()
    computational_grid = ComputationalGrid(nx, 1, nz)

    state = get_state(computational_grid, hdf5_reader, gt4py_config=config.gt4py_config)
    dt = hdf5_reader.get_timestep()

    yoecldp_paramaters = hdf5_reader.get_yoecldp_parameters()
    yoethf_parameters = hdf5_reader.get_yoethf_parameters()
    yomcst_parameters = hdf5_reader.get_yomcst_parameters()
    yrecldp_parameters = hdf5_reader.get_yrecldp_parameters()

    cloudsc = cloudsc_cls(
        computational_grid,
        yoecldp_paramaters,
        yoethf_parameters,
        yomcst_parameters,
        yrecldp_parameters,
        enable_checks=config.sympl_enable_checks,
        gt4py_config=config.gt4py_config,
    )
    tends, diags = cloudsc(state, dt)

    config.gt4py_config.reset_exec_info()

    runtime_l = []
    for i in range(config.num_runs):
        with timing(f"run_{i}") as timer:
            cloudsc(state, dt, out_tendencies=tends, out_diagnostics=diags)
        runtime_l.append(timer.get_time(f"run_{i}"))

    runtime_mean, runtime_stddev, mflops_mean, mflops_stddev = print_performance(nx, runtime_l)

    if io_config.output_csv_file is not None:
        write_performance_to_csv(
            io_config.output_csv_file,
            io_config.host_name,
            config.precision,
            config.gt4py_config.backend,
            nx,
            config.num_threads,
            1,
            config.num_runs,
            runtime_mean,
            runtime_stddev,
            mflops_mean,
            mflops_stddev,
        )

    if config.enable_validation:
        hdf5_reader_ref = HDF5Reader(config.reference_file, config.data_types)
        tends_ref = get_reference_tendencies(
            computational_grid, hdf5_reader_ref, gt4py_config=config.gt4py_config
        )
        diags_ref = get_reference_diagnostics(
            computational_grid, hdf5_reader_ref, gt4py_config=config.gt4py_config
        )
        print("\n== Validation:")
        validate(tends, tends_ref, atol=config.atol, rtol=config.rtol)
        validate(diags, diags_ref, atol=config.atol, rtol=config.rtol)


@click.command()
@click.option(
    "--backend",
    type=str,
    default="numpy",
    help="GT4Py backend (options: dace:cpu, dace:gpu, gt:cpu_ifirst, gt:cpu_kfirst, gt:gpu, "
    "numpy; default: numpy).",
)
@click.option(
    "--enable-checks/--disable-checks",
    is_flag=True,
    type=bool,
    default=False,
    help="Enable/disable sanity checks performed by Sympl and GT4Py (default: enabled).",
)
@click.option(
    "--enable-validation/--disable-validation",
    is_flag=True,
    type=bool,
    default=True,
    help="Enable/disable data validation (default: enabled).",
)
@click.option("--num-cols", type=int, default=1, help="Number of domain columns (default: 1).")
@click.option("--num-runs", type=int, default=1, help="Number of executions (default: 1).")
@click.option(
    "--precision",
    type=str,
    default="double",
    help="Select either `double` (default) or `single` precision.",
)
@click.option(
    "--variant",
    type=str,
    default="fused",
    help="The stencil variant (options: fused, split; default: fused).",
)
@click.option("--host-alias", type=str, default=None, help="Name of the host machine (optional).")
@click.option(
    "--output-csv-file",
    type=str,
    default=None,
    help="Path to the CSV file where writing performance counters (optional).",
)
@click.option(
    "--output-csv-file-stencils",
    type=str,
    default=None,
    help="Path to the CSV file where writing performance counters for each stencil (optional).",
)
@click.option("--atol", type=float, default=None, help="Absolute tolerance used in validation.")
@click.option("--rtol", type=float, default=None, help="Relative tolerance used in validation.")
def main(
    backend: str,
    enable_checks: bool,
    enable_validation: bool,
    num_cols: int,
    num_runs: int,
    precision: Literal["double", "single"],
    variant: Literal["fused", "split"],
    host_alias: Optional[str],
    output_csv_file: Optional[str],
    output_csv_file_stencils: Optional[str],
    atol: Optional[float],
    rtol: Optional[float],
) -> None:
    config = (
        DEFAULT_CONFIG.with_precision(precision)
        .with_backend(backend)
        .with_checks(enable_checks)
        .with_validation(enable_validation, atol, rtol)
        .with_num_cols(num_cols)
        .with_num_runs(num_runs)
    )
    io_config = DEFAULT_IO_CONFIG.with_output_csv_file(output_csv_file).with_host_name(host_alias)
    cloudsc_cls = CloudscSplit if variant == "split" else Cloudsc
    core(config, io_config, cloudsc_cls=cloudsc_cls)

    if output_csv_file_stencils is not None:
        write_stencils_performance_to_csv(
            output_csv_file_stencils,
            io_config.host_name,
            config.precision,
            config.gt4py_config.backend,
            config.num_cols,
            config.num_threads,
            config.num_runs,
            config.gt4py_config.exec_info,
            key_patterns=["fluxes", "tendencies"] if variant == "split" else ["cloudsc"],
        )


if __name__ == "__main__":
    main()
