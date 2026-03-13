"""StorAge Selection (SAS) transport model.

This module implements the :class:`Model` class, the main entry point for
building and running SAS models.  A Model wraps:

- a timeseries DataFrame of influx / outflux data,
- SAS function specifications (one per outflux), and
- optional solute transport parameters.

Call :meth:`Model.run` to invoke the solver; results are then
available through accessor methods (``get_sT``, ``get_pQ``, etc.).
"""

from __future__ import annotations

import copy
import json
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from mesas.sas.specs import SAS_Spec

from ._solve_numba import solve

dtype = np.float64


def _processinputs(input: dict | str) -> dict:
    """Load model input from a dict or a JSON file path.

    Parameters
    ----------
    input : dict or str
        Either a dict (returned as-is) or a path to a ``.json`` file.

    Returns
    -------
    dict
        The parsed input dictionary.
    """
    if isinstance(input, dict):
        return input
    elif isinstance(input, str):
        if os.path.isfile(input):
            with open(input) as json_file:
                data = json.load(json_file)
            return data
        else:
            raise FileNotFoundError(f"I could not find the input file: {input}")
    else:
        raise TypeError("input must be either a dict, or a path to a .json file")


class Model:
    """StorAge Selection (SAS) transport model.

    Construct with timeseries data, SAS function specifications, and
    (optionally) solute parameters, then call :meth:`run` to execute the
    Fortran solver.

    Parameters
    ----------
    data_df : pd.DataFrame or str
        Timeseries data, or a path to a CSV file.  Must contain at least
        the influx column (default ``"J"``) and one outflux column per
        entry in *sas_specs*.
    config : dict or str, optional
        Configuration dict or path to a JSON config file.  May contain
        ``"options"``, ``"sas_specs"``, and ``"solute_parameters"`` keys.
    sas_specs : dict or str, optional
        SAS specification dict (one key per outflux) or JSON path.
    solute_parameters : dict or str, optional
        Solute transport parameters or JSON path.
    **kwargs
        Any model option (``dt``, ``verbose``, ``num_scheme``, etc.)
        passed as keyword arguments overrides config-file values.
    """

    def __init__(
        self,
        data_df: pd.DataFrame | str,
        config: dict | str | None = None,
        sas_specs: dict | str | None = None,
        solute_parameters: dict | str | None = None,
        **kwargs: Any,
    ) -> None:
        # defaults
        self._result = None
        self.jacobian = {}
        # load the timeseries data
        self.data_df = data_df
        # check for a configuration file
        if config:
            config = _processinputs(config)
        # process any options
        self._default_options = {
            "dt": 1,
            "verbose": False,
            "num_scheme": 4,
            "debug": False,
            "warning": True,
            "jacobian": False,
            "n_substeps": 1,
            "max_age": None,
            "sT_init": None,
            "influx": "J",
            "ST_smallest_segment": 1.0 / 100,
            "ST_largest_segment": np.inf,
            "record_state": False,
        }
        self._options = self._default_options
        components_to_learn = kwargs.get("components_to_learn")
        if config and "options" in config.keys():
            self.options = config["options"]
        self.options = kwargs
        # get the SAS specification
        if config and "sas_specs" in config.keys():
            sas_specs = config["sas_specs"]
        elif sas_specs:
            sas_specs = _processinputs(sas_specs)
        else:
            raise ValueError("No SAS specification found!")
        self.sas_specs = self.parse_sas_specs(sas_specs)
        self._numflux = len(self.sas_specs)
        self._fluxorder = list(self.sas_specs.keys())
        # get solute transport parameters
        if config and "solute_parameters" in config.keys():
            solute_parameters = config["solute_parameters"]
        elif solute_parameters:
            solute_parameters = _processinputs(solute_parameters)
        # defaults for solute transport
        self._default_parameters = {
            "mT_init": 0.0,
            "C_old": 0.0,
            "k1": 0.0,
            "C_eq": 0.0,
            "alpha": dict((flux, 1.0) for flux in self._fluxorder),
            "observations": {},
        }
        self.solute_parameters = solute_parameters
        self.components_to_learn = components_to_learn

    def __repr__(self):
        """Creates a repr for the model"""
        result = ""
        for flux, sas_spec in self.sas_specs.items():
            result += f"flux = {flux}\n"
            result += sas_spec.__repr__()
        return result

    def parse_sas_specs(self, sas_specs: dict) -> dict[str, SAS_Spec]:
        """Validate and convert raw SAS spec dicts into :class:`SAS_Spec` objects."""
        for flux, spec_in in sas_specs.items():
            spec = deepcopy(spec_in)
            assert flux in self.data_df.columns
            if isinstance(spec, SAS_Spec):
                sas_specs[flux] = spec
            else:
                assert isinstance(spec, dict)
                assert all([isinstance(component_spec, dict) for component_spec in spec.values()])
                sas_specs[flux] = SAS_Spec(spec, self.data_df)
        return sas_specs

    def copy_without_results(self) -> Model:
        """Return a deep copy of this model with results cleared."""
        return Model(
            copy.deepcopy(self._data_df),
            copy.deepcopy(self._sas_specs),
            copy.deepcopy(self._solute_parameters),
            copy.deepcopy(self._components_to_learn),
            **copy.deepcopy(self._options),
        )

    def subdivided_copy(self, flux, label, segment):
        """
        Creates a copy of the model with one segment of a sas function subdivided in two

        :param flux: name of the flux
        :param label: name of the component
        :param segment: segment (numbered from 0 for the youngest)
        :return: a new Model instance with one piecewise segment subdivided
        """

        # make a copy
        new_model = self.copy_without_results()

        # subdivide the component
        new_model.sas_specs[flux] = new_model.sas_specs[flux].subdivided_copy(label, segment)

        return new_model

    @property
    def fluxorder(self):
        return self._fluxorder

    @fluxorder.setter
    def fluxorder(self, new_fluxorder):
        raise AttributeError("fluxorder property is read-only")

    @property
    def solorder(self):
        return self._solorder

    @solorder.setter
    def solorder(self, new_solorder):
        raise AttributeError("solorder property is read-only")

    @property
    def result(self):
        """Results of running the sas model with the current parameters. See :ref:`results`"""
        if self._result is None:
            raise AttributeError(
                "results are only defined once the model is run. Use .run() method to generate results "
            )
        else:
            return self._result

    @result.setter
    def result(self, result):
        raise AttributeError("Model results are read-only. Use .run() method to generate results ")

    @property
    def data_df(self):
        """Pandas dataframe holding the model inputs. See :ref:`inputs`"""
        return self._data_df

    @data_df.setter
    def data_df(self, new_data_df):
        if isinstance(new_data_df, pd.core.frame.DataFrame):
            self._data_df = new_data_df.copy()
        elif isinstance(new_data_df, str):
            self._data_df = pd.read_csv(new_data_df)
        else:
            raise TypeError("data_df must be either a pandas dataframe, or a path to a .csv file")
        self._timeseries_length = len(self._data_df)

    @property
    def options(self):
        """Options for running the model. See :ref:`options`"""
        return self._options

    @options.setter
    def options(self, new_options):
        new_options = _processinputs(new_options)
        invalid_options = [optkey for optkey in new_options.keys() if optkey not in self._default_options.keys()]
        if any(invalid_options):
            raise KeyError("Invalid options: {}".format(invalid_options))
        self._options.update(new_options)
        if self._options["max_age"] is None:
            self._options["max_age"] = self._timeseries_length
        assert self._options["max_age"] <= self._timeseries_length
        if self._options["sT_init"] is None:
            self._options["sT_init"] = np.zeros(self._options["max_age"])
        else:
            self._options["max_age"] = len(self._options["sT_init"])
        self._max_age = self._options["max_age"]
        if self._options["record_state"] is False:
            self._index_ts = np.array([self._timeseries_length - 1])
        elif self._options["record_state"] is True:
            self._index_ts = np.arange(self._timeseries_length)
        else:
            self._index_ts = np.where(self.data_df[self._options["record_state"]])[0]

    @property
    def sas_specs(self):
        """Specification of the SAS functions. See :ref:`sasspec`"""
        return self._sas_specs

    @sas_specs.setter
    def sas_specs(self, new_sas_specs):
        self._sas_specs = _processinputs(new_sas_specs)
        self._numflux = len(self._sas_specs)
        self._fluxorder = list(self._sas_specs.keys())

    def set_sas_spec(self, flux: str, sas_spec: SAS_Spec) -> None:
        """Replace the SAS specification for a given flux."""
        self._sas_specs[flux] = sas_spec
        self._sas_specs[flux].make_spec_ts(self.data_df)

    def set_component(self, flux: str, component: Any) -> None:
        """Replace a single component within a flux's SAS specification."""
        label = component.label
        self._sas_specs[flux].components[label] = component
        self._sas_specs[flux].make_spec_ts()

    def set_sas_fun(self, flux: str, label: str, sas_fun: Any) -> None:
        """Replace the SAS function of a named component."""
        self._sas_specs[flux].components[label].sas_fun = sas_fun
        self._sas_specs[flux].make_spec_ts()

    def get_component_labels(self) -> dict[str, list[str]]:
        """Return ``{flux: [label, ...]}`` for all SAS components."""
        component_labels = {}
        for flux in self._fluxorder:
            component_labels[flux] = list(self._sas_specs[flux].components.keys())
        return component_labels

    @property
    def components_to_learn(self):
        """
        A dictionary of lists giving the component labels that you want to train with MESAS

        The components in this dictionary can be set by assigning a dictionary. For example, this would
        include only the 'min' and 'max' components of the sas spec associated with the 'Q' flux.

            >>> model.components_to_learn = {'Q':['min', 'max']}

        The parameters associated with these components can be obtained as a 1-D array with:

            >>> seglist = model.get_parameter_list()

        and modified using:

            >>> model.update_from_parameter_list(seglst)

        :return:
        """
        return self._components_to_learn

    @components_to_learn.setter
    def components_to_learn(self, label_dict):
        if label_dict is not None:
            self._components_to_learn = OrderedDict(
                (flux, [label for label in self._sas_specs[flux]._componentorder if label in label_dict[flux]])
                for flux in self._fluxorder
                if flux in label_dict.keys()
            )
            self._comp2learn_fluxorder = list(self._components_to_learn.keys())
            for flux in self._comp2learn_fluxorder:
                self._sas_specs[flux]._comp2learn_componentorder = self._components_to_learn[flux]

    def get_parameter_list(self):
        """
        Returns a concatenated list of segments from self.components_to_learn
        """
        return np.concatenate([self.sas_specs[flux].get_parameter_list() for flux in self._comp2learn_fluxorder])

    def update_from_parameter_list(self, parameter_list):
        """
        Modifies the components in self.components_to_learn from a concatenated list of segments
        """
        starti = 0
        for flux in self._comp2learn_fluxorder:
            nparams = len(self.sas_specs[flux].get_parameter_list())
            self.sas_specs[flux].update_from_parameter_list(parameter_list[starti : starti + nparams])
            starti += nparams

    @property
    def solute_parameters(self):
        """Parameters describing solute behavior. See :ref:`solspec`"""
        return self._solute_parameters

    @solute_parameters.setter
    def solute_parameters(self, new_solute_parameters):
        # set parameters for solute transport
        # provide defaults if absent
        if new_solute_parameters is not None:
            new_solute_parameters = _processinputs(new_solute_parameters)
            self._solute_parameters = {}
            for sol, params in new_solute_parameters.items():
                self._solute_parameters[sol] = deepcopy(self._default_parameters)
                self.set_solute_parameters(sol, params)
            self._numsol = len(self._solute_parameters)
            self._solorder = list(self._solute_parameters.keys())
        else:
            self._numsol = 0
            self._solute_parameters = {}
            self._solorder = []

    def set_solute_parameters(self, sol: str, params: dict) -> None:
        invalid_parameters = [paramkey for paramkey in params.keys() if paramkey not in self._default_parameters.keys()]
        if any(invalid_parameters):
            raise KeyError("invalid parameters for {}: {}".format(sol, invalid_parameters))
        self._solute_parameters[sol].update(params)

    def _create_solute_inputs(self):
        numsol = max(self._numsol, 1)
        C_J = np.zeros((self._timeseries_length, numsol), dtype=dtype)
        mT_init = np.zeros((self._max_age, numsol), dtype=dtype)
        C_old = np.zeros(numsol, dtype=dtype)
        k1 = np.zeros((self._timeseries_length, numsol), dtype=dtype)
        C_eq = np.zeros((self._timeseries_length, numsol), dtype=dtype)
        alpha = np.ones((self._timeseries_length, self._numflux, numsol), dtype=dtype)
        if self.solute_parameters is not None:

            def _get_array(param, N):
                if param in self.data_df:
                    return self.data_df[param].values
                else:
                    return param * np.ones(N)

            for isol, sol in enumerate(self._solorder):
                C_J[:, isol] = self.data_df[sol].values
                C_old[isol] = self.solute_parameters[sol]["C_old"]
                mT_init[:, isol] = _get_array(self.solute_parameters[sol]["mT_init"], self._max_age)
                k1[:, isol] = _get_array(self.solute_parameters[sol]["k1"], self._timeseries_length)
                C_eq[:, isol] = _get_array(self.solute_parameters[sol]["C_eq"], self._timeseries_length)
                for iflux, flux in enumerate(self._fluxorder):
                    alpha[:, iflux, isol] = _get_array(
                        self.solute_parameters[sol]["alpha"][flux], self._timeseries_length
                    )
        return C_J, mT_init, C_old, alpha, k1, C_eq

    def _create_sas_lookup(self):
        """Build flattened SAS lookup tables for the Fortran solver.

        The Fortran solver expects all SAS function parameters packed into
        flat arrays.  This method iterates over fluxes and their components
        to produce:

        - ``SAS_args``: breakpoint ST values, shape ``(nC_total, max_nargs)``
        - ``P_list``: breakpoint P values, same shape
        - ``weights``: component blending weights, shape ``(timeseries_length, nC_total)``
        - ``component_type``: int type code per component (0=piecewise, 1+=continuous)
        - ``nC_list``: number of components per flux
        - ``nargs_list``: number of breakpoints per component
        """
        nC_list = []  # number of components for each flux
        component_list = []  # flat list of all Component objects across all fluxes
        for flux in self._fluxorder:
            nC_list.append(len(self._sas_specs[flux].components))
            for component in self._sas_specs[flux]._componentorder:
                component_list.append(self.sas_specs[flux].components[component])
        nargs_list = [len(component.argsS) for component in component_list]
        component_type = [component.type for component in component_list]
        nC_total = np.sum(nC_list)
        nargs_total = np.sum(nargs_list)
        # Pack breakpoint arrays: each row is one component's ST or P values
        SAS_args = np.column_stack([[component.argsS] for component in component_list]).T
        P_list = np.column_stack([[component.argsP] for component in component_list]).T
        # Blending weights: shape (timeseries_length, nC_total)
        weights = np.column_stack([component.weights for component in component_list])
        return SAS_args, P_list, weights, component_type, nC_list, nC_total, nargs_list, nargs_total

    def run(self) -> None:
        """Execute the SAS model with current parameters.

        Calls the solver (``solve``) and stores results in
        ``self._result``.  Predicted outflux concentrations are also
        written back into ``self.data_df`` as columns named
        ``"<solute> --> <flux>"``.

        Results are accessible via :attr:`result` and the ``get_*`` methods.
        """
        # --- 1. Extract water flux timeseries ---
        J = self.data_df[self.options["influx"]].values  # influx [T]
        Q = self.data_df[self._fluxorder].values  # outfluxes [T, numflux]
        sT_init = self.options["sT_init"]
        timeseries_length = self._timeseries_length
        numflux = self._numflux

        # --- 2. Build SAS lookup tables (flattened for Fortran) ---
        SAS_args, P_list, weights, component_type, nC_list, nC_total, nargs_list, nargs_total = (
            self._create_sas_lookup()
        )
        SAS_args = np.ascontiguousarray(SAS_args)
        P_list = np.ascontiguousarray(P_list)
        weights = np.ascontiguousarray(weights)

        # --- 3. Build solute input arrays ---
        C_J, mT_init, C_old, alpha, k1, C_eq = self._create_solute_inputs()
        numsol = max(self._numsol, 1)

        # --- 4. Unpack scalar options ---
        dt = self.options["dt"]
        verbose = self.options["verbose"]
        debug = self.options["debug"]
        warning = self.options["warning"]
        jacobian = self.options["jacobian"]
        n_substeps = self.options["n_substeps"]
        max_age = self.options["max_age"]
        num_scheme = self.options["num_scheme"]

        # Which timesteps to record (depends on record_state option)
        index_ts = self._index_ts

        # Truncate initial conditions to max_age
        sT_init = sT_init[:max_age]
        mT_init = mT_init[:max_age, :]

        # --- 5. Call the solver ---
        # The solver expects ~30 positional arguments in a specific order.
        # SAS_args/P_list are transposed (numargs_total, timeseries_length).
        fresult = solve(
            J,  # influx timeseries
            Q,  # outflux timeseries [T, numflux]
            np.ascontiguousarray(SAS_args.T),  # SAS breakpoint ST values
            np.ascontiguousarray(P_list.T),  # SAS breakpoint P values
            weights,  # component blending weights
            sT_init,  # initial age-ranked storage
            dt,  # timestep size
            verbose,
            debug,
            warning,
            jacobian,  # flags
            mT_init,  # initial age-ranked solute mass
            C_J,  # influx concentrations
            alpha,  # evapoconcentration factors
            k1,  # first-order reaction rates
            C_eq,  # equilibrium concentrations
            C_old,  # concentration of pre-initial water
            n_substeps,  # sub-timesteps per full step
            component_type,  # int type code per component
            nC_list,  # num components per flux
            nargs_list,  # num breakpoints per component
            index_ts,  # which timesteps to record
            num_scheme,  # RK scheme (1=Euler, 2=RK2, 4=RK4)
            numflux,
            numsol,
            max_age,
            timeseries_length,
            len(index_ts),
            nC_total,
            nargs_total,
        )
        sT, pQ, WaterBalance, mT, mQ, mR, C_Q, dsTdSj, dmTdSj, dCdSj, SoluteBalance = fresult

        # --- 6. Store results ---
        # Solver arrays use (timestep, ..., age) ordering; moveaxis converts
        # the last axis (age) to the first so Python arrays are (age, timestep, ...).
        if self._numsol > 0:
            self._result = {"C_Q": C_Q}
            # Write predicted concentrations back into the DataFrame
            for isol, sol in enumerate(self._solorder):
                for iflux, flux in enumerate(self._fluxorder):
                    colname = sol + " --> " + flux
                    self._data_df[colname] = C_Q[:, iflux, isol]
        else:
            self._result = {}
        self._result.update(
            {
                "sT": np.moveaxis(sT, -1, 0),  # age-ranked storage density
                "pQ": np.moveaxis(pQ, -1, 0),  # age-ranked outflux probability
                "WaterBalance": np.moveaxis(WaterBalance, -1, 0),  # conservation residual
                "dsTdSj": np.moveaxis(dsTdSj, -1, 0),  # Jacobian of sT w.r.t. SAS params
            }
        )
        if self._numsol > 0:
            self._result.update(
                {
                    "mT": np.moveaxis(mT, -1, 0),  # age-ranked solute mass density
                    "mQ": np.moveaxis(mQ, -1, 0),  # age-ranked solute outflux
                    "mR": np.moveaxis(mR, -1, 0),  # age-ranked reaction mass
                    "SoluteBalance": np.moveaxis(SoluteBalance, -1, 0),
                    "dmTdSj": np.moveaxis(dmTdSj, -1, 0),
                    "dCdSj": dCdSj,
                }
            )

    def get_jacobian(self, mode="segment", logtransform=True):
        J = None
        self.jacobian = {}
        for isol, sol in enumerate(self._solorder):
            if "observations" in self.solute_parameters[sol]:
                self.jacobian[sol] = {}
                param_offset = 0
                for isolflux, solflux in enumerate(self._fluxorder):
                    if solflux in self.solute_parameters[sol]["observations"]:
                        J_seg = None
                        for iflux, flux in enumerate(self._comp2learn_fluxorder):
                            for label in self._components_to_learn[flux]:
                                n_breakpoints = len(self.sas_specs[flux].components[label].sas_fun.P)
                                J_S = np.squeeze(
                                    self.result["dCdSj"][:, param_offset : param_offset + n_breakpoints, isolflux, isol]
                                )
                                if mode == "endpoint":
                                    pass
                                elif mode == "segment":
                                    # To get the derivative with respect to the segment length, we add up the derivative w.r.t. the
                                    # endpoints that would be displaced by varying that segment
                                    endpoint_to_segment = np.triu(np.ones(n_breakpoints), k=0)
                                    J_S = np.dot(endpoint_to_segment, J_S.T).T
                                    if logtransform:
                                        J_S = J_S * self.sas_specs[flux].components[label].sas_fun._parameter_list
                                if J_seg is None:
                                    J_seg = J_S
                                else:
                                    J_seg = np.c_[J_seg, J_S]
                            PQ = np.sum(self.result["pQ"][:, :, iflux], axis=0)
                            J_old = 1 - PQ
                            J_old_sol = np.zeros((self._timeseries_length, self._numsol))
                            J_old_sol[:, list(self._solorder).index(sol)] = J_old.T
                            J_sol = np.c_[J_seg, J_old_sol]
                        if J is None:
                            J = J_sol
                        else:
                            J = np.concatenate((J, J_sol), axis=0)
                        self.jacobian[sol][flux] = {}
                        self.jacobian[sol][flux]["seg"] = J_seg
                        self.jacobian[sol][flux]["C_old"] = J_old
                    param_offset += n_breakpoints
        return J

    def get_residuals(self):
        residuals = None
        for isol, sol in enumerate(self._solorder):
            if "observations" in self.solute_parameters[sol]:
                for iflux, flux in enumerate(self._comp2learn_fluxorder):
                    if flux in self.solute_parameters[sol]["observations"]:
                        obs = self.solute_parameters[sol]["observations"][flux]
                        C_obs = self.data_df[obs]
                        iflux = list(self._fluxorder).index(flux)
                        isol = list(self._solorder).index(sol)
                        this_residual = self.result["C_Q"][:, iflux, isol] - C_obs.values
                        if residuals is None:
                            residuals = this_residual
                        else:
                            residuals = np.concatenate((residuals, this_residual), axis=0)
                        self.data_df[f"residual {flux}, {sol}, {obs}"] = this_residual
        return residuals

    def get_obs_index(self):
        index = None
        for isol, sol in enumerate(self._solorder):
            if "observations" in self.solute_parameters[sol]:
                for iflux, flux in enumerate(self._comp2learn_fluxorder):
                    if flux in self.solute_parameters[sol]["observations"]:
                        obs = self.solute_parameters[sol]["observations"][flux]
                        C_obs = self.data_df[obs]
                        this_index = ~np.isnan(C_obs.values)
                        if index is None:
                            index = this_index
                        else:
                            index = np.concatenate((index, this_index), axis=0)
        return np.nonzero(index)[0]

    def _get_result(
        self, X: np.ndarray, timestep: int | None = None, agestep: int | None = None, inputtime: int | None = None
    ) -> np.ndarray:
        if timestep is not None:
            # Only one can be given
            assert agestep is None
            assert inputtime is None
            return X[:, timestep]
        if agestep is not None:
            # Only one can be given
            assert timestep is None
            assert inputtime is None
            return X[agestep, :]
        if inputtime is not None:
            # Only one can be given
            assert agestep is None
            assert timestep is None
            return np.diagonal(X, offset=inputtime)
        return X

    def get_WaterBalance(self, **kwargs) -> np.ndarray:
        """Water conservation residual array, shape ``(max_age, n_output_steps)``.

        Should be near machine precision when ``record_state=True``.
        """
        X = self.result["WaterBalance"]
        return self._get_result(X, **kwargs)

    def get_sT(self, **kwargs) -> np.ndarray:
        """Age-ranked storage density ``sT``, shape ``(max_age, n_output_steps)``."""
        X = self.result["sT"]
        return self._get_result(X, **kwargs)

    def get_pQ(self, flux: str, **kwargs) -> np.ndarray:
        """Age-ranked outflux probability ``pQ`` for a given flux."""
        iflux = list(self._fluxorder).index(flux)
        X = self.result["pQ"][:, :, iflux]
        return self._get_result(X, **kwargs)

    def get_mT(self, sol: str, **kwargs) -> np.ndarray:
        """Age-ranked solute mass density ``mT`` for a given solute."""
        isol = list(self._solorder).index(sol)
        X = self.result["mT"][:, :, isol]
        return self._get_result(X, **kwargs)

    def get_CT(self, sol: str, **kwargs) -> np.ndarray:
        """Age-ranked solute concentration ``CT = mT / sT``."""
        isol = list(self._solorder).index(sol)
        sT = self._get_result(self.result["sT"], **kwargs)
        mT = self._get_result(self.result["mT"][:, :, isol], **kwargs)
        return np.where(sT > 0, mT / sT, np.nan)

    def get_mR(self, sol: str, **kwargs) -> np.ndarray:
        """Age-ranked reaction mass ``mR`` for a given solute."""
        isol = list(self._solorder).index(sol)
        X = self.result["mR"][:, :, isol]
        return self._get_result(X, **kwargs)

    def get_SoluteBalance(self, sol: str, **kwargs) -> np.ndarray:
        """Solute conservation residual for a given solute.

        Should be near machine precision when ``record_state=True``.
        """
        isol = list(self._solorder).index(sol)
        X = self.result["SoluteBalance"][:, :, isol]
        return self._get_result(X, **kwargs)

    def get_mQ(self, flux: str, sol: str, **kwargs) -> np.ndarray:
        """Age-ranked solute outflux ``mQ`` for a given flux and solute."""
        iflux = list(self._fluxorder).index(flux)
        isol = list(self._solorder).index(sol)
        X = self.result["mQ"][:, :, iflux, isol]
        return self._get_result(X, **kwargs)

    def get_ST(self, **kwargs) -> np.ndarray:
        """Cumulative storage ``ST = cumsum(sT) * dt``."""
        sT = self.get_sT()
        ST = np.cumsum(sT, axis=0) * self.options["dt"]
        return self._get_result(ST, **kwargs)
