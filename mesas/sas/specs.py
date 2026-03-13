from collections import OrderedDict
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from mesas.sas.functions import Continuous, Piecewise


class SAS_Spec:
    def __init__(self, spec, data_df):
        """
        Initializes a sas spec
        """

        self.components = OrderedDict()
        is_only_component = len(spec) == 1
        for label, component_spec in spec.items():
            self.components[label] = Component(label, component_spec, data_df, is_only_component)
        self._componentorder = list(self.components.keys())
        self._comp2learn_componentorder = self._componentorder
        self._data_df = data_df

    def make_spec_ts(self, data_df=None):
        """
        Create functions that can be called to return the CDF and inverse CDF
        """
        if data_df is None:
            data_df = self._data_df

        # Trigger the method to create interpolators
        self.N = len(data_df)
        self._interp1d_inv = list(range(self.N))
        self._interp1d = list(range(self.N))
        self.ST_lists = list(range(self.N))
        self.P_lists = list(range(self.N))
        self.ST = None
        self.P = None
        for i in range(self.N):
            self.ST_lists[i] = np.sort(
                np.unique(np.concatenate([component.sas_fun[i].ST for label, component in self.components.items()]))
            )
            self.P_lists[i] = np.zeros_like(self.ST_lists[i])
            for label, component in self.components.items():
                self.P_lists[i] += component.sas_fun[i](self.ST_lists[i]) * component.weights.values[i]
            ST_min = self.ST_lists[i][0]
            ST_max = self.ST_lists[i][-1]
            self._interp1d_inv[i] = interp1d(
                self.P_lists[i],
                self.ST_lists[i],
                fill_value=(ST_min, ST_max),
                kind="linear",
                copy=False,
                bounds_error=False,
                assume_sorted=True,
            )
            self._interp1d[i] = interp1d(
                self.ST_lists[i],
                self.P_lists[i],
                fill_value=(0.0, 1.0),
                kind="linear",
                copy=False,
                bounds_error=False,
                assume_sorted=True,
            )
        # create matricies of ST, P padded with extra values on the left
        maxj = np.max([len(ST) for ST in self.ST_lists])
        self.ST = np.zeros((maxj, self.N))
        self.P = np.zeros_like(self.ST)
        for i in range(self.N):
            Nj = len(self.ST_lists[i])
            self.ST[maxj - Nj : maxj, i] = self.ST_lists[i]
            self.P[maxj - Nj : maxj, i] = self.P_lists[i]
            if Nj < maxj:
                self.ST[: maxj - Nj, i] = -1 - np.arange(maxj - Nj)
                self.P[: maxj - Nj, i] = np.zeros(maxj - Nj)
        self.ST = self.ST.T
        self.P = self.P.T

    def __call__(self, ST, i):
        """
        The Cumulative Distribution Function of the SAS function at a timestep

        :param ST: Age-ranked Storage values (can be a number, list or array-like)
        :param i: index of the timestep
        :return: A numpy array of corresponding probabilities
        """
        return self._interp1d[i](ST)

    def inv(self, P, i):
        """
        The inverse Cumulative Distribution Function of the SAS function at a timestep

        :param P: Probabilities (can be a number, list or array-like)
        :param i: index of the timestep
        :return: A numpy array of corresponding ST values
        """
        return self._interp1d_inv[i](P)

    def __repr__(self):
        """produce a string representation of the spec"""
        repr = ""
        for label, component in self.components.items():
            repr += component.__repr__()
        return repr

    def get_parameter_list(self):
        """
        Returns a list of the parameters in each component sas function
        """
        return np.concatenate(
            [self.components[label].sas_fun.parameter_list for label in self._comp2learn_componentorder]
        )

    def update_from_parameter_list(self, parameter_list):
        """Modify the component sas functions using a modified parameter_list"""
        starti = 0
        for label in self._comp2learn_componentorder:
            component = self.components[label]
            nparams = len(component.sas_fun.parameter_list)
            component.sas_fun.parameter_list = parameter_list[starti : starti + nparams]
            starti += nparams
        self.make_spec_ts()

    def plot(self, ax=None, **kwargs):
        """Plot the component sas_functions"""
        if ax is None:
            fig, ax = plt.subplots()
        componentlines = dict(
            [(label, component.plot(ax=ax, **kwargs)) for label, component in self.components.items()]
        )
        ax.set_xlabel("$S_T$")
        ax.set_ylabel(r"$\Omega(S_T)$")
        ax.legend(frameon=False)
        return componentlines

    def get_jacobian(self, *args, index=None, **kwargs):
        """
        Get the jacobian of each component sas function

        Calls the get_jacobian method of each component sas function.
        Components are concatenated columnwise and returned as a single numpy array.
        """
        if index is None:
            index = np.arange(self.N)
        cat_me = [
            self.components[label].sas_fun.get_jacobian(*args, index=index, **kwargs).T
            * self.components[label].weights.values[index]
            for label in self._comp2learn_componentorder
        ]
        return np.concatenate(cat_me, axis=0).T


class Component:
    """
    The Component class defines simple objects that package together sas
    functions with their weights timeseries and a label. The class also defines __repr__ and plot functions
    that call the corresponding functions of the underlying sas functions.
    """

    def __init__(self, label, spec, data_df, is_only_component):
        self.label = label
        self._spec = spec.copy()
        self._data_df = data_df
        self.N = len(data_df)
        self.weights = pd.Series(index=data_df.index, data=1.0) if is_only_component else data_df[label]
        ST = self.expand(spec.pop("ST")) if "ST" in spec else _NoneList()
        P = self.expand(spec.pop("P")) if "P" in spec else _NoneList()
        if "args" in spec:
            assert isinstance(spec["args"], dict)
            argdict = OrderedDict()
            for arg, value in spec.pop("args").items():
                argdict[arg] = self.expand(value)
            self._args = [dict(zip(argdict, values)) for values in zip(*argdict.values())]
            if "scipy.stats" in spec:
                use = "scipy.stats"
                func = spec.pop("scipy.stats")
            else:
                use = spec.pop("use", "builtin")
                func = spec.pop("func")
            self._sas_funs = [
                Continuous(use, func, func_kwargs=self._args[i], P=P[i, :], **spec) for i in range(self.N)
            ]
            if use == "scipy.stats":
                self.type = -1
            else:
                assert use == "builtin"
                self.type = self._sas_funs[0]._builtinfunctype
        elif ST is not None:
            self.type = -1
            self._sas_funs = [Piecewise(ST=ST[i, :], P=P[i, :], **spec) for i in range(self.N)]
        else:
            raise Exception(f"Insufficient information to specify SAS function for {label}")

    def expand(self, value):
        if isinstance(value, Iterable) and not isinstance(value, str):
            return np.concatenate([[self.expand(x)] for x in value], axis=0).T
        else:
            if isinstance(value, str):
                return self._data_df[value].values
            else:
                return float(value) * np.ones(self.N)

    @property
    def ST(self):
        return np.concatenate([[sas_fun.ST] for sas_fun in self._sas_funs]).T

    @property
    def P(self):
        return np.concatenate([[sas_fun.P] for sas_fun in self._sas_funs]).T

    @property
    def argsS(self):
        return np.concatenate([[sas_fun.argsS] for sas_fun in self._sas_funs]).T

    @property
    def argsP(self):
        return np.concatenate([[sas_fun.argsP] for sas_fun in self._sas_funs]).T

    @property
    def sas_fun(self):
        return [sas_fun for sas_fun in self._sas_funs]

    def __getitem__(self, i):
        return self._sas_funs[i]

    def __repr__(self):
        repr = ""
        repr += self._spec.__repr__()
        # repr += '    component = {}'.format(self.label) + '\n'
        # for i in range(self.N):
        # repr += self[i].__repr__()
        return repr

    def plot(self, *args, **kwargs):
        return self.sas_fun.plot(*args, label=self.label, **kwargs)


class _NoneList:
    def __init__(self):
        pass

    def __getitem__(self, *args, **kwargs):
        return None
