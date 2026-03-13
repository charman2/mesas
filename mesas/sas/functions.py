"""

================
Module functions
================

This module defines classes representing SAS functions.

Currently there are two classes : :class:`Piecewise` and :class:`Continuous`. The first allows the SAS function to be
specified as a set of breakpoints in a piecewise linear form of the cumulative distribution. The second allows any
distribution specified as a scipy.stats ``rv_continuous`` class to be used as a SAS function. However, it will be
converted into a piecewise form when the model is run.

Calling an instance of either class supplied with values of ST (as an array, list, or number) evaluates
the CDF, and returns corresponding cumulative probabilities. The :func:`Piecewise.inv` method takes cumulative probabilities and
returns values of ST.

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.interpolate import interp1d
from scipy.stats import rv_continuous


class _SASFunctionBase:
    """
    Base function for SAS functions
    """

    _use = None

    def __init__(self):
        pass

    def _make_interpolators(self):
        """
        Create functions that can be called to return the CDF and inverse CDF

        Uses scipy.optimize.interp1d
        """
        self.interp1d_inv = interp1d(
            self.P,
            self.ST,
            fill_value=(self.ST_min, self.ST_max),
            kind="linear",
            copy=False,
            bounds_error=False,
            assume_sorted=True,
        )
        self.interp1d = interp1d(
            self.ST, self.P, fill_value=(0.0, 1.0), kind="linear", copy=False, bounds_error=False, assume_sorted=True
        )

    def __getitem__(self, i):
        return self

    def __setitem__(self, i):
        raise TypeError("You're not allowed to modify the SAS function for a specific index")

    # Next we define a number of properties. These act like attributes, but special functions
    # are called when the attributes are queried or assigned to

    # @property
    # def has_params(self):
    #    return self._has_params

    @property
    def argsS(self):
        return None

    @argsS.setter
    def argsS(self, new_args):
        raise NotImplementedError("Method for setting args directly has not been defined")

    @property
    def argsP(self):
        return None

    @argsP.setter
    def argsP(self, new_args):
        raise NotImplementedError("Method for setting args directly has not been defined")

    @property
    def ST(self):
        return self._ST

    @ST.setter
    def ST(self, new_ST):
        raise NotImplementedError("Method for setting ST directly has not been defined")

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, new_P):
        raise NotImplementedError("Method for setting P directly has not been defined")

    @property
    def parameter_list(self):
        return self._parameter_list

    @parameter_list.setter
    def parameter_list(self, new_parameter_list):
        raise NotImplementedError("Method for setting parameters directly has not been defined")

    def subdivided_copy(self, *args, **kwargs):
        raise TypeError("Cannot subdivide this type of function")

    def _convert_segment_list_to_ST(self, seglst):
        """converts a segment_list array into an array of ST endpoints"""
        return np.cumsum(seglst)

    def _convert_ST_to_segment_list(self, ST):
        """converts a segment_list array into an array of ST endpoints"""
        return np.r_[ST[0], np.diff(ST)]

    def inv(self, P):
        """
        The inverse Cumulative Distribution Function of the SAS function
        """
        raise NotImplementedError("Inverse CDF method not been defined")

    def __call__(self, ST):
        """
        The Cumulative Distribution Function of the SAS function
        """
        raise NotImplementedError("CDF method not been defined")

    def __repr__(self):
        """Return a repr of the SAS function"""
        raise NotImplementedError("__repr__ not been defined")

    def plot(self, ax=None, **kwargs):
        """Return a plot of the SAS function"""
        if ax is None:
            fig, ax = plt.subplots()
        return ax.plot(self.ST, self.P, **kwargs)

    def get_jacobian(self, dCdSj, index=None, mode="segment", logtransform=True):
        """Calculates a limited jacobian of the model predictions"""
        raise NotImplementedError("get_jacobian not been defined")


class Piecewise(_SASFunctionBase):
    """
    Class for piecewise-linear SAS functions

    You can define a piecewise sas function in several different ways. The default is a single segment from 0 to 1:

        >>>sas_fun = Piecewise()
        >>>sas_fun([0.1, 0.3, 0.9])
        array([0.1, 0.3, 0.9])

    The range of the single segment can be modified by providing ``ST_min`` and ``ST_max`` keywords

        >>>sas_fun = Piecewise(ST_min=100, ST_max=110)
        >>>sas_fun([99, 101, 103, 109, 111])
        array([0. , 0.1, 0.3, 0.9, 1. ])

    The (ST,P) pairs defining the CDF are given by the ``ST`` and ``P`` attributes

        >>>sas_fun.ST
        array([100.        , 103.00391164, 104.17022005,
        110.        ])
        >>>sas_fun.P
        array([0.        , 0.33333333, 0.66666667, 1.        ])

    The segments defining the function are stored in the ``segment_list``
    attribute as a numpy array:

        >>>seglst = sas_fun.segment_list
        array([100.        ,   3.00391164,   1.1663084 ,   5.82977995])

    The first item in the array is ``ST_min``, and each following value is the interval
    along the ST axis to the end of each segment.

    Once created, the values of ``segment_list`` or ``ST`` can be modified and the other will update too.
    Values of ``P`` can also be changed.

    """

    def __init__(self, ST=None, P=None, nsegment=1, ST_max=1.0, ST_min=0.0, auto="uniform"):
        """
        Initializes a Piecewise sas function.

        :param segment_list: Optional list of segment lengths in ST. First element is assumed to be ST_min.
        :param ST: Optional list of ST vales at the end of segments. Ignored if `segment_list` is passed
        :param P: Optional list of probabilities at the end of segments. Defaults to evenly spaced intervals
        :param nsegment: If neither `segment_list` nor `ST` are provided, the range between ST_min and ST_max is split into this many randomly, recursively split intervals.
        :param ST_max: The lower bound of the sas function support
        :param ST_min: The upper bound of the sas function support
        """

        self.ST_min = float(ST_min)
        self.ST_max = float(ST_max)
        # self._has_params = False

        # Note that the variables _ST, _P and _parameter_list are assigned to below
        # instead of their corresponding properties. This is to avoid triggering the _make_interpolators function
        # until the end

        if ST is not None:
            # Use the given ST values
            assert ST[0] >= 0
            # assert np.all(np.diff(ST) > 0)
            assert len(ST) > 1
            self.nsegment = len(ST) - 1
            self._ST = np.array(ST, dtype=float)
            self._parameter_list = self._convert_ST_to_segment_list(self._ST)

        else:
            self.nsegment = nsegment

            if auto == "uniform":
                self._ST = np.linspace(self.ST_min, self.ST_max, self.nsegment + 1)
                self._parameter_list = self._convert_ST_to_segment_list(self._ST)
            elif auto == "random":
                # Generate a random function
                # Note this should probably be encapsulated in a function
                self._ST = np.zeros(self.nsegment + 1)
                ST_scaled = np.zeros(self.nsegment + 1)
                ST_scaled[-1] = 1.0
                for i in range(self.nsegment - 1):
                    ST_scaled[self.nsegment - i - 1] = np.random.uniform(0, ST_scaled[self.nsegment - i], 1)

                self._ST[:] = self.ST_min + (self.ST_max - self.ST_min) * ST_scaled
                self._parameter_list = self._convert_ST_to_segment_list(self._ST)

        # Make a list of probabilities P
        if P is not None:
            # Use the supplied values
            assert P[0] == 0
            assert P[-1] == 1
            assert np.all(np.diff(P) >= 0)
            assert len(P) == len(self._ST)
            self._P = np.r_[P]

        else:
            # Make P equally-spaced
            self._P = np.linspace(0, 1, self.nsegment + 1, endpoint=True)

        # self._has_params = True
        # call this to make the interpolation functions
        self._make_interpolators()

    @property
    def argsS(self):
        return self._ST

    @property
    def argsP(self):
        return self._P

    @_SASFunctionBase.ST.setter
    def ST(self, new_ST):
        try:
            assert new_ST[0] >= 0
            assert np.all(np.diff(new_ST) > 0)
            assert len(new_ST) > 1
        except Exception as ex:
            print("Problem with new ST")
            print(f"Attempting to set ST = {new_ST}")
            if not np.all(np.diff(new_ST) > 0):
                print(
                    "   -- if ST values are not distinct, try changing 'ST_largest_segment' and/or 'ST_smallest_segment'"
                )
            raise ex
        self._ST = new_ST
        self.ST_min = self._ST[0]
        self.ST_max = self._ST[-1]
        self._parameter_list = self._convert_ST_to_segment_list(self._ST)
        self._make_interpolators()

    @_SASFunctionBase.P.setter
    def P(self, new_P):
        assert new_P[0] == 0
        assert new_P[-1] == 1
        assert len(new_P) == len(self._ST)
        self._P = new_P
        self._make_interpolators()

    @_SASFunctionBase.parameter_list.setter
    def parameter_list(self, new_parameter_list):
        self._parameter_list = new_parameter_list
        self.ST = self._convert_segment_list_to_ST(self._parameter_list)

    def subdivided_copy(self, segment, split_frac=0.5):
        """Returns a new Piecewise object instance with one segment divided into two

        :param segment: The segment to be split
        :param split_frac: (Optional) The fraction of the segment where the split is made. Default is 0.5
        """
        assert segment < self.nsegment
        P1 = self.P[segment]
        P2 = self.P[segment + 1]
        ST1 = self.ST[segment]
        ST2 = self.ST[segment + 1]
        P_new = P1 + split_frac * (P2 - P1)
        ST_new = ST1 + split_frac * (ST2 - ST1)
        P = np.insert(self.P, segment + 1, P_new)
        ST = np.insert(self.ST, segment + 1, ST_new)
        return Piecewise(ST=ST, P=P)

    def inv(self, P):
        """
        The inverse Cumulative Distribution Function of the SAS function

        :param P: Probabilities (can be a number, list or array-like)
        :return: A numpy array of corresponding ST values
        """
        P_arr = np.array(P)
        P_ravel = P_arr.ravel()
        return self.interp1d_inv(P_ravel).reshape(P_arr.shape)

    def __call__(self, ST):
        """
        The Cumulative Distribution Function of the SAS function

        :param ST: Age-ranked Storage values (can be a number, list or array-like)
        :return: A numpy array of corresponding probabilities
        """
        ST_arr = np.array(ST)
        ST_ravel = ST_arr.ravel()
        return self.interp1d(ST_ravel).reshape(ST_arr.shape)

    def __repr__(self):
        """Return a repr of the SAS function"""
        if not hasattr(self, "nsegment"):
            return "Not initialized"
        repr = "        ST: "
        for i in range(self.nsegment + 1):
            repr += "{ST:<10.4}  ".format(ST=self.ST[i])
        repr += "\n"
        repr += "        P : "
        for i in range(self.nsegment + 1):
            repr += "{P:<10.4}  ".format(P=self.P[i])
        repr += "\n"
        return repr

    def get_jacobian(self, dCdSj, index=None, mode="segment", logtransform=True):
        """
        Calculates a limited jacobian of the model predictions

        If :math:`\vec{y} = f(\vec{x})` the Jacobian is a matrix where each i, j entry is

        .. math:

            J_{i,j}=\frac{\\partial y_i}{\\partial x_j}

        ..

        Here, the :math:`y_i` values are the predicted concentrations at each timestep,
        and the :math:`x_j` are either the lengths of each piecewise segment along with ST_min (default),
        or the values of ST at the endpoints of each segment (with `mode='endpoint'`).

        The Jacobian is limited as it represents the sensitivity of the model predictions of output
        concentrations at each timestep to variations in the sas function at that timestep, and neglects
        the cumulative effects that changing the sas function would have on the state variables (ST and MS).

        :param dCdSj: jacobian produced by solve.f90
        :param index: index of timesteps to calculate the jacobian for
        :param mode: If 'endpoint' return the derivatives with respect to the ST of the endpoints,
        otherwise if 'segment' return the derivatives wrt the segment lengths
        :param logtransform: If True, return the derivatives with respect to the log transform of the segment lengths.
         Ignored if `mode='endpoint'. Default is True.
        :return: numpy array of the Jacobian

        """

        if index is None:
            index = np.arange(dCdSj.shape[0])

        J_S = dCdSj[index, :]

        if mode == "endpoint":
            return J_S

        # To get the derivative with respect to the segment length, we add up the derivative w.r.t. the
        # endpoints that would be displaced by varying that segment
        A = np.triu(np.ones(self.nsegment + 1), k=0)
        J_seg = np.dot(A, J_S.T).T

        if logtransform:
            J_seglog = J_seg * self._parameter_list
            return J_seglog
        else:
            return J_seg


class kumaraswamy_gen(rv_continuous):
    "Kumaraswamy distribution"

    def _cdf(self, x, a, b):
        return 1 - (1 - x**a) ** b

    def _pdf(self, x, a, b):
        return a * b * x ** (a - 1) * (1 - x**a) ** (b - 1)


kumaraswamy = kumaraswamy_gen(a=0.0, b=1.0, name="kumaraswamy")


class Continuous(_SASFunctionBase):
    """
    Base function for SAS functions
    """

    _builtinfuncdict = {"gamma": 1, "beta": 2, "kumaraswamy": 3}

    def __init__(self, use, func, func_kwargs, P=None, nsegment=25, ST_max=1.797693134862315e308):
        if use == "builtin" and func in self._builtinfuncdict.keys():
            try:
                self._func = getattr(scipy.stats, func)
            except AttributeError:
                if func == "kumaraswamy":
                    self._func = kumaraswamy
                else:
                    self._func = None
            self._builtinfunctype = self._builtinfuncdict[func]
        elif use == "scipy.stats" and isinstance(func, rv_continuous):
            self._func = func
            self._builtinfunctype = None
        elif use == "scipy.stats" and isinstance(func, str):
            self._func = getattr(scipy.stats, func)
            self._builtinfunctype = None
        else:
            raise Exception("'use' keyword must be either 'builtin' or 'scipy.stats'")
        self._use = use

        self._frozen_func = self._func(**func_kwargs)
        if self._use == "builtin":
            if func == "gamma":
                self._argsS = []
                self._argsS += [func_kwargs["loc"], func_kwargs["scale"]]
                self._argsS += [func_kwargs["a"]]
            if func == "beta":
                self._argsS = []
                self._argsS += [func_kwargs["loc"], func_kwargs["scale"]]
                self._argsS += [func_kwargs["a"], func_kwargs["b"]]
            if func == "kumaraswamy":
                self._argsS = []
                self._argsS += [func_kwargs["loc"], func_kwargs["scale"]]
                self._argsS += [func_kwargs["a"], func_kwargs["b"]]
            self._argsP = np.ones_like(self._argsS) * np.NaN
        elif self._use == "scipy.stats":
            # generate a piecewise approximation
            self.ST_max = float(ST_max)
            # Make a list of probabilities P
            if P is not None:
                # Use the supplied values
                self.nsegment = len(P) - 1
                self.P = np.r_[P]
            else:
                # Make P equally-spaced
                self.nsegment = nsegment
                self.P = np.linspace(0, 1, self.nsegment + 1, endpoint=True)
            self._argsS = self._ST
            self._argsP = self._P

    @property
    def argsS(self):
        return self._argsS

    @property
    def argsP(self):
        return self._argsP

    @property
    def func(self):
        return self._frozen_func

    @func.setter
    def func(selfs, new_func):
        raise ValueError("Function type cannot be changed")

    @_SASFunctionBase.ST.setter
    def ST(self, new_ST):
        try:
            assert new_ST[0] >= 0
            assert np.all(np.diff(new_ST) > 0)
            assert len(new_ST) > 1
        except Exception as ex:
            print("Problem with new ST")
            print(f"Attempting to set ST = {new_ST}")
            raise ex
        if new_ST[-1] > self.ST_max:
            self._ST = new_ST
            self.ST_max = self._ST[-1]
        else:
            self._ST = np.r_[new_ST, self.ST_max]
        self._P = self.func.cdf(self._ST)

    @_SASFunctionBase.P.setter
    def P(self, new_P):
        assert new_P[0] == 0
        assert new_P[-1] == 1
        assert np.all(np.diff(new_P) >= 0)
        self._P = new_P
        # if self._has_params:
        self._ST = self.func.ppf(self._P)

    def inv(self, P):
        """
        The inverse Cumulative Distribution Function of the SAS function
        """
        return self.func.ppf(P)

    def __call__(self, ST):
        """
        The Cumulative Distribution Function of the SAS function
        """
        return self.func.cdf(ST)

    def __repr__(self):
        """Return a repr of the SAS function"""
        repr = f"       {self._func.name} distribution\n"
        # if self._has_params:
        repr += f"        parameters: {self._frozen_func.args}\n"
        repr += "        Lookup table version:\n"
        repr += "          ST: "
        for i in range(self.nsegment + 1):
            repr += "{ST:<10.4}  ".format(ST=self.ST[i])
        repr += "\n"
        repr += "          P : "
        for i in range(self.nsegment + 1):
            repr += "{P:<10.4}  ".format(P=self.P[i])
        repr += "\n"
        # else:
        # repr = f'        (parameters not set)'
        return repr

    def get_jacobian(self, dCdSj, index=None, mode="segment", logtransform=True):
        """Calculates a limited jacobian of the model predictions"""
        raise NotImplementedError("get_jacobian not been defined")
