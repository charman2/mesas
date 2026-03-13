"""Pure-Python + Numba replacement for the Fortran ``solve.f90`` solver.

This module implements the SAS (StorAge Selection) transport solver using
the method of characteristics with Runge-Kutta time integration.  The
algorithm is numerically equivalent to the original Fortran implementation,
but written in Python and accelerated with Numba ``@njit`` for performance
parity.

The public entry point is :func:`solve`, which has the same signature and
return values as the Fortran ``solvesas`` subroutine (via f2py).
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:
    # Fallback: run without JIT compilation
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


# Set to False to disable JIT compilation during development/debugging
_USE_NUMBA = True


def _maybe_jit(func):
    """Apply @_maybe_jit if Numba is available and enabled."""
    if _USE_NUMBA:
        return njit(cache=True)(func)
    return func


# ---------------------------------------------------------------------------
# Special-function helpers (Numba-compatible, no scipy dependency)
# ---------------------------------------------------------------------------


@_maybe_jit
def _alngam(x):
    """Log-gamma function (Stirling-based, matches AS 245 in solve.f90)."""
    if x <= 0.0:
        return 0.0
    # Use Stirling's series for lgamma
    # For small x, use recurrence relation gamma(x+1) = x*gamma(x)
    result = 0.0
    y = x
    while y < 8.0:
        result -= math.log(y)
        y += 1.0
    # Stirling approx for y >= 8
    r = 1.0 / y
    r2 = r * r
    result += (y - 0.5) * math.log(y) - y + 0.9189385332046727  # 0.5*ln(2*pi)
    result += r * (1.0 / 12.0 - r2 * (1.0 / 360.0 - r2 * (1.0 / 1260.0 - r2 * (1.0 / 1680.0 - r2 / 1188.0))))
    return result


@_maybe_jit
def _gammad(x, p):
    """Incomplete gamma integral P(p, x) = gammainc(p, x).

    Faithful port of the AS 239 / solve.f90 ``gammad`` implementation.
    """
    tol = 1.0e-14
    elimit = -88.0
    oflo = 1.0e37

    if x < 0.0 or p <= 0.0:
        return 0.0
    if x == 0.0:
        return 0.0

    # Pearson's series expansion (x <= 1 or x < p)
    if x <= 1.0 or x < p:
        arg = p * math.log(x) - x - _alngam(p + 1.0)
        cc = 1.0
        result = 1.0
        a = p
        while True:
            a += 1.0
            cc = cc * x / a
            result += cc
            if cc <= tol:
                break
        arg = arg + math.log(result)
        if arg >= elimit:
            return math.exp(arg)
        return 0.0
    else:
        # Continued fraction expansion
        arg = p * math.log(x) - x - _alngam(p)
        a = 1.0 - p
        b = a + x + 1.0
        cc = 0.0
        pn1 = 1.0
        pn2 = x
        pn3 = x + 1.0
        pn4 = x * b
        result = pn3 / pn4
        while True:
            a += 1.0
            b += 2.0
            cc += 1.0
            an = a * cc
            pn5 = b * pn3 - an * pn1
            pn6 = b * pn4 - an * pn2
            if abs(pn6) > 0.0:
                rn = pn5 / pn6
                if abs(result - rn) <= min(tol, tol * rn):
                    arg = arg + math.log(result)
                    if arg >= elimit:
                        return 1.0 - math.exp(arg)
                    return 1.0
                result = rn
            pn1 = pn3
            pn2 = pn4
            pn3 = pn5
            pn4 = pn6
            # Rescale to prevent overflow
            if abs(pn5) >= oflo:
                pn1 /= oflo
                pn2 /= oflo
                pn3 /= oflo
                pn4 /= oflo
        # Should not reach here
        return 1.0 - math.exp(arg) * result


@_maybe_jit
def _betain(x, p, q):
    """Incomplete beta function I_x(p, q).

    Matches AS 63 / solve.f90 ``betain`` implementation.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use the continued fraction in the region where it converges faster
    psq = p + q
    if p < psq * x:
        xx = 1.0 - x
        cx = x
        pp = q
        qq = p
        indx = True
    else:
        xx = x
        cx = 1.0 - x
        pp = p
        qq = q
        indx = False

    beta = _alngam(p) + _alngam(q) - _alngam(p + q)
    term = 1.0
    ai = 1.0
    result = 1.0
    ns = int(qq + cx * psq)

    # Soper's reduction
    rx = xx / cx
    temp = qq - ai
    if ns == 0:
        rx = xx

    for _ in range(1000):
        term = term * temp * rx / (pp + ai)
        result += term
        temp = abs(term)
        if temp <= 1e-14 and temp <= 1e-14 * abs(result):
            result = result * math.exp(pp * math.log(xx) + (qq - 1.0) * math.log(cx) - beta) / pp
            if indx:
                result = 1.0 - result
            return result
        ai += 1.0
        ns -= 1
        if ns >= 0:
            temp = qq - ai
            if ns == 0:
                rx = xx
        else:
            temp = psq
            psq += 1.0

    # Should not reach here, but return best estimate
    result = result * math.exp(pp * math.log(xx) + (qq - 1.0) * math.log(cx) - beta) / pp
    if indx:
        result = 1.0 - result
    return result


# ---------------------------------------------------------------------------
# SAS function evaluators
# ---------------------------------------------------------------------------


@_maybe_jit
def _piecewise_linear_cdf(ST, SAS_args, P_list, grad_precalc, ai, jt_c, nargs):
    """Evaluate a piecewise-linear SAS function at cumulative storage ST.

    Uses full arrays + offset ``ai`` and column ``jt_c`` to avoid Numba
    issues with dynamically-sized array slices.
    """
    if ST <= SAS_args[ai, jt_c]:
        return P_list[ai, jt_c]
    for i in range(nargs):
        if ST < SAS_args[ai + i, jt_c]:
            ia = i - 1
            return P_list[ai + ia, jt_c] + (ST - SAS_args[ai + ia, jt_c]) * grad_precalc[ai + ia, jt_c]
    return P_list[ai + nargs - 1, jt_c]


@_maybe_jit
def _kumaraswamy_cdf(ST, loc, scale, a, b):
    """Kumaraswamy CDF: F(x) = 1 - (1 - x^a)^b."""
    x = (ST - loc) / scale
    x = min(max(0.0, x), 1.0)
    return 1.0 - (1.0 - x**a) ** b


@_maybe_jit
def _beta_cdf(ST, loc, scale, a, b):
    """Beta CDF via incomplete beta function."""
    x = (ST - loc) / scale
    x = min(max(0.0, x), 1.0)
    return _betain(x, a, b)


@_maybe_jit
def _gamma_cdf(ST, loc, scale, a):
    """Gamma CDF via incomplete gamma integral."""
    if ST <= loc:
        return 0.0
    x = (ST - loc) / scale
    return _gammad(x, a)


# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------


@_maybe_jit
def _precompute_gradients(
    SAS_args,
    P_list,
    component_type,
    component_index_list,
    args_index_list,
    numargs_list,
    numflux,
    timeseries_length,
    numargs_total,
):
    """Pre-compute slopes for piecewise-linear SAS function segments."""
    grad_precalc = np.zeros((numargs_total, timeseries_length))
    for iq in range(numflux):
        ic_start = int(component_index_list[iq])
        ic_end = int(component_index_list[iq + 1])
        for ic_ in range(ic_start, ic_end):
            if component_type[ic_] == -1:
                na = int(numargs_list[ic_])
                ai = int(args_index_list[ic_])
                for ia in range(na - 1):
                    ii = ai + ia
                    for jt in range(timeseries_length):
                        s1 = float(SAS_args[ii + 1, jt])
                        s0 = float(SAS_args[ii, jt])
                        dS = s1 - s0
                        if dS != 0.0:
                            p1 = float(P_list[ii + 1, jt])
                            p0 = float(P_list[ii, jt])
                            grad_precalc[ii, jt] = (p1 - p0) / dS
    return grad_precalc


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


@_maybe_jit
def _solve_core(
    J_fullstep,  # (timeseries_length,)
    Q_fullstep,  # (timeseries_length, numflux)
    SAS_args,  # (numargs_total, timeseries_length) -- note: transposed
    P_list,  # (numargs_total, timeseries_length) -- note: transposed
    weights_fullstep,  # (timeseries_length, numcomponent_total)
    sT_init_fullstep,  # (max_age,)
    dt,  # scalar
    verbose,  # bool
    mT_init_fullstep,  # (max_age, numsol)
    C_J_fullstep,  # (timeseries_length, numsol)
    alpha_fullstep,  # (timeseries_length, numflux, numsol)
    k1_fullstep,  # (timeseries_length, numsol)
    C_eq_fullstep,  # (timeseries_length, numsol)
    C_old,  # (numsol,)
    n_substeps,  # int
    component_type,  # (numcomponent_total,)
    numcomponent_list,  # (numflux,)
    numargs_list,  # (numcomponent_total,)
    output_these_fullsteps,  # (num_output_fullsteps,)
    num_scheme,  # int
    numflux,  # int
    numsol,  # int
    max_age,  # int
    timeseries_length,  # int
    num_output_fullsteps,  # int
    numcomponent_total,  # int
    numargs_total,  # int
):
    """Core SAS solver — Numba-compiled equivalent of solveSAS in solve.f90."""

    total_num_substeps = timeseries_length * n_substeps
    dt_substep = dt / n_substeps
    norm = 1.0 / n_substeps / n_substeps

    # RK coefficients
    rk4_stepfraction = np.array([0.0, 0.5, 0.5, 1.0, 1.0])
    rk4_coeff = np.array([1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0])
    rk2_stepfraction = np.array([0.0, 1.0, 1.0])
    rk2_coeff = np.array([0.5, 0.5])

    # Pre-compute index mappings
    args_index_list = np.zeros(numcomponent_total + 1, dtype=np.int64)
    component_index_list = np.zeros(numflux + 1, dtype=np.int64)
    for iq in range(numflux):
        component_index_list[iq + 1] = component_index_list[iq] + numcomponent_list[iq]
        for ic in range(component_index_list[iq], component_index_list[iq + 1]):
            args_index_list[ic + 1] = args_index_list[ic] + numargs_list[ic]

    # Pre-compute gradients for piecewise-linear SAS functions
    grad_precalc = _precompute_gradients(
        SAS_args,
        P_list,
        component_type,
        component_index_list,
        args_index_list,
        numargs_list,
        numflux,
        timeseries_length,
        numargs_total,
    )

    # Allocate output arrays
    C_Q_fullstep = np.zeros((timeseries_length, numflux, numsol))
    sT_outputstep = np.zeros((num_output_fullsteps + 1, max_age))
    mT_outputstep = np.zeros((num_output_fullsteps + 1, numsol, max_age))
    pQ_outputstep = np.zeros((num_output_fullsteps, numflux, max_age))
    mQ_outputstep = np.zeros((num_output_fullsteps, numflux, numsol, max_age))
    mR_outputstep = np.zeros((num_output_fullsteps, numsol, max_age))
    WaterBalance_outputstep = np.zeros((num_output_fullsteps, max_age))
    SoluteBalance_outputstep = np.zeros((num_output_fullsteps, numsol, max_age))
    # Jacobian outputs (placeholders — not yet implemented in Numba)
    ds_outputstep = np.zeros((num_output_fullsteps + 1, numargs_total, max_age))
    dm_outputstep = np.zeros((num_output_fullsteps + 1, numargs_total, numsol, max_age))
    dC_fullstep = np.zeros((timeseries_length, numargs_total, numflux, numsol))

    P_old_fullstep = np.ones((timeseries_length, numflux))

    # Working arrays
    STcum_topbot_start = np.zeros((total_num_substeps + 1, 2))
    pQ_temp = np.zeros((total_num_substeps, numflux))
    pQ_aver = np.zeros((total_num_substeps, numflux))
    mQ_temp = np.zeros((total_num_substeps, numflux, numsol))
    mQ_aver = np.zeros((total_num_substeps, numflux, numsol))
    mR_temp = np.zeros((total_num_substeps, numsol))
    mR_aver = np.zeros((total_num_substeps, numsol))
    sT_start = np.zeros(total_num_substeps)
    sT_temp = np.zeros(total_num_substeps)
    mT_start = np.zeros((total_num_substeps, numsol))
    mT_temp = np.zeros((total_num_substeps, numsol))
    jt_fullstep_at = np.zeros(total_num_substeps, dtype=np.int64)
    jt_substep_at = np.zeros(total_num_substeps, dtype=np.int64)

    # Initial conditions
    sT_outputstep[0, :] = sT_init_fullstep
    # mT_init is (max_age, numsol) but mT_outputstep is (output_steps+1, numsol, max_age)
    for s in range(numsol):
        for t in range(max_age):
            mT_outputstep[0, s, t] = mT_init_fullstep[t, s]

    iT_prev_fullstep = -1

    # =========================================================================
    # MAIN LOOP: iterate over water age (outer) and substeps (inner)
    # =========================================================================
    for iT_fullstep in range(max_age):
        for substep in range(n_substeps):
            iT_substep = iT_fullstep * n_substeps + substep

            # Map each characteristic to its current full timestep
            for c in range(total_num_substeps):
                jt_sub = (c + iT_substep) % total_num_substeps
                jt_substep_at[c] = jt_sub
                jt_is_which_substep = jt_sub % n_substeps
                jt_fullstep_at[c] = (jt_sub - jt_is_which_substep) // n_substeps

            # Reset RK averages
            for c in range(total_num_substeps):
                for iq in range(numflux):
                    pQ_aver[c, iq] = 0.0
                    for s in range(numsol):
                        mQ_aver[c, iq, s] = 0.0
                for s in range(numsol):
                    mR_aver[c, s] = 0.0

            # Initial condition for newest characteristic
            if iT_substep > 0:
                idx = total_num_substeps - iT_substep
                sT_start[idx] = sT_init_fullstep[iT_prev_fullstep]
                for s in range(numsol):
                    mT_start[idx, s] = mT_init_fullstep[iT_prev_fullstep, s]

            # Copy start to temp
            for c in range(total_num_substeps):
                sT_temp[c] = sT_start[c]
                for s in range(numsol):
                    mT_temp[c, s] = mT_start[c, s]

            # ---- Runge-Kutta integration ----
            if num_scheme == 1:
                # Forward Euler
                _get_flux(
                    sT_temp,
                    mT_temp,
                    pQ_temp,
                    mQ_temp,
                    mR_temp,
                    0.0,
                    iT_substep,
                    total_num_substeps,
                    n_substeps,
                    dt_substep,
                    jt_fullstep_at,
                    jt_substep_at,
                    STcum_topbot_start,
                    sT_init_fullstep,
                    Q_fullstep,
                    alpha_fullstep,
                    k1_fullstep,
                    C_eq_fullstep,
                    weights_fullstep,
                    SAS_args,
                    P_list,
                    grad_precalc,
                    component_type,
                    component_index_list,
                    args_index_list,
                    numargs_list,
                    numflux,
                    numsol,
                    numcomponent_total,
                    timeseries_length,
                )
                _add_to_average(
                    pQ_aver, mQ_aver, mR_aver, pQ_temp, mQ_temp, mR_temp, 1.0, total_num_substeps, numflux, numsol
                )
                _new_state(
                    sT_temp,
                    mT_temp,
                    sT_start,
                    mT_start,
                    pQ_aver,
                    mQ_aver,
                    mR_aver,
                    1.0,
                    iT_substep,
                    dt_substep,
                    jt_fullstep_at,
                    J_fullstep,
                    C_J_fullstep,
                    Q_fullstep,
                    total_num_substeps,
                    numflux,
                    numsol,
                )

            elif num_scheme == 2:
                # RK2
                for rk in range(2):
                    sf = rk2_stepfraction[rk]
                    _get_flux(
                        sT_temp,
                        mT_temp,
                        pQ_temp,
                        mQ_temp,
                        mR_temp,
                        sf,
                        iT_substep,
                        total_num_substeps,
                        n_substeps,
                        dt_substep,
                        jt_fullstep_at,
                        jt_substep_at,
                        STcum_topbot_start,
                        sT_init_fullstep,
                        Q_fullstep,
                        alpha_fullstep,
                        k1_fullstep,
                        C_eq_fullstep,
                        weights_fullstep,
                        SAS_args,
                        P_list,
                        grad_precalc,
                        component_type,
                        component_index_list,
                        args_index_list,
                        numargs_list,
                        numflux,
                        numsol,
                        numcomponent_total,
                        timeseries_length,
                    )
                    _add_to_average(
                        pQ_aver,
                        mQ_aver,
                        mR_aver,
                        pQ_temp,
                        mQ_temp,
                        mR_temp,
                        rk2_coeff[rk],
                        total_num_substeps,
                        numflux,
                        numsol,
                    )
                    if rk < 1:
                        sf_next = rk2_stepfraction[rk + 1]
                        _new_state(
                            sT_temp,
                            mT_temp,
                            sT_start,
                            mT_start,
                            pQ_temp,
                            mQ_temp,
                            mR_temp,
                            sf_next,
                            iT_substep,
                            dt_substep,
                            jt_fullstep_at,
                            J_fullstep,
                            C_J_fullstep,
                            Q_fullstep,
                            total_num_substeps,
                            numflux,
                            numsol,
                        )
                # Final state update
                _new_state(
                    sT_temp,
                    mT_temp,
                    sT_start,
                    mT_start,
                    pQ_aver,
                    mQ_aver,
                    mR_aver,
                    rk2_stepfraction[2],
                    iT_substep,
                    dt_substep,
                    jt_fullstep_at,
                    J_fullstep,
                    C_J_fullstep,
                    Q_fullstep,
                    total_num_substeps,
                    numflux,
                    numsol,
                )

            elif num_scheme == 4:
                # RK4
                for rk in range(4):
                    sf = rk4_stepfraction[rk]
                    _get_flux(
                        sT_temp,
                        mT_temp,
                        pQ_temp,
                        mQ_temp,
                        mR_temp,
                        sf,
                        iT_substep,
                        total_num_substeps,
                        n_substeps,
                        dt_substep,
                        jt_fullstep_at,
                        jt_substep_at,
                        STcum_topbot_start,
                        sT_init_fullstep,
                        Q_fullstep,
                        alpha_fullstep,
                        k1_fullstep,
                        C_eq_fullstep,
                        weights_fullstep,
                        SAS_args,
                        P_list,
                        grad_precalc,
                        component_type,
                        component_index_list,
                        args_index_list,
                        numargs_list,
                        numflux,
                        numsol,
                        numcomponent_total,
                        timeseries_length,
                    )
                    _add_to_average(
                        pQ_aver,
                        mQ_aver,
                        mR_aver,
                        pQ_temp,
                        mQ_temp,
                        mR_temp,
                        rk4_coeff[rk],
                        total_num_substeps,
                        numflux,
                        numsol,
                    )
                    if rk < 3:
                        sf_next = rk4_stepfraction[rk + 1]
                        _new_state(
                            sT_temp,
                            mT_temp,
                            sT_start,
                            mT_start,
                            pQ_temp,
                            mQ_temp,
                            mR_temp,
                            sf_next,
                            iT_substep,
                            dt_substep,
                            jt_fullstep_at,
                            J_fullstep,
                            C_J_fullstep,
                            Q_fullstep,
                            total_num_substeps,
                            numflux,
                            numsol,
                        )
                # Final state update
                _new_state(
                    sT_temp,
                    mT_temp,
                    sT_start,
                    mT_start,
                    pQ_aver,
                    mQ_aver,
                    mR_aver,
                    rk4_stepfraction[4],
                    iT_substep,
                    dt_substep,
                    jt_fullstep_at,
                    J_fullstep,
                    C_J_fullstep,
                    Q_fullstep,
                    total_num_substeps,
                    numflux,
                    numsol,
                )

            # Commit RK result
            for c in range(total_num_substeps):
                sT_start[c] = sT_temp[c]
                for s in range(numsol):
                    mT_start[c, s] = mT_temp[c, s]

            # Update cumulative storage along characteristics
            for c in range(total_num_substeps + 1):
                STcum_topbot_start[c, 0] = STcum_topbot_start[c, 1]
            for c in range(total_num_substeps):
                jt_c = jt_substep_at[c]
                if jt_c < total_num_substeps:
                    STcum_topbot_start[jt_c + 1, 1] = STcum_topbot_start[jt_c + 1, 0] + sT_start[c] * dt_substep
            STcum_topbot_start[0, 1] = STcum_topbot_start[0, 0] + sT_init_fullstep[iT_fullstep] * dt_substep

            # Update records
            _update_records(
                iT_fullstep,
                iT_substep,
                substep,
                total_num_substeps,
                n_substeps,
                dt_substep,
                dt,
                norm,
                jt_fullstep_at,
                jt_substep_at,
                Q_fullstep,
                alpha_fullstep,
                pQ_aver,
                mQ_aver,
                mR_aver,
                sT_start,
                mT_start,
                C_Q_fullstep,
                P_old_fullstep,
                pQ_outputstep,
                mQ_outputstep,
                mR_outputstep,
                sT_outputstep,
                mT_outputstep,
                output_these_fullsteps,
                num_output_fullsteps,
                numflux,
                numsol,
                max_age,
                timeseries_length,
            )

            iT_prev_fullstep = iT_fullstep

        if verbose and iT_fullstep % 10 == 0:
            print(" ...Done", iT_fullstep, "of", max_age)

    # Finalization: add contribution from water older than max_age
    for s in range(numsol):
        for iq in range(numflux):
            for jt in range(timeseries_length):
                if Q_fullstep[jt, iq] > 0:
                    C_Q_fullstep[jt, iq, s] += alpha_fullstep[jt, iq, s] * C_old[s] * P_old_fullstep[jt, iq]

    # Calculate mass balances
    _calculate_balances(
        iT_fullstep,
        max_age,
        num_output_fullsteps,
        output_these_fullsteps,
        J_fullstep,
        Q_fullstep,
        C_J_fullstep,
        sT_outputstep,
        mT_outputstep,
        pQ_outputstep,
        mQ_outputstep,
        mR_outputstep,
        WaterBalance_outputstep,
        SoluteBalance_outputstep,
        dt,
        numflux,
        numsol,
    )

    if verbose:
        print(" ...Finished...")

    return (
        sT_outputstep,
        pQ_outputstep,
        WaterBalance_outputstep,
        mT_outputstep,
        mQ_outputstep,
        mR_outputstep,
        C_Q_fullstep,
        ds_outputstep,
        dm_outputstep,
        dC_fullstep,
        SoluteBalance_outputstep,
    )


@_maybe_jit
def _get_flux(
    sT,
    mT,
    pQ,
    mQ,
    mR,
    stepfraction,
    iT_substep,
    total_num_substeps,
    n_substeps,
    dt_substep,
    jt_fullstep_at,
    jt_substep_at,
    STcum_topbot_start,
    sT_init_fullstep,
    Q_fullstep,
    alpha_fullstep,
    k1_fullstep,
    C_eq_fullstep,
    weights_fullstep,
    SAS_args,
    P_list,
    grad_precalc,
    component_type,
    component_index_list,
    args_index_list,
    numargs_list,
    numflux,
    numsol,
    numcomponent_total,
    timeseries_length,
):
    """Compute fluxes from current state using SAS functions."""

    # Calculate pQ (transit time distribution)
    _calculate_pQ(
        sT,
        pQ,
        stepfraction,
        iT_substep,
        total_num_substeps,
        n_substeps,
        dt_substep,
        jt_fullstep_at,
        jt_substep_at,
        STcum_topbot_start,
        sT_init_fullstep,
        weights_fullstep,
        SAS_args,
        P_list,
        grad_precalc,
        component_type,
        component_index_list,
        args_index_list,
        numargs_list,
        numflux,
        numcomponent_total,
        timeseries_length,
    )

    # Solute mass flux: mQ = mT * alpha * Q * pQ / sT (well-mixed)
    for c in range(total_num_substeps):
        for iq in range(numflux):
            for s in range(numsol):
                mQ[c, iq, s] = 0.0
    for s in range(numsol):
        for iq in range(numflux):
            for c in range(total_num_substeps):
                if sT[c] > 0.0:
                    jt_c = jt_fullstep_at[c]
                    mQ[c, iq, s] = mT[c, s] * alpha_fullstep[jt_c, iq, s] * Q_fullstep[jt_c, iq] * pQ[c, iq] / sT[c]

    # Reaction mass: mR = k1 * (C_eq * sT - mT)
    for s in range(numsol):
        for c in range(total_num_substeps):
            jt_c = jt_fullstep_at[c]
            if k1_fullstep[jt_c, s] > 0.0:
                mR[c, s] = k1_fullstep[jt_c, s] * (C_eq_fullstep[jt_c, s] * sT[c] - mT[c, s])
            else:
                mR[c, s] = 0.0


@_maybe_jit
def _calculate_pQ(
    sT,
    pQ,
    stepfraction,
    iT_substep,
    total_num_substeps,
    n_substeps,
    dt_substep,
    jt_fullstep_at,
    jt_substep_at,
    STcum_topbot_start,
    sT_init_fullstep,
    weights_fullstep,
    SAS_args,
    P_list,
    grad_precalc,
    component_type,
    component_index_list,
    args_index_list,
    numargs_list,
    numflux,
    numcomponent_total,
    timeseries_length,
):
    """Evaluate SAS functions to compute pQ (transit time distribution)."""

    if iT_substep == 0 and stepfraction == 0.0:
        for c in range(total_num_substeps):
            for iq in range(numflux):
                pQ[c, iq] = 0.0
        return

    # Compute STcum at top and bottom of each age bin
    STcum_top = np.zeros(total_num_substeps)
    STcum_bot = np.zeros(total_num_substeps)

    if iT_substep == 0:
        # All tops are zero
        pass
    else:
        for c in range(total_num_substeps):
            jt_sub = jt_substep_at[c]
            STcum_top[c] = (
                STcum_topbot_start[jt_sub, 0] * (1.0 - stepfraction) + STcum_topbot_start[jt_sub + 1, 1] * stepfraction
            )

    for c in range(total_num_substeps):
        STcum_bot[c] = STcum_top[c] + sT[c] * dt_substep

    # Evaluate SAS functions at top and bottom
    PQcum_top = np.zeros((total_num_substeps, numflux))
    PQcum_bot = np.zeros((total_num_substeps, numflux))

    for iq in range(numflux):
        for ic in range(int(component_index_list[iq]), int(component_index_list[iq + 1])):
            ctype = int(component_type[ic])
            ai = int(args_index_list[ic])
            nargs = int(numargs_list[ic])

            for c in range(total_num_substeps):
                if sT[c] <= 0.0 and ctype != -1:
                    continue
                jt_c = int(jt_fullstep_at[c])
                w = float(weights_fullstep[jt_c, ic])

                # Evaluate at top
                ST_val = STcum_top[c]
                if ctype == -1:
                    p_top = _piecewise_linear_cdf(ST_val, SAS_args, P_list, grad_precalc, ai, jt_c, nargs)
                elif ctype == 1:
                    p_top = _gamma_cdf(
                        ST_val, float(SAS_args[ai, jt_c]), float(SAS_args[ai + 1, jt_c]), float(SAS_args[ai + 2, jt_c])
                    )
                elif ctype == 2:
                    p_top = _beta_cdf(
                        ST_val,
                        float(SAS_args[ai, jt_c]),
                        float(SAS_args[ai + 1, jt_c]),
                        float(SAS_args[ai + 2, jt_c]),
                        float(SAS_args[ai + 3, jt_c]),
                    )
                elif ctype == 3:
                    p_top = _kumaraswamy_cdf(
                        ST_val,
                        float(SAS_args[ai, jt_c]),
                        float(SAS_args[ai + 1, jt_c]),
                        float(SAS_args[ai + 2, jt_c]),
                        float(SAS_args[ai + 3, jt_c]),
                    )
                else:
                    p_top = 0.0

                PQcum_top[c, iq] += w * p_top

                # Evaluate at bottom
                ST_val = STcum_bot[c]
                if ctype == -1:
                    p_bot = _piecewise_linear_cdf(ST_val, SAS_args, P_list, grad_precalc, ai, jt_c, nargs)
                elif ctype == 1:
                    p_bot = _gamma_cdf(
                        ST_val, float(SAS_args[ai, jt_c]), float(SAS_args[ai + 1, jt_c]), float(SAS_args[ai + 2, jt_c])
                    )
                elif ctype == 2:
                    p_bot = _beta_cdf(
                        ST_val,
                        float(SAS_args[ai, jt_c]),
                        float(SAS_args[ai + 1, jt_c]),
                        float(SAS_args[ai + 2, jt_c]),
                        float(SAS_args[ai + 3, jt_c]),
                    )
                elif ctype == 3:
                    p_bot = _kumaraswamy_cdf(
                        ST_val,
                        float(SAS_args[ai, jt_c]),
                        float(SAS_args[ai + 1, jt_c]),
                        float(SAS_args[ai + 2, jt_c]),
                        float(SAS_args[ai + 3, jt_c]),
                    )
                else:
                    p_bot = 0.0

                PQcum_bot[c, iq] += w * p_bot

    # pQ = (PQcum_bottom - PQcum_top) / dt_substep
    for c in range(total_num_substeps):
        for iq in range(numflux):
            if sT[c] == 0.0:
                pQ[c, iq] = 0.0
            else:
                pQ[c, iq] = (PQcum_bot[c, iq] - PQcum_top[c, iq]) / dt_substep


@_maybe_jit
def _add_to_average(pQ_aver, mQ_aver, mR_aver, pQ, mQ, mR, coeff, total_num_substeps, numflux, numsol):
    """Accumulate RK-weighted flux estimates into running average."""
    for c in range(total_num_substeps):
        for iq in range(numflux):
            pQ_aver[c, iq] += coeff * pQ[c, iq]
            for s in range(numsol):
                mQ_aver[c, iq, s] += coeff * mQ[c, iq, s]
        for s in range(numsol):
            mR_aver[c, s] += coeff * mR[c, s]


@_maybe_jit
def _new_state(
    sT,
    mT,
    sT_start,
    mT_start,
    pQ,
    mQ,
    mR,
    stepfraction,
    iT_substep,
    dt_substep,
    jt_fullstep_at,
    J_fullstep,
    C_J_fullstep,
    Q_fullstep,
    total_num_substeps,
    numflux,
    numsol,
):
    """Advance state by one RK sub-stage."""
    dt_num = dt_substep * stepfraction

    for c in range(total_num_substeps):
        sT[c] = sT_start[c]
        for s in range(numsol):
            mT[c, s] = mT_start[c, s] + mR[c, s] * dt_num

    # Influx at age 0
    if iT_substep == 0:
        for c in range(total_num_substeps):
            jt_c = jt_fullstep_at[c]
            sT[c] += J_fullstep[jt_c] * stepfraction
            for s in range(numsol):
                mT[c, s] += J_fullstep[jt_c] * C_J_fullstep[jt_c, s] * stepfraction

    # Outflux
    for c in range(total_num_substeps):
        jt_c = jt_fullstep_at[c]
        total_Q_pQ = 0.0
        for iq in range(numflux):
            total_Q_pQ += Q_fullstep[jt_c, iq] * pQ[c, iq]
        sT[c] -= total_Q_pQ * dt_num
        if sT[c] < 0.0:
            sT[c] = 0.0

    # Solute outflux
    for c in range(total_num_substeps):
        for s in range(numsol):
            total_mQ = 0.0
            for iq in range(numflux):
                total_mQ += mQ[c, iq, s]
            mT[c, s] -= total_mQ * dt_num


@_maybe_jit
def _update_records(
    iT_fullstep,
    iT_substep,
    substep,
    total_num_substeps,
    n_substeps,
    dt_substep,
    dt,
    norm,
    jt_fullstep_at,
    jt_substep_at,
    Q_fullstep,
    alpha_fullstep,
    pQ_aver,
    mQ_aver,
    mR_aver,
    sT_start,
    mT_start,
    C_Q_fullstep,
    P_old_fullstep,
    pQ_outputstep,
    mQ_outputstep,
    mR_outputstep,
    sT_outputstep,
    mT_outputstep,
    output_these_fullsteps,
    num_output_fullsteps,
    numflux,
    numsol,
    max_age,
    timeseries_length,
):
    """Accumulate results into output arrays."""

    # Update output concentration
    for jt_fullstep in range(timeseries_length):
        for jt_ws in range(n_substeps):
            for iq in range(numflux):
                if Q_fullstep[jt_fullstep, iq] > 0.0:
                    c = (total_num_substeps + jt_fullstep * n_substeps + jt_ws - iT_substep) % total_num_substeps
                    for s in range(numsol):
                        C_Q_fullstep[jt_fullstep, iq, s] += (
                            mQ_aver[c, iq, s] * dt_substep / Q_fullstep[jt_fullstep, iq] / n_substeps
                        )

    # Update old water fraction
    for c in range(total_num_substeps):
        jt_c = jt_fullstep_at[c]
        for iq in range(numflux):
            P_old_fullstep[jt_c, iq] -= pQ_aver[c, iq] * dt_substep / n_substeps

    # Timestep-averaged transit time distribution
    if iT_fullstep < max_age - 1:
        for outputstep in range(num_output_fullsteps):
            jt_fullstep = output_these_fullsteps[outputstep]
            for jt_ws in range(n_substeps):
                if jt_ws < substep:
                    c = (total_num_substeps + jt_fullstep * n_substeps + jt_ws - iT_substep) % total_num_substeps
                    for iq in range(numflux):
                        pQ_outputstep[outputstep, iq, iT_fullstep + 1] += pQ_aver[c, iq] * norm
                        for s in range(numsol):
                            mQ_outputstep[outputstep, iq, s, iT_fullstep + 1] += mQ_aver[c, iq, s] * norm
                    for s in range(numsol):
                        mR_outputstep[outputstep, s, iT_fullstep + 1] += mR_aver[c, s] * norm

    for outputstep in range(num_output_fullsteps):
        jt_fullstep = output_these_fullsteps[outputstep]
        for jt_ws in range(n_substeps):
            if jt_ws >= substep:
                c = (total_num_substeps + jt_fullstep * n_substeps + jt_ws - iT_substep) % total_num_substeps
                for iq in range(numflux):
                    pQ_outputstep[outputstep, iq, iT_fullstep] += pQ_aver[c, iq] * norm
                    for s in range(numsol):
                        mQ_outputstep[outputstep, iq, s, iT_fullstep] += mQ_aver[c, iq, s] * norm
                for s in range(numsol):
                    mR_outputstep[outputstep, s, iT_fullstep] += mR_aver[c, s] * norm

    # Extract substep state at output timesteps
    for outputstep in range(num_output_fullsteps):
        jt_fullstep = output_these_fullsteps[outputstep]
        jt_ws = n_substeps - 1
        c = (total_num_substeps + jt_fullstep * n_substeps + jt_ws - iT_substep) % total_num_substeps
        sT_outputstep[outputstep + 1, iT_fullstep] += sT_start[c] / n_substeps
        for s in range(numsol):
            mT_outputstep[outputstep + 1, s, iT_fullstep] += mT_start[c, s] / n_substeps


@_maybe_jit
def _calculate_balances(
    iT_fullstep_unused,
    max_age,
    num_output_fullsteps,
    output_these_fullsteps,
    J_fullstep,
    Q_fullstep,
    C_J_fullstep,
    sT_outputstep,
    mT_outputstep,
    pQ_outputstep,
    mQ_outputstep,
    mR_outputstep,
    WaterBalance_outputstep,
    SoluteBalance_outputstep,
    dt,
    numflux,
    numsol,
):
    """Compute water and solute mass balance residuals."""
    for iT_fullstep in range(max_age):
        for outputstep in range(num_output_fullsteps):
            jt_fullstep = output_these_fullsteps[outputstep]

            # Water balance
            if iT_fullstep == 0:
                wb = J_fullstep[jt_fullstep] - sT_outputstep[outputstep + 1, iT_fullstep]
            else:
                wb = sT_outputstep[outputstep, iT_fullstep - 1] - sT_outputstep[outputstep + 1, iT_fullstep]
            for iq in range(numflux):
                wb -= Q_fullstep[jt_fullstep, iq] * pQ_outputstep[outputstep, iq, iT_fullstep] * dt
            WaterBalance_outputstep[outputstep, iT_fullstep] = wb

            # Solute balance
            for s in range(numsol):
                if iT_fullstep == 0:
                    sb = (
                        C_J_fullstep[jt_fullstep, s] * J_fullstep[jt_fullstep]
                        - mT_outputstep[outputstep + 1, s, iT_fullstep]
                    )
                else:
                    sb = mT_outputstep[outputstep, s, iT_fullstep - 1] - mT_outputstep[outputstep + 1, s, iT_fullstep]
                for iq in range(numflux):
                    sb -= mQ_outputstep[outputstep, iq, s, iT_fullstep] * dt
                sb += mR_outputstep[outputstep, s, iT_fullstep] * dt
                SoluteBalance_outputstep[outputstep, s, iT_fullstep] = sb


# ---------------------------------------------------------------------------
# Public API — matches the f2py interface
# ---------------------------------------------------------------------------


def solve(
    J_fullstep,
    Q_fullstep,
    SAS_args,
    P_list,
    weights_fullstep,
    sT_init_fullstep,
    dt,
    verbose,
    debug,
    warning,
    jacobian,
    mT_init_fullstep,
    C_J_fullstep,
    alpha_fullstep,
    k1_fullstep,
    C_eq_fullstep,
    C_old,
    n_substeps,
    component_type,
    numcomponent_list,
    numargs_list,
    output_these_fullsteps,
    num_scheme,
    numflux,
    numsol,
    max_age,
    timeseries_length,
    num_output_fullsteps,
    numcomponent_total,
    numargs_total,
):
    """SAS transport solver — drop-in replacement for the Fortran solvesas.

    Parameters and return values match the f2py interface exactly.
    """
    # Ensure contiguous 2D float64 arrays.
    # SAS_args and P_list may arrive as 3D from _create_sas_lookup;
    # the Fortran f2py binding auto-squeezed them, so we reshape here.
    J_fullstep = np.ascontiguousarray(J_fullstep, dtype=np.float64)
    Q_fullstep = np.ascontiguousarray(Q_fullstep, dtype=np.float64)
    SAS_args = np.ascontiguousarray(SAS_args, dtype=np.float64).ravel().reshape(numargs_total, timeseries_length)
    P_list = np.ascontiguousarray(P_list, dtype=np.float64).ravel().reshape(numargs_total, timeseries_length)
    weights_fullstep = np.ascontiguousarray(weights_fullstep, dtype=np.float64)
    sT_init_fullstep = np.ascontiguousarray(sT_init_fullstep, dtype=np.float64)
    mT_init_fullstep = np.ascontiguousarray(mT_init_fullstep, dtype=np.float64)
    C_J_fullstep = np.ascontiguousarray(C_J_fullstep, dtype=np.float64)
    alpha_fullstep = np.ascontiguousarray(alpha_fullstep, dtype=np.float64)
    k1_fullstep = np.ascontiguousarray(k1_fullstep, dtype=np.float64)
    C_eq_fullstep = np.ascontiguousarray(C_eq_fullstep, dtype=np.float64)
    C_old = np.ascontiguousarray(C_old, dtype=np.float64)
    component_type = np.ascontiguousarray(component_type, dtype=np.int64)
    numcomponent_list = np.ascontiguousarray(numcomponent_list, dtype=np.int64)
    numargs_list = np.ascontiguousarray(numargs_list, dtype=np.int64)
    output_these_fullsteps = np.ascontiguousarray(output_these_fullsteps, dtype=np.int64)

    return _solve_core(
        J_fullstep,
        Q_fullstep,
        SAS_args,
        P_list,
        weights_fullstep,
        sT_init_fullstep,
        float(dt),
        bool(verbose),
        mT_init_fullstep,
        C_J_fullstep,
        alpha_fullstep,
        k1_fullstep,
        C_eq_fullstep,
        C_old,
        int(n_substeps),
        component_type,
        numcomponent_list,
        numargs_list,
        output_these_fullsteps,
        int(num_scheme),
        int(numflux),
        int(numsol),
        int(max_age),
        int(timeseries_length),
        int(num_output_fullsteps),
        int(numcomponent_total),
        int(numargs_total),
    )
