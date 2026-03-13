# MESAS Refactoring Progress

This document tracks progress against `REFACTORING_PLAN.md` so that a new session can pick up where the last left off.

## Current State

**Branch:** `stochastic`
**Working stage:** Stage 4 — COMPLETE
**Last commit:** (uncommitted — ready for commit)

## Completed

### Pre-work
- [x] Explored entire codebase and wrote detailed evaluation (in REFACTORING_PLAN.md)
- [x] Fetched and summarised the GMD paper for scientific context
- [x] Created staged refactoring plan with user-approved scope
- [x] Reviewed git state: aborted stale cherry-pick, unstaged generated PDF
- [x] Added untracked files that should be tracked (test files, example files)
- [x] Updated `.gitignore` to exclude debug/, generated files, IDE configs, .claude/
- [x] Committed baseline (commit `2bf9a57`)

### Stage 0: Testing Baseline
- [x] Fixed pytest discovery (`pytest.ini` and `pyproject.toml` pointed to wrong path)
- [x] Added `test/conftest.py` with autouse chdir fixture
- [x] Wrote `test_stage0_benchmarks.py` — 4 analytical benchmarks (3 steady, 1 unsteady)
- [x] Wrote `test_stage0_examples.py` — regression tests for lower_hafren and hyporheic
- [x] Wrote `test_stage0_mass_balance.py` — water balance + solute balance closure tests
- [x] Wrote `test_stage0_edge_cases.py` — no solutes, short timeseries, numerical schemes, piecewise segments, result accessors
- [x] Wrote `test_stage0_performance.py` — 4 wall-clock benchmarks
- [x] **Fixed bug in `solve.f90` SoluteBalance calculation** — removed erroneous `*dt` factor on mT storage terms that broke solute mass conservation
- [x] Added pytest-cov config (baseline: 44% coverage)
- [x] Added ruff config to pyproject.toml
- [x] Added `.pre-commit-config.yaml` with ruff linter + formatter
- [x] Added GitHub Actions CI (`.github/workflows/tests.yml`) for Linux/macOS, Python 3.10/3.11
- [x] All 46 new tests pass (60 total, 2 pre-existing path failures in test_time.py/test_time2.py)

### Stage 1: Code Quality and Documentation
- [x] Ran ruff linter + formatter across Python codebase (formatting-only commit `42207c3`)
- [x] Added NumPy-style docstrings to all public classes/methods in `model.py`, `specs.py`, `functions.py`
- [x] Added module docstrings to `recursive_split.py` and `vis.py` (noting broken `sas_blends` references)
- [x] Added `from __future__ import annotations` and type hints to all public Python APIs
- [x] Added inline comments to `model.py:run()` and `model.py:_create_sas_lookup()` explaining Python–Fortran data marshalling
- [x] Added block-level comments to `solve.f90` explaining major sections (init, RK stepping, flux calc, mass balance)
- [x] Renamed unclear variables: `repr` → `result` (shadowed builtin), `iP` → `param_offset`, `nP` → `n_breakpoints`, `A` → `endpoint_to_segment`, `ri` → `residuals`, `ex` → `err`
- [x] Fixed typo: `selfs` → `self` in `Continuous.func` setter
- [x] Removed dead commented-out code in `__repr__` methods
- [x] Fixed `np.NaN` → `np.nan`, `== True/False` → `is True/False`, import order issues
- [x] Created `CONTRIBUTING.md` with dev setup, testing, and code style instructions
- [x] Updated `mesas/__init__.py` with convenience import (`from mesas import Model`)
- [x] All 46 Stage 0 tests pass — zero tolerance change

### Stage 2: Fix the Build System
- [x] Replaced `numpy.distutils` + `setup.py` with `meson-python` build backend
- [x] Created `meson.build` files (root, mesas/, mesas/sas/, mesas/me/, mesas/utils/)
- [x] f2py Fortran extension builds via `custom_target` + `py.extension_module`
- [x] Updated `pyproject.toml`: meson-python backend, fixed classifiers (Fortran not Cython), `requires-python >= 3.10`
- [x] Removed old `setup.py` and `recompile.sh`
- [x] Cleaned up `mesas/sas/__init__.py` (removed DLL-loading hack for old build)
- [x] Updated CI to install meson/ninja, added Python 3.12 and Windows to matrix
- [x] Validated compiler flags portably (`fc.has_argument`)
- [x] Editable install (`pip install -e .`) works
- [x] All 46 Stage 0 tests pass — zero tolerance change

### Stage 3: Replace the Fortran Solver
- [x] Wrote `mesas/sas/_solve_numba.py` — pure Python + Numba replacement for `solve.f90`
  - Implements all SAS function evaluators: piecewise-linear, gamma, beta, Kumaraswamy
  - Includes Numba-compatible special functions (_alngam, _gammad, _betain) matching AS 245/239/63
  - Core solver `_solve_core` with RK1/RK2/RK4 integration, identical to Fortran
  - Public `solve()` wrapper handles array type/shape conversion (3D→2D, Fortran→C layout)
- [x] Updated `model.py`: import from `_solve_numba`, use `np.ascontiguousarray` instead of `np.asfortranarray`
- [x] Fixed off-by-one bug in STcum copy loop (`range(tns)` → `range(tns+1)`)
- [x] Fixed Numba type inference: replaced array slices with full-array+offset indexing, explicit `float()` casts
- [x] Removed Fortran from build system: no `'c'`/`'fortran'` languages in `meson.build`, removed f2py extension
- [x] Added `numba` to project dependencies in `pyproject.toml`
- [x] Updated CI: removed `compilers` dependency, added `numba`
- [x] Fixed critical bug in `_gammad` (incomplete gamma integral): series branch had extra 1/p factor, continued fraction was missing `cc` counter and had wrong initial result
- [x] Fixed reshape in `solve()` wrapper for `max_age < timeseries_length` cases (`.ravel()` before `.reshape()`)
- [x] All 46 Stage 0 tests pass — including Lower Hafren (gamma SAS) and Hyporheic examples
- [x] All 4 performance benchmarks pass
- [x] Numba `@njit(cache=True)` compilation works — cached after first run

### Stage 4: API Cleanup and Modernisation
- [x] Created `ModelOptions` dataclass — replaces mutable default dict with typed, validated fields
  - `from_dict()` / `update()` / `to_dict()` for backward compat with dict-based API
  - Proper `KeyError` on invalid option names
  - `_apply_options()` consolidates max_age/sT_init/index_ts resolution
- [x] Created `SoluteSpec` dataclass — typed alternative to raw dicts for solute parameters
  - `from_dict()` constructor fills in default alpha per flux
  - Internal solute storage remains dict-based for backward compat
- [x] Created `ModelResult` class — attribute-style and dict-style access to results
  - `result.sT`, `result["sT"]`, `result.water_balance` all work
  - Deprecated camelCase keys (`"WaterBalance"`, `"SoluteBalance"`) emit `DeprecationWarning`
  - `__repr__` shows result shapes
- [x] Standardised naming to snake_case
  - Result keys: `water_balance` (was `WaterBalance`), `solute_balance` (was `SoluteBalance`)
  - Methods: `get_water_balance()` (was `get_WaterBalance()`), `get_solute_balance()` (was `get_SoluteBalance()`)
  - Old camelCase methods kept as deprecated aliases
- [x] Replaced all `assert` statements with descriptive `ValueError`/`TypeError`
  - `model.py`: `parse_sas_specs()`, `_get_result()`, `_apply_options()`
  - `functions.py`: ST/P validation in `Piecewise` and `Continuous` constructors/setters
  - `specs.py`: Component `args` validation
  - Error messages include actual values and suggestions (e.g. ST_largest_segment hint)
- [x] Fixed `np.NaN` → `np.nan` in `functions.py`
- [x] Fixed `copy_without_results()` — handles missing `_components_to_learn`, uses keyword args
- [x] Updated `mesas/__init__.py` to export `ModelOptions`, `ModelResult`, `SoluteSpec`
- [x] Updated all 46 tests to use new snake_case API
- [x] All 46 tests pass with `DeprecationWarning` treated as error (zero regressions)

## Not Yet Started

- Stage 5: Documentation Overhaul
- Stage 6: Extended Testing

## Key Decisions / Context for Future Sessions

1. **Performance is a hard constraint.** User explicitly stated speed must not regress. Stage 3 targets Fortran-parity via Numba `@njit`, not "acceptable slowdown."

2. **All optional items were approved** except benchmarking against tran-SAS (Stage 6).

3. **The `recursive_split` module is broken** — references `model.sas_blends` which doesn't exist (should be `sas_specs`). Same bug in `vis.py:plot_SAS_cumulative`. Don't fix these yet (that's Stage 1+), but be aware tests involving these modules will fail.

4. **Build system now uses meson-python** — replaced broken `numpy.distutils` + `setup.py`. Build with `pip install --no-build-isolation -e .` (requires meson, ninja, numpy in environment).

5. **Branch topology:** `stochastic` is the active development branch. Refactoring work is done here.

6. **Fortran is no longer required.** The solver is now pure Python + Numba. Build with `pip install --no-build-isolation -e .`.

7. **Water/solute balance tests use `record_state=True`** — the balance arrays are only meaningful when all consecutive timesteps are recorded. Example tests (large datasets) check only age=0 with default `record_state=False`.

8. **SoluteBalance bug was fixed** — `solve.f90` lines 457-464 had `mT*dt` where it should have been just `mT` (paralleling the WaterBalance formula which uses `sT` without `*dt`).

## Environment Notes

- Platform: macOS (Darwin 23.4.0)
- Conda environment: `mesas11` (Python 3.11.9, NumPy 1.26.4)
- All commands must use `conda run -n mesas11` prefix
- Pre-commit hooks installed (ruff linter + formatter)
