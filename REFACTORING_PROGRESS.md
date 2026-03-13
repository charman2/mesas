# MESAS Refactoring Progress

This document tracks progress against `REFACTORING_PLAN.md` so that a new session can pick up where the last left off.

## Current State

**Branch:** `stochastic`
**Working stage:** Stage 1 — COMPLETE
**Last commit:** `fa3f88f` — Stage 1 docstrings, type hints, comments, and variable renames

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

## Not Yet Started

- Stage 2: Fix the Build System
- Stage 3: Replace the Fortran Solver
- Stage 4: API Cleanup
- Stage 5: Documentation Overhaul
- Stage 6: Extended Testing

## Key Decisions / Context for Future Sessions

1. **Performance is a hard constraint.** User explicitly stated speed must not regress. Stage 3 targets Fortran-parity via Numba `@njit`, not "acceptable slowdown."

2. **All optional items were approved** except benchmarking against tran-SAS (Stage 6).

3. **The `recursive_split` module is broken** — references `model.sas_blends` which doesn't exist (should be `sas_specs`). Same bug in `vis.py:plot_SAS_cumulative`. Don't fix these yet (that's Stage 1+), but be aware tests involving these modules will fail.

4. **Build system is broken on NumPy >= 2.0** — `setup.py` uses removed `numpy.distutils`. The package currently only builds on older NumPy. Stage 2 addresses this.

5. **Branch topology:** `stochastic` is the active development branch. Refactoring work is done here.

6. **The Fortran solver must be installed** for any tests to run. Use `conda run -n mesas11 pip install --no-build-isolation -e .` to build.

7. **Water/solute balance tests use `record_state=True`** — the balance arrays are only meaningful when all consecutive timesteps are recorded. Example tests (large datasets) check only age=0 with default `record_state=False`.

8. **SoluteBalance bug was fixed** — `solve.f90` lines 457-464 had `mT*dt` where it should have been just `mT` (paralleling the WaterBalance formula which uses `sT` without `*dt`).

## Environment Notes

- Platform: macOS (Darwin 23.4.0)
- Conda environment: `mesas11` (Python 3.11.9, NumPy 1.26.4)
- All commands must use `conda run -n mesas11` prefix
- Pre-commit hooks installed (ruff linter + formatter)
