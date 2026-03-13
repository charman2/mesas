# MESAS Refactoring Progress

This document tracks progress against `REFACTORING_PLAN.md` so that a new session can pick up where the last left off.

## Current State

**Branch:** `stochastic` (the active development branch, ahead of `origin/main` by ~100 commits)
**Working stage:** Stage 0 — Establish a Testing Baseline
**Last commit:** `2bf9a57` — Added refactoring plan, tracked untracked files, updated .gitignore

## Completed

### Pre-work
- [x] Explored entire codebase and wrote detailed evaluation (in REFACTORING_PLAN.md)
- [x] Fetched and summarised the GMD paper for scientific context
- [x] Created staged refactoring plan with user-approved scope
- [x] Reviewed git state: aborted stale cherry-pick, unstaged generated PDF
- [x] Added untracked files that should be tracked (test files, example files)
- [x] Updated `.gitignore` to exclude debug/, generated files, IDE configs, .claude/
- [x] Committed baseline (commit `2bf9a57`)

## In Progress

### Stage 0: Testing Baseline

Items to complete (in order):
1. **Fix pytest discovery** — `pytest.ini` points to `mesas/test` (doesn't exist); tests are in `test/`
2. **Formalise steady-state benchmarks** — rewrite `test_time.py` as proper pytest with assertions and tolerances
3. **Regression tests for examples** — `lower_hafren` and `hyporheic` examples with saved reference outputs
4. **Mass-balance tests** — verify water/solute balance closure
5. **Edge-case tests** — no solutes, single timestep, zero fluxes, etc.
6. **Performance benchmark** — record wall-clock time for regression detection
7. **Code coverage** — add pytest-cov, establish baseline
8. **Pre-commit hooks** — ruff + black
9. **CI** — GitHub Actions for Linux/macOS/Windows

## Not Yet Started

- Stage 1: Code Quality and Documentation
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

5. **Branch topology:** `stochastic` == `master` in content, both ahead of `origin/main`. The refactoring work is being done on `stochastic`.

6. **The Fortran solver must be installed** for any tests to run. Ensure `pip install .` (or equivalent) has been done in the environment before running pytest.

## Environment Notes

- Platform: macOS (Darwin 23.4.0)
- Python: check with `python --version`
- The package must be built from source (Fortran compilation required) before tests can run
- Conda environment likely needed (see `environment.yml`)
