.. _results:

==================
Extracting results
==================

After calling ``model.run()``, results are available in two ways:

1. **Outflux concentrations** appear as new columns in ``model.data_df``
   with names like ``"<solute> --> <flux>"``.

2. **Age-ranked state variables** are accessible through getter methods
   or the :class:`~mesas.sas.model.ModelResult` object at
   ``model.result``.

Saving concentration timeseries
================================

.. code-block:: python

    model.run()
    model.data_df.to_csv("results.csv")

Getter methods
==============

The following methods return age-ranked arrays. Each accepts optional
``timestep``, ``agestep``, or ``inputtime`` keyword arguments to
slice the result.

``model.get_sT()``
    Age-ranked storage density. Shape ``(max_age, n_output_steps)``.

``model.get_pQ(flux)``
    Age-ranked transit time distribution for the named flux.

``model.get_mT(sol)``
    Age-ranked solute mass density for the named solute.

``model.get_CT(sol)``
    Age-ranked concentration ``mT / sT`` (NaN where ``sT = 0``).

``model.get_mQ(flux, sol)``
    Age-ranked solute mass flux for a given flux and solute.

``model.get_mR(sol)``
    Age-ranked reaction mass for a given solute.

``model.get_water_balance()``
    Water conservation residual. Should be near machine precision when
    ``record_state=True``.

``model.get_solute_balance(sol)``
    Solute conservation residual for a given solute.

``model.get_ST()``
    Cumulative storage ``ST = cumsum(sT) * dt``.

The ModelResult object
======================

``model.result`` returns a :class:`~mesas.sas.model.ModelResult` that
supports both dict-style and attribute-style access:

.. code-block:: python

    # These are equivalent:
    sT = model.result["sT"]
    sT = model.result.sT

    # Snake_case names for balance arrays:
    wb = model.result.water_balance
    sb = model.result.solute_balance

Available keys: ``sT``, ``pQ``, ``water_balance``, ``dsTdSj``,
``C_Q``, ``mT``, ``mQ``, ``mR``, ``solute_balance``, ``dmTdSj``,
``dCdSj``.

.. note::

    The camelCase names ``WaterBalance`` and ``SoluteBalance`` are
    deprecated and will emit a warning. Use ``water_balance`` and
    ``solute_balance`` instead.

Array ordering
==============

Getter methods return arrays with age as the first axis and timestep
as the second:

- ``sT``: ``(max_age, n_output_steps)``
- ``pQ``: ``(max_age, n_output_steps, n_fluxes)``
- ``mT``: ``(max_age, n_output_steps, n_solutes)``
- ``C_Q``: ``(n_timesteps, n_fluxes, n_solutes)``

The order of fluxes and solutes matches ``model.fluxorder`` and
``model.solorder``.
