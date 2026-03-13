
============
Installation
============

From PyPI or conda-forge
========================

The simplest way to install mesas is with pip::

    pip install mesas

Or using Conda::

    conda install -c conda-forge mesas

Either method will install all required dependencies (NumPy, SciPy, pandas,
Numba, matplotlib).

From source
===========

To install the latest development version from GitHub::

    git clone https://github.com/charman2/mesas.git
    cd mesas
    pip install -e .

.. note::

   No Fortran compiler is needed. The numerical solver uses
   `Numba <https://numba.pydata.org/>`_ for JIT compilation of
   pure Python code.

Requirements
============

- Python >= 3.10
- NumPy >= 1.22
- SciPy
- pandas
- Numba
- matplotlib
