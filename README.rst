`hupa-voicedb`: Príncipe de Asturias Hospital Voice Disorders Database Reader module
==========================================================================

|pypi| |status| |pyver| |license|

.. |pypi| image:: https://img.shields.io/pypi/v/hupa-voicedb
  :alt: PyPI
.. |status| image:: https://img.shields.io/pypi/status/hupa-voicedb
  :alt: PyPI - Status
.. |pyver| image:: https://img.shields.io/pypi/pyversions/hupa-voicedb
  :alt: PyPI - Python Version
.. |license| image:: https://img.shields.io/github/license/tikuma-lsuhsc/python-hupa-voicedb
  :alt: GitHub

.. note::
   This Python package is still under development.

This Python module provides functions to retrieve data and information easily from 
Príncipe de Asturias Hospital Voice Disorders Database.

This module is not the database itself. User must contact Dr. Juan Ignacio Godino Llorente at
Universidad Politécnica de Madrid to obtain the license and to download the database.

Install
-------

.. code-block:: bash

  pip install hupa-voicedb

Use
---

.. code-block:: python

  from hupa import HUPA

  # to initialize (must call this once in every Python session)
  db = HUPA('<path to the root directory of the extracted database>')

  # to get a copy of the full database as a Pandas dataframe
  df = db.query()

  # to get age, gender, and R scores
  df = db.query(["edad", "sexo", "R"])

  # to get a dataframe of all the WAV files with pathology code
  df = db.get_files(auxdata_fields=['Codido'])

  # to iterate acoustic data over all female speakers along with age and G score
  for id, fs, x, auxdata in db.iter_data(auxdata_fields=["edad","G"],
                                         sexo="M"):
    # run the acoustic data through your analysis function, get measurements
    params = my_analysis_function(fs, x)

    # log the measurements along with the age and GRBAS info
    my_logger.log_outcome(id, *auxdata, *params)
