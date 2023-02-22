`hupa-voicedb`: Príncipe de Asturias Hospital Voice Disorders Database Reader module
====================================================================================

|pypi| |status| |pyver| |license|

.. |pypi| image:: https://img.shields.io/pypi/v/hupa-voicedb
  :alt: PyPI
.. |status| image:: https://img.shields.io/pypi/status/hupa-voicedb
  :alt: PyPI - Status
.. |pyver| image:: https://img.shields.io/pypi/pyversions/hupa-voicedb
  :alt: PyPI - Python Version
.. |license| image:: https://img.shields.io/github/license/tikuma-lsuhsc/python-hupa-voicedb
  :alt: GitHub

This Python module provides functions to retrieve data and information easily from 
Príncipe de Asturias Hospital Voice Disorders Database.

**This module does not contain the database itself.** The database belongs to Prof. Juan I. 
Godino-Llorente (email: ignacio.godino@upm.es) at Universidad Politécnica de Madrid, and 
he kindly makes it available for free to non-commercial research use. Users must 
contact him to obtain the license and to download the database.

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
  df = db.query() # default columns: "edad", "sexo", "Codigo"

  # to get the patholgy code-name lookup table 
  # (note: not all pathologies are included in the database)
  lut = db.pathologies

  # to get age, gender, and R scores
  df = db.query(["edad", "sexo", "R"])

  # use Pandas' itertuples to read audio data iteratively
  for id, *info in df.itertuples():
    # read audio data
    # (normalize to [0,1] unless given additional argument: normlize=False)
    fs, x = db.read_data(id) 

    # run the acoustic data through your analysis function, get measurements
    params = my_analysis_function(fs, x)

    # log the measurements along with the age and GRBAS info
    my_logger.log_outcome(id, *auxdata, *params)

  # alternately, use database's `iter_data` method to process acoustic data 
  # iteratively over queried data (all female speakers along with age and G score)
  for id, fs, x, auxdata in db.iter_data(auxdata_fields=["edad", "G"],
                                         sexo="M"):
    # run the acoustic data through your analysis function, get measurements
    params = my_analysis_function(fs, x)

    # log the measurements along with the age and GRBAS info
    my_logger.log_outcome(id, *auxdata, *params)

  # Finally, to get a dataframe of all the WAV files with their full paths
  df = db.get_files(auxdata_fields=['Codigo'])
