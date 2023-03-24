from hupa import HUPA
import pandas as pd
import numpy as np
from glob import glob
import re
import pytest
import winreg
from os import path


def test_query(hupa):
    df = hupa.query()
    print(df)
    assert df['sexo'].isin(['M','H']).all()

    df = hupa.query(edad=[50, 70])
    print(df)


def test_files(hupa):
    print(hupa.get_files())
    print(hupa.get_files(["edad", "Codigo", "R"], sexo="H"))


def test_iter_data(hupa):
    for id, fs, x in hupa.iter_data():
        pass


def test_read_data(hupa):
    id = "jcfa"
    hupa.read_data(id)  # full data file

    # to get age, gender, and R scores
    df = hupa.query(["edad", "sexo", "R", "Codigo"])

    # use Pandas' itertuples to read audio data iteratively
    for id, *info in df.itertuples():
        try:
            fs, x = hupa.read_data(id) # normalize to [0,1] unless given additional argument: normlize=False
        except:
            id
