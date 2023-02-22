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

    df = hupa.query(edad=[50, 70])
    print(df)


def test_files(hupa):
    print(hupa.get_files())
    print(hupa.get_files(["edad", "Codido", "R"], sexo="male"))


def test_iter_data(hupa):
    for id, fs, x in hupa.iter_data():
        pass


def test_read_data(hupa):
    id = "jcfa"
    hupa.read_data(id)  # full data file
