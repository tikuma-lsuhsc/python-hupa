from hupa import HUPA
import pandas as pd
import numpy as np
from glob import glob
import re
import pytest
import winreg
from os import path


def load_db():
    key = winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, "Software\Microsoft\OneDrive\Accounts\Business1"
    )
    datadir = path.join(
        winreg.QueryValueEx(key, "UserFolder")[0], "data", "BDAtos HUPA Segmentada"
    )
    return HUPA(datadir)


@pytest.fixture(scope="module")
def hupa():
    return load_db()


def test_query(hupa):
    df = hupa.query()
    df = hupa.query(include_cape_v=True)
    df = hupa.query(include_cape_v=["severity"], rating_stats=["mean"])
    df = hupa.query(include_grbas="breathiness", rating_stats="mean")
    print(df)


def test_files(hupa):
    print(hupa.get_files("/a/"))
    print(
        hupa.get_files(
            "blue",
            "age",
            include_cape_v="severity",
            include_grbas="grade",
            Gender="male",
        )
    )


def test_iter_data(hupa):
    for id, fs, x in hupa.iter_data("/a/"):
        pass
    for id, fs, x, info in hupa.iter_data(
        "/i/", auxdata_fields=["Gender", "Age"], include_cape_v="severity"
    ):
        pass


def test_read_data(hupa):
    id = "BL01"
    hupa.read_data(id, padding=0.01)  # full data file
    types = hupa.task_types  # audio segment types
    for t in types:
        hupa.read_data(id, t)
