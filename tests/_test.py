import pandas as pd
import numpy as np

import winreg
from os import path

from contextlib import suppress
import itertools

reg_base = "Software\Microsoft\OneDrive\Accounts"
with suppress(WindowsError), winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_base) as key:
    for i in itertools.count():
        with winreg.OpenKey(key, winreg.EnumKey(key, i)) as acct:
            drive_dir = winreg.QueryValueEx(acct, "UserFolder")[0]
            if "LSUHSC" in drive_dir:
                break

dbdir = path.join(drive_dir, "data", "BDAtos HUPA Segmentada")

xlsfile = path.join(dbdir, "HUPA segmentada.xls")

assert path.exists(xlsfile)

df1 = pd.concat(
    {
        sheet: pd.read_excel(
            xlsfile,
            sheet_name=sheet,
            skiprows=1,
            dtype={
                "Archivo": "string",
                "Fs": "string",
                "Tipo": "string",
                "EGG": "string",
                "edad": "Int8",
                "sexo": "string",
                "G": "Int8",
                "R": "Int8",
                "A": "Int8",
                "B": "Int8",
                "S": "Int8",
                "Total": "Int8",
                "Codigo": "string",
                "Patología": "string",
                # "F0": float,
                # "F1": float,
                # "F2": float,
                # "F3": float,
                "Formantes": "string",
                "Picos": "string",
                "Jitter": "string",
                "Comentarios": "string",
            },
        )
        for sheet in ("Normales", "Patológicos")
    },
    ignore_index=True,
)

df1["Fs"] = df1["Fs"].apply(lambda v: {"25 kHz": 25000}[v])
df1.index = df1["Archivo"].str[:-4].rename("ID")

df3 = (
    pd.read_excel(xlsfile, sheet_name="Clasificación patologías", dtype="string")
    .rename(columns={"Unnamed: 2": "Pathology"})
    .set_index("Codigo")
)

# ,
#     dtype={"Age": "Int32", "Diagnosis ": "string"},
#     nrows=297,
#     keep_default_na=False,
#     na_values={"Diagnosis": ""},
#     # fmt:off
#     converters={
#         "Participant ID ": lambda v: v.strip().upper(),
#         "Gender": lambda v: {"m": "male", "f": "female"}.get(v.lower(), v.lower()),
#     },
#     # fmt:on
# ).set_index("Participant ID ")
# df.columns = df.columns.str.strip()
# df.index.name = "ID"
# self._df = df
