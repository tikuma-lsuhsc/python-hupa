"""Voice Foundation Pathological Voice Quality Database Reader module

TODO: download files directly from https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/9dz247gnyb-2.zip
"""


__version__ = "0.1.0.dev1"
__all__ = ["HUPA"]

import pandas as pd
from os import path
import numpy as np
import wave
from collections.abc import Iterator, Sequence
from typing import Tuple, Optional, List, Union


class HUPA:
    def __init__(self, dbdir: str):
        """PVQD constructor

        :param dbdir: path to the directory of the downloaded database
        :type dbdir: str
        """

        # database variables
        self._dir = None  # database dir
        self._df = None  # main database table
        self._df_dx = None  # patient diagnosis table

        # load the database
        self._load_db(dbdir)

    def _load_db(self, dbdir):
        """load disordered voice database

        :param dbdir: path to the cdrom drive or the directory hosting a copy of the database
        :type dbdir: str
        """

        if self._dir == dbdir:
            return

        xlsfile = path.join(dbdir, "HUPA segmentada.xls")
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

        df1.loc[df1["Archivo"] == "Mpda.nsp.wav", "Archivo"] = "Mpda.wav"  # fix

        df1["Fs"] = df1["Fs"].apply(lambda v: {"25 kHz": 25000}[v])
        df1.index = df1["Archivo"].str.split(".", n=1, expand=True)[0].rename("ID")
        df1["Archivo"] = df1["Archivo"].str.lower()  # files are all in lower case

        df2 = (
            pd.read_excel(
                xlsfile, sheet_name="Clasificación patologías", dtype="string"
            )
            .rename(columns={"Unnamed: 2": "Pathology"})
            .set_index("Codigo")
        )

        self._df = df1
        self._df_dx = df2
        self._dir = dbdir

    def query(
        self,
        columns: Sequence[str] = None,
        **filters,
    ) -> pd.DataFrame:
        """query database

        :param columns: database columns to return, defaults to None
        :type columns: Sequence[str], optional
        :param **filters: query conditions (values) for specific per-database columns
        :type **filters: dict
        :return: query result
        :rtype: pandas.DataFrame

        Default Output Columns
        ----------------------

        If `columns` is not specified, the following columns are returned by default:

        * edad (age)
        * sexo (gender)
        * Codigo (pathology code, normophonic if '0')

        Valid `filters` keyword argument values
        ---------------------------------------

        * A scalar value
        * For numeric and date columns, 2-element sequence to define a range: [start, end)
        * For all other columns, a sequence of allowable values

        """

        if columns is None:
            columns = ["edad", "sexo", "Codigo"]

        # work on a copy of the dataframe
        df = self._df.copy(deep=True)

        # apply the filters to reduce the rows
        for fcol, fcond in filters.items():
            try:
                s = df.index if fcol == "ID" else df[fcol]
            except:
                raise ValueError(f"{fcol} is not a valid column label")

            try:  # try range/multi-choices
                if s.dtype.kind in "iufcM":  # numeric/date
                    # 2-element range condition
                    df = df[(s >= fcond[0]) & (s < fcond[1])]
                else:  # non-numeric
                    df = df[s.isin(fcond)]  # choice condition
            except:
                # look for the exact match
                df = df[s == fcond]

        # return only the selected columns
        try:
            df = df[columns]
        except:
            ValueError(
                f'At least one label in the "columns" argument is invalid: {columns}'
            )

        return df

    def get_files(
        self,
        auxdata_fields: Sequence[str] = None,
        **filters,
    ) -> pd.DataFrame:
        """get WAV filepaths, and starting and ending time markers

        :param auxdata_fields: names of auxiliary data fields to return, defaults to None
        :type auxdata_fields: Sequence[str], optional
        :param **filters: query conditions (values) for specific per-database columns (keys)
        :type **filters: dict
        :return: data frame containing file path and auxdata
        :rtype: pandas.DataFrame
        """

        columns = ["Archivo", "Codigo"]
        if auxdata_fields is not None:
            columns.extend(auxdata_fields)

        df = self.query(columns, **filters)

        dir = {
            True: path.join(self._dir, "Normal"),
            False: path.join(self._dir, "Pathol"),
        }

        files = [
            path.join(dir[p == "0"], f)
            for f, p in df.iloc[:, :2].itertuples(index=False)
        ]

        return pd.concat(
            [pd.Series(files, df.index, "string", "File"), df.iloc[:, 2:]], axis=1
        )

    def iter_data(
        self,
        auxdata_fields: Sequence[str] = None,
        normalize: bool = True,
        **filters,
    ) -> Iterator[Tuple[int, np.array, Optional[pd.Series]]]:
        """iterate over data samples

        :param auxdata_fields: names of auxiliary data fields to return, defaults to None
        :type auxdata_fields: sequence of str, optional
        :param normalize: True to return normalized f64 data, False to return i16 data, defaults to True
        :type normalize: bool, optional
        :param **filters: query conditions (values) for specific per-database columns (keys)
        :type **filters: dict
        :yield:
            - sampling rate : audio sampling rate in samples/second
            - data  : audio data, 1-D for 1-channel NSP (only A channel), or 2-D of shape
                    (Nsamples, 2) for 2-channel NSP
            - auxdata : (optional) requested auxdata of the data if auxdata_fields is specified
        :rtype: Iterator[Tuple[int, np.array, Optional[pd.Series]]]

        """

        df = self.get_files(auxdata_fields, **filters)
        aux_data = df.iloc[:, 3:]

        for id, file in df["File"].items():
            data = self._read_file(file, normalize)
            yield (id, *data) if aux_data.empty else (id, *data, aux_data.loc[id])

    def read_data(self, id: str, normalize: bool = True) -> Tuple[int, np.array]:
        """read audio data of the specified recording

        :param id: recording identifier, as returned by query()
        :type id: str
        :param normalize: True to normalize the int16 data between (-1,1), defaults to True
        :type normalize: bool, optional
        :return: a pair of the sampling rate and numpy array
        :rtype: Tuple[int, np.array]
        """

        file, code = self._df.loc[id, ["Archivo", "Codigo"]]

        norm = code == "0"
        dir = path.join(self._dir, "Normal" if norm else "Pathol")

        return self._read_file(path.join(dir, file), normalize)

    def _read_file(self, file, normalize=True):
        with wave.open(file) as wobj:
            nchannels, sampwidth, framerate, nframes, *_ = wobj.getparams()
            b = wobj.readframes(nframes)

        assert nchannels == 1
        assert sampwidth == 2

        x = np.frombuffer(b, "<i2")

        if normalize:
            x = x / 2.0 ** (sampwidth * 8 - 1 if sampwidth > 1 else 8)

        return framerate, x

    def __getitem__(self, key: str) -> Tuple[int, np.array]:
        return self.read_data(key)
