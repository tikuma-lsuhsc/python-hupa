"""Voice Foundation Pathological Voice Quality Database Reader module

TODO: download files directly from https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/9dz247gnyb-2.zip
"""


__version__ = "0.1.0.dev1"

import pandas as pd
from os import path
import numpy as np
from glob import glob as _glob
import re
import wave
from collections.abc import Sequence


class PVQD:
    def __init__(self, dbdir, default_type="/a/", padding=0.0, _timingpath=None):
        """PVQD constructor

        :param dbdir: path to the cdrom drive or the directory hosting a copy of the database
        :type dbdir: str
        """

        self.default_type = default_type
        self.default_padding = padding

        # database variables
        self._dir = None  # database dir
        self._df = None  # main database table
        self._df_rates = None  # patient diagnosis table
        self._df_times = None  # patient diagnoses series
        self._wavs = None

        # load the database
        self._load_db(dbdir, _timingpath)

    def _load_db(self, dbdir, timingpath):
        """load disordered voice database

        :param dbdir: path to the cdrom drive or the directory hosting a copy of the database
        :type dbdir: str

        * This function must be called at the beginning of each Python session
        * Database is loaded from the text file found at: <dbdir>/EXCEL50/TEXT/KAYCDALL.TXT
        * Only entries with NSP files are included
        * PAT_ID of the entries without PAT_ID field uses the "FILE VOWEL 'AH'" field value
        as the PAT_ID value

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

        df1["Fs"] = df1["Fs"].apply(lambda v: {"25 kHz": 25000}[v])
        df1.index = df1["Archivo"].str[:-4].rename("ID")

        df2 = (
            pd.read_excel(
                xlsfile, sheet_name="Clasificación patologías", dtype="string"
            )
            .rename(columns={"Unnamed: 2": "Pathology"})
            .set_index("Codigo")
        )

        self._df = df1
        self._df_dx = df2

    def query(
        self,
        columns=None,
        **filters,
    ):
        """query database

        :param columns: database columns to return, defaults to None
        :type columns: sequence of str, optional
        :param **filters: query conditions (values) for specific per-database columns (keys)
        :type **filters: dict
        :return: query result
        :rtype: pandas.DataFrame

        Valid `filters` keyword argument values
        ---------------------------------------

        * A scalar value
        * For numeric and date columns, 2-element sequence to define a range: [start, end)
        * For all other columns, a sequence of allowable values

        """

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
        if columns is not None:
            try:
                df = df[columns]
            except:
                ValueError(
                    f'At least one label in the "columns" argument is invalid: {columns}'
                )

        return df

    def get_files(
        self,
        auxdata_fields=None,
        **filters,
    ):
        """get WAV filepaths, and starting and ending time markers

        :param auxdata_fields: names of auxiliary data fields to return, defaults to None
        :type auxdata_fields: sequence of str, optional
        :param **filters: query conditions (values) for specific per-database columns (keys)
        :type **filters: dict
        :return: data frame containing file path, start and end time marks, and auxdata
        :rtype: pandas.DataFrame
        """

        columns = ["Archivo", "Codigo"]
        if auxdata_fields is not None:
            columns.extend(auxdata_fields)

        filter_on = len(filters) or bool(auxdata_fields)
        df = (
            self.query(auxdata_fields, **filters)
            if filter_on or bool(auxdata_fields)
            else self._df.copy()
        )

        dir = {
            True: path.join(self._dir, "Normal"),
            False: path.join(self._dir, "Pathol"),
        }

        files = [
            path.join(dir[p == "0"], f)
            for f, p in df.iloc[:, :2].itertuples(index=False)
        ]

        return (
            files
            if auxdata_fields is None
            else (files, df.iloc[:, 2:].reset_index(drop=True))
        )

    def iter_data(
        self,
        type=None,
        auxdata_fields=None,
        normalize=True,
        include_cape_v=None,
        include_grbas=None,
        rating_stats=None,
        **filters,
    ):
        """iterate over data samples

        :param type: utterance type
        :type type: "rainbow" or "ah"
        :param channels: audio channels to read ('a', 'b', 0-1, or a sequence thereof),
                        defaults to None (all channels)
        :type channels: str, int, sequence, optional
        :param auxdata_fields: names of auxiliary data fields to return, defaults to None
        :type auxdata_fields: sequence of str, optional
        :param normalize: True to return normalized f64 data, False to return i16 data, defaults to True
        :type normalize: bool, optional
        :param diagnoses_filter: Function with the signature:
                                    diagnoses_filter(diagnoses: List[str]) -> bool
                                 Return true to include the database row to the query
        :type diagnoses_filter: Function
        :param **filters: query conditions (values) for specific per-database columns (keys)
        :type **filters: dict
        :yield:
            - sampling rate : audio sampling rate in samples/second
            - data  : audio data, 1-D for 1-channel NSP (only A channel), or 2-D of shape
                    (Nsamples, 2) for 2-channel NSP
            - auxdata : (optional) requested auxdata of the data if auxdata_fields is specified
        :ytype: tuple(int, numpy.ndarray(int16)[, pandas.Series])

        Iterates over all the DataFrame columns, returning a tuple with the column name and the content as a Series.

        Yields

            labelobject

                The column names for the DataFrame being iterated over.
            contentSeries

                The column entries belonging to each label, as a Series.



        Valid values of `auxdata_fields` argument
        ---------------------------------

        * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
        (except for "DIAGNOSIS" and "#")
        * "DIAGNOSES" - A list containing all the original "DIAGNOSIS" associated with the subject
        * "NORM" - True if normal data, False if pathological data
        * "MDVP" - Short-hand notation to include all the MDVP parameter measurements: from "Fo" to "PER"

        Valid `filters` keyword arguments
        ---------------------------------

        * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
        (except for "DIAGNOSIS" and "#")
        * "DIAGNOSES" - A list containing all the original "DIAGNOSIS" associated with the subject
        * "NORM" - True if normal data, False if pathological data

        Valid `filters` keyword argument values
        ---------------------------------------

        * A scalar value
        * For numeric and date columns, 2-element sequence to define a range: [start, end)
        * For all other columns, a sequence of allowable values
        """

        df = self.get_files(
            type,
            auxdata_fields,
            include_cape_v,
            include_grbas,
            rating_stats,
            **filters,
        )

        aux_cols = df.columns[3:]

        for id, file, tstart, tend, *auxdata in df.itertuples():
            framerate, x = self._read_file(file, tstart, tend, normalize)

            if bool(auxdata):
                yield id, framerate, x, pd.Series(
                    list(auxdata), index=aux_cols, name=id
                )
            else:
                yield id, framerate, x

    def read_data(self, id, type=None, normalize=True, padding=None):
        if not type:
            type = self.default_type

        if type != "all":
            tstart, tend = self._audio_timing.loc[id, type]

        else:
            tstart = tend = None

        return self._read_file(
            path.join(self._audio_dir, self._audio_files[id]),
            tstart,
            tend,
            normalize,
            padding,
        )

    def _read_file(self, file, tstart=None, tend=None, normalize=True, padding=None):
        if not padding:
            padding = self.default_padding

        if padding:
            id = re.match(r"(\D+\d+)", path.basename(file))[1].upper()

            ts = self._audio_timing.loc[id].sort_values()
            type = ts[ts == tstart].index[0][0]
            i = np.where(ts.index.get_loc(type))[0]

            tstart -= padding
            tend += padding

            talt = ts.iloc[i[0] - 1] if i[0] > 0 else 0.0
            if tstart < talt:
                tstart = talt

            if i[1] + 1 < len(ts):
                talt = ts.iloc[i[1] + 1]
                if tend > talt:
                    tend = talt

        with wave.open(file) as wobj:
            nchannels, sampwidth, framerate, nframes, *_ = wobj.getparams()
            dtype = f"<i{sampwidth}" if sampwidth > 1 else "u1"
            n0 = round(framerate * tstart) if tstart else 0
            n1 = round(framerate * tend) if tend else nframes
            b = wobj.readframes(n1)

        x = np.frombuffer(b, dtype, offset=n0 * nchannels * sampwidth)

        if nchannels > 1:
            x = x.reshape(-1, nchannels)

        if normalize:
            x = x / 2.0 ** (sampwidth * 8 - 1 if sampwidth > 1 else 8)

        return framerate, x

    def __getitem__(self, key):
        return self.read_data(key)
