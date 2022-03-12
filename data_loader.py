"""
data_loader.py
    data loading module

    @author: Nicholas Nordstrom
"""

import pandas as pd


def __normalize_dataframe__(df: pd.DataFrame):
    """
    Normalizes Dataset DataFrame
    :param df: raw, read DataFrame from poorly formatted Excel file
    :return: Normalized DataFrame Object
    """

    col_names = [x[2:-4].replace('\"', "") for x in df.iloc[1, :9:2]]
    df = df.drop([x for x in range(11)[::2]], axis=1)
    df.columns = col_names

    return df


def export_dataframe(df: pd.DataFrame, ref: str):
    """
    Exports a DataFrame to an Excel file
    :param df: DataFrame to save to Excel file
    :param ref: relative or full path location to save Excel file to
    :return: None
    """

    df.to_excel(ref, index=False, header=True)


def read_csv(ref: str):
    """
    Reads a single Excel file into memory
    :param ref: relative or full path to Excel file to read in
    :return: pandas DataFrame read into memory from an Excel file
    """

    return pd.read_csv(ref)
