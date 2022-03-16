"""
data_loader.py
    data loading module

    @author: Nicholas Nordstrom
"""

import pandas as pd


def _parse_site_(row):
    """
    Given a row from a pandas dataframe, parse main website domain from URL column which contains a single URL
    :param row: a single row of a data frame
    :return:main site extracted from url field of row
    """
    site = row['URL'].split('/')[2]

    bad_seg = ['www.', '.com', '.org', 'i.', 'v.']
    for seg in bad_seg:
        site = site.replace(seg, '')

    return site.replace('.', '')


def _normalize_dataframe_(df: pd.DataFrame):
    """
    Normalizes Dataset DataFrame
    :param df: raw, read DataFrame from poorly formatted Excel file
    :return: Normalized DataFrame Object
    """
    bad_cols = ['Score', 'Id', 'Subreddit', 'Num of Comments', 'Date Created']
    df = df.drop(bad_cols, axis=1)
    df['Site'] = df.apply(lambda row: _parse_site_(row), axis=1)
    return df


def export_dataframe(df: pd.DataFrame, ref: str):
    """
    Exports a DataFrame to an Excel file
    :param df: DataFrame to save to Excel file
    :param ref: relative or full path location to save Excel file to
    :return: None
    """
    df.to_excel(ref, index=False, header=True)


def export_list_to_text(lines, name: str):
    """
    exports a list of items (string) to a file
    :param lines: list of lines to write to file
    :param name: name of file to write to
    :return: error code, 0 if no error
    """
    with open(name, 'w') as f:
        for item in lines:
            f.write("%s\n" % item)
    return 0


def read_csv(ref: str):
    """
    Reads a single Excel file into memory
    :param ref: relative or full path to Excel file to read in
    :return: pandas DataFrame read into memory from an Excel file
    """
    return pd.read_csv(ref)


def normalize_read_csv(ref: str):
    """
    chaining normalize and read_csv functions
    :param ref: reference to file to read
    :return:read, normalized dataframe
    """
    return _normalize_dataframe_(read_csv(ref))
