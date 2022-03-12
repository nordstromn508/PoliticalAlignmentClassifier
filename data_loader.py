"""
data_loader.py
    data loading module

    @author: Nicholas Nordstrom
"""

import pandas as pd


def read_excels():
    """
    Uses Pandas library to read in datasets from excel into Pandas DataFrame objects
    :return: list of pandas DataFrame objects
    """

    hinnews = pd.read_excel("Data/Fake News (hinnews.com) Questionable.xlsx", header=None)
    mzanzi = pd.read_excel("Data/Fake News (mzanzi stories) Fake.xlsx", header=None)
    sa_news = pd.read_excel("Data/Fake News (sa-news.com) Fake.xlsx", header=None)
    search67 = pd.read_excel("Data/Fake News (search67.com) Fake.xlsx", header=None)
    whatsappgroup = pd.read_excel("Data/Fake News (whatsappgroup.co.za) Fake.xlsx", header=None)

    return hinnews, mzanzi, sa_news, search67, whatsappgroup


def normalize_dataframe(df: pd.DataFrame):
    """
    Normalizes Dataset DataFrame
    :param df: raw, read DataFrame from poorly formatted Excel file
    :return: Normalized DataFrame Object
    """

    col_names = [x[2:-4].replace('\"', "") for x in df.iloc[1, :9:2]]
    df = df.drop([x for x in range(11)[::2]], axis=1)
    df.columns = col_names

    return df


def coalesce_dataframes(dfs: list(pd.DataFrame)):
    """
    concatenates a list of DataFrames into a single DataFrame
    :param dfs: list of DataFrames to concatenate together
    :return: DataFrame consisting of the data from each DataFrame passed in order
    """

    return pd.concat(dfs)


def export_dataframe(df: pd.DataFrame, ref: str):
    """
    Exports a DataFrame to an Excel file
    :param df: DataFrame to save to Excel file
    :param ref: relative or full path location to save Excel file to
    :return: None
    """

    df.to_excel(ref, index=False, header=True)


def read_normalize_coalesce_export():
    """
    Reads in dataset from excel files, normalizes columns and data, coalesces them into one DataFrame and exports
    them to an Excel File.
    :return: A DataFrame of the entire dataset, coalesced and normalized
    """

    # read excel files
    hinnews, mzanzi, sa_news, search67, whatsappgroup = read_excels()

    # normalize dataframes
    hinnews, mzanzi, sa_news, whatsappgroup = [normalize_dataframe(x) for x in [hinnews, mzanzi, sa_news, whatsappgroup]]

    # Search67 dataset needs extra preprocessing
    search67[[8, 9]] = search67[8].str.split(' ', expand=True, n=1)
    search67[[9, 10]] = search67[9].str.split('}', expand=True, n=1)
    search67 = normalize_dataframe(search67)
    search67.columns = mzanzi.columns
    search67['medium'] = search67['medium'].str.strip('\"')

    # coalesce
    dataset = coalesce_dataframes([hinnews, mzanzi, sa_news, search67, whatsappgroup])

    # export
    export_dataframe(dataset, 'Data/Dataset.xlsx')

    return dataset


def read_excel(ref: str):
    """
    Reads a single Excel file into memory
    :param ref: relative or full path to Excel file to read in
    :return: pandas DataFrame read into memory from an Excel file
    """

    return pd.read_excel(ref)
