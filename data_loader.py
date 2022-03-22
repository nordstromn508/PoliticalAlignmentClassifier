"""
data_loader.py
    data loading module

    @author: Nicholas Nordstrom
"""
import nltk
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

bad_cols = ['Score', 'Id', 'Subreddit', 'Num of Comments', 'Date Created']
bad_seg = ['www.', '.com', '.org', 'i.', 'v.']
preprocess_cols = ['Text', 'Title']

stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()


def _parse_site_(row):
    """
    Given a row from a pandas dataframe, parse main website domain from URL column which contains a single URL
    :param row: a single row of a data frame
    :return:main site extracted from url field of row
    """
    site = row['URL'].split('/')[2]

    for seg in bad_seg:
        site = site.replace(seg, '')

    return site.replace('.', '')


def _tokenize_(row):
    """
    tokenize a row from a dataframe
    :param row: row from a dataframe
    :return: tokenized data frame row
    """
    for col in preprocess_cols:
        row['Tokenized ' + col] = [word for word in nltk.word_tokenize(row[col]) if word.isalnum()]
    return row


def _lemmatize_(row):
    """
    lemmatize a tokenized row from a pandas dataframe
    :param row: a single, tokenized row from a pandas dataframe
    :return: lemmatized, tokenized, row of the pandas dataframe
    """
    for col in preprocess_cols:
        row['Tokenized ' + col] = [lemmatizer.lemmatize(token.lower()) for token in row['Tokenized ' + col] if token not in stopwords]
    return row


def _normalize_dataframe_(df: pd.DataFrame):
    """
    Normalizes Dataset DataFrame
    :param df: raw, read DataFrame from poorly formatted Excel file
    :return: Normalized DataFrame Object
    """
    # Dimensionality Reduction
    df = df.drop(bad_cols, axis=1)

    # Tokenization
    df['Text'] = df['Text'].fillna(' ')
    df = df.apply(lambda row: _tokenize_(row), axis=1)

    # lemmatize
    df = df.apply(lambda row: _lemmatize_(row), axis=1)

    # Feature extraction: Domain website
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
    Reads a single csv file into memory
    :param ref: relative or full path to csv file to read in
    :return: pandas DataFrame read into memory from a csv file
    """
    return pd.read_csv(ref)


def read_excel(ref: str):
    """
    reads a single excel file into memory
    :param ref: relative or full path to excel file to read in
    :return: pandas dataframe read into memory from excel file
    """
    return pd.read_excel(ref)


def normalize_read_csv(ref: str):
    """
    chaining normalize and read_csv functions
    :param ref: reference to file to read
    :return:read, normalized dataframe
    """
    return _normalize_dataframe_(read_csv(ref))
