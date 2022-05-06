"""
main.py
    main thread of execution

    @author Nicholas Nordstrom and Jason Zou
"""
import sys
from time import time, sleep
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
import data_loader
from model import random_forests, dense_dropout_nn

INPUT_FILE = "Reddit_posts.csv"
PROCESSED_FILE = "data.xlsx"
LOGGING = True


def main():
    start = time()
    start_logging = time()
    # set up logging
    if LOGGING:
        cur_date, cur_time = str(datetime.now()).split(' ')
        cur_time = cur_time[:8].replace(':', '.')
        logfile_name = "main " + cur_date + ' ' + cur_time[:8].replace(':', '-') + '.txt'
        sys.stdout = open('logs/' + logfile_name, 'w')
        print('Log File: ' + logfile_name)
        print('Logging Setup Took: {0:.4f} Seconds'.format(time() - start_logging))

    # skip processing because its already been done and saved
    start_processing = time()
    df = data_loader.process_read_csv(INPUT_FILE)
    # data_loader.export_dataframe(df, PROCESSED_FILE)
    print('Data Loading and Processing Took: {0:.4f} Seconds'.format(time() - start_processing))

    # start_reading = time()
    # load saved, processed file
    # df = data_loader.read_excel(PROCESSED_FILE)
    print('DF Columns: ', df.columns)
    print("DF Data:\n", df)
    print("DF Unique Sites:\n", df['Site'].value_counts())
    print("DF Unique Labels:\n", df['Political Lean'].value_counts())
    # print('DF Loading Took: {0:.4f} Seconds'.format(time() - start_reading))

    start_splitting = time()
    # Select columns to use as features
    # Columns: ['Title', 'Political Lean', 'URL', 'Text', 'Tokenized Text', 'Tokenized Title', 'Site',
    # 'Untokenized Title', 'Untokenized Text', 'bi_gram_vec_title_0', 'bi_gram_vec_text_0', 'tri_gram_vec_title_0',
    # 'tri_gram_vec_text_', 'freq_vec_title_', 'freq_vec_text_', 'bow_vec_title_', 'bow_vec_text_']
    X_mask = [col for col in df if col.startswith('bow_vec_title')]
    X_train, X_test, y_train, y_test = train_test_split(df[X_mask],
                                                        df['Political Lean'],
                                                        test_size=0.2, random_state=1)
    print('Data Splitting Took: {0:.4f} Seconds'.format(time() - start_splitting))

    start_model = time()
    # random_forests(x,y)
    dense_dropout_nn(X_train, y_train,X_test, y_test)
    print(df.info())
    print('Model Creation, Training and Testing Took: {0:.4f} Seconds'.format(time() - start_model))
    print('Total Time Elapsed: {0:.4f} Seconds'.format(time() - start))


if __name__ == "__main__":
    main()
