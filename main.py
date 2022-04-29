"""
main.py
    main thread of execution

    @author Nicholas Nordstrom and Jason Zou
"""
import sys
from datetime import datetime

import data_loader
from model import random_forests

INPUT_FILE = "Reddit_posts.csv"
PROCESSED_FILE = "data.xlsx"
LOGGING = True


def main():
    # set up logging
    if LOGGING:
        cur_date, cur_time = str(datetime.now()).split(' ')
        cur_time = cur_time[:8].replace(':', '.')
        logfile_name = "main " + cur_date + ' ' + cur_time[:8].replace(':', '-') + '.txt'
        sys.stdout = open('logs/' + logfile_name, 'w')
        print('Log File: ' + logfile_name)

    # skip processing because its already been done and saved
    df, df_bi_gram_vec_title, df_bi_gram_vec_text, df_tri_gram_vec_title, df_tri_gram_vec_text, df_freq_vec_title, \
    df_freq_vec_text, df_bow_vec_title, df_bow_vec_text = data_loader.process_read_csv(INPUT_FILE)

    # load saved, processed file
    # df = data_loader.read_excel(PROCESSED_FILE)
    print('DF Columns: ', df.columns)
    print("DF Data:\n", df)

    print("df_bi_gram_vec_title Data:\n", df_bi_gram_vec_title)
    print("df_bi_gram_vec_text Data:\n", df_bi_gram_vec_text)
    print("df_tri_gram_vec_title Data:\n", df_tri_gram_vec_title)
    print("df_tri_gram_vec_text Data:\n", df_tri_gram_vec_text)
    print("df_freq_vec_title Data:\n", df_freq_vec_title)
    print("df_freq_vec_text Data:\n", df_freq_vec_text)
    print("df_bow_vec_title Data:\n", df_bow_vec_title)
    print("df_bow_vec_text Data:\n", df_bow_vec_text)

    y = df['Political Lean']
    df.pop('Political Lean')
    x = df['Title','URL','Text','Site']
    random_forests(x,y)


if __name__ == "__main__":
    main()
