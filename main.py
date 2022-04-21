"""
main.py
    main thread of execution

    @author Nicholas Nordstrom and Jason Zou
"""

import data_loader
from model import random_forests

INPUT_FILE = "Reddit_posts.csv"


def main():
    df = data_loader.normalize_read_csv(INPUT_FILE)
    print(df.columns)
    y = df['Political Lean']
    df.pop('Political Lean')
    x = df['Title','URL','Text','Site']
    random_forests(x,y)


if __name__ == "__main__":
    main()
