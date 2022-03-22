"""
main.py
    main thread of execution

    @author Nicholas Nordstrom and Jason Zou
"""

import data_loader

INPUT_FILE = "Reddit_posts.csv"


def main():
    df = data_loader.normalize_read_csv(INPUT_FILE)
    print(df.columns)
    print(df)


if __name__ == "__main__":
    main()
