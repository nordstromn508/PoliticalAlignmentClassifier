"""
main.py
    main thread of execution

    @author Nicholas Nordstrom
"""

import data_loader

DATA_FILE = "Reddit_posts.csv"


def main():
    df = data_loader.normalize_read_csv(DATA_FILE)
    print(df)


if __name__ == "__main__":
    main()
