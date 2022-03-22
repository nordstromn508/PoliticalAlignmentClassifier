"""
main.py
    main thread of execution

    @author Nicholas Nordstrom
"""

import data_loader

INPUT_FILE = "Reddit_posts.csv"
PROCESSED_FILE = "data.xlsx"


def main():
    # df = data_loader.normalize_read_csv(INPUT_FILE)
    df = data_loader.read_excel(PROCESSED_FILE)
    print(df.columns)
    print(df)


if __name__ == "__main__":
    main()
