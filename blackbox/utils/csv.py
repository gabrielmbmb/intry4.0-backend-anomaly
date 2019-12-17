import os
import csv
import pandas as pd


class CSVReader:
    """
    CSV file reader able to detect the separator and if the file has header or not.

    Args:
        path (str): path of the file.

    Todo:
        * Autodetect if the file has index column (the index is usually a date or 
            succession of numbers, try to detect this?)
    """

    def __init__(self, path):
        self.path = path
        self.sep = None
        self.has_header = None
        if os.path.exists(self.path):
            self.get_csv_structure()

    def get_csv_structure(self) -> None:
        """Detects the separator and the header of a CSV file."""
        with open(self.path, newline="") as f:
            sniffer = csv.Sniffer()
            self.sep = sniffer.sniff(f.read(2048)).delimiter  # get file separator
            f.seek(0)
            self.has_header = sniffer.has_header(
                f.read(2048)
            )  # check if CSV has header
            f.seek(0)

    def get_df(self) -> pd.DataFrame:
        """
        Reads the CSV with the parameters detected.

        Returns:
            pandas.DataFrame: DataFrame with the data read.
        """
        if self.has_header:
            df = pd.read_csv(self.path, sep=self.sep, index_col=0)
        else:
            df = pd.read_csv(self.path, sep=self.sep, header=None, index_col=0)

        return df

    def append_to_csv(self, to_append) -> None:
        """
        Appends a DataFrame at the end of the CSV file. If the CSV file doesn't exist, 
        then it will be created.

        Args:
            to_append (pandas.DataFrame): dataframe to be appended or to be written in 
                the new CSV file.

        Returns:

        """
        if os.path.exists(self.path):
            df = pd.read_csv(self.path)
            df = df.append(to_append, sort=False, ignore_index=True)
            df.to_csv(self.path, index=False)
        else:
            to_append.to_csv(self.path, index=False)
