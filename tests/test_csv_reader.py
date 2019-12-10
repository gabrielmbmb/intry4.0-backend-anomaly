import os
import pandas as pd
from unittest import TestCase
from blackbox.utils.csv import CSVReader


class TestCsvReader(TestCase):
    CSV_WITH_HEADER = "./tests/train_data.csv"
    CSV_WITHOUT_HEADER = "./tests/train_data_no_header.csv"
    DF_COLS = 4
    DF_ROWS = 984

    def setUp(self) -> None:
        self.reader = CSVReader(self.CSV_WITH_HEADER)
        self.reader_no_header = CSVReader(self.CSV_WITHOUT_HEADER)

    def test_read_csv_with_header(self):
        """Tests if a CSV file is read correctly"""
        df = self.reader.get_df()
        self.assertEqual(",", self.reader.sep)
        self.assertEqual(True, self.reader.has_header)
        self.assertEqual(self.DF_ROWS, df.shape[0])
        self.assertEqual(self.DF_COLS, df.shape[1])

    def test_read_csv_without_header(self):
        """Tests if a CSV file with no header is read correctly"""
        df = self.reader_no_header.get_df()
        self.assertEqual(",", self.reader_no_header.sep)
        self.assertEqual(False, self.reader_no_header.has_header)
        self.assertEqual(self.DF_ROWS, df.shape[0])
        self.assertEqual(self.DF_COLS, df.shape[1])

    def test_append_to_csv(self):
        """Tests the function to append DataFrame to an existing CSV file"""
        path = "test.csv"
        reader = CSVReader(path)
        df1 = pd.DataFrame([[True, False, True, False]], columns=["a", "b", "c", "d"])
        df2 = pd.DataFrame([[True, True, False]], columns=["a", "c", "d"])
        df_appended = df1.append(df2, sort=False, ignore_index=True)

        # if file does not exist
        reader.append_to_csv(df1)
        df = pd.read_csv(path)
        pd.testing.assert_frame_equal(df, df1)

        # if file does exist
        reader.append_to_csv(df2)
        df = pd.read_csv(path)
        pd.testing.assert_frame_equal(df, df_appended)

        os.remove(path)
