import os
import pandas as pd
from glob import glob


class FileReader:
    """
    A class used to load a CSV, JSON, PICKLE or a batch of CSV files and get a Pandas DataFrame object.

    Args:
        path (str): path of the CSV file or the directory where a batch of CSV files is located. If a directory is
            specified but batch_name_pattern is not specified, then every CSV in the directory will be tried to be
            read.
        separator (str): character used to separate columns in CSV file. If the file is not a CSV this param will be
        ignored. Default is ','.
        batch_name_pattern (str): pattern that matches CSVs stored in directory specified in path argument. If this
            argument is specified, then the path argument has to be the path of the directory storing the CSV files.
            Default to None.
        columns (list): list of strings that are the names of the columns in CSV file. This argument can be used
        when the CSV file doesn't contain the columns names. Default to None.
        index_column (str): name of the index column. Default to None.
        batch_file_row (bool): indicates if the CSV file contained in the batch has measures of a specific date and
            average function has to be applied in order to get a single row in the final DataFrame for each file
            contained in the batch. If this option is True, the CSV must be named with a date and path argument has
            to be a directory. Default to False.
        apply_func_col (dict): dictionary containing as keys the names of the columns and as values a function to
            apply to a column, e.g: {"col_1": np.sqrt, "col_2": lambda x: x + 1}. Default to None.
        new_col (dict): dictionary containing as keys the names of the new columns and as values.
        verbose (bool): set verbosity mode. Default to False.

        Todo:
            * define new_col argument structure
            * implement generate_new_cols method
    """
    def __init__(self, path, separator=',', batch_name_pattern=None, columns=None, index_column=None,
                 batch_file_row=False, apply_func_col=None, new_col=None, verbose=False) -> None:
        self.path = path
        self.separator = separator
        self.batch_name_pattern = batch_name_pattern
        self.columns = columns
        self.index_column = index_column
        self.batch_file_row = batch_file_row
        self.apply_func_col = apply_func_col
        self.new_col = new_col
        self.verbose = verbose
        self.df = None

    def get_df(self) -> pd.DataFrame:
        """
        Returns the Pandas DataFrame previously created with read_csv or read_batch methods.

        Returns:
            DataFrame with data from the file or the batch of CSV files.
        """
        if self.df is None:
            if os.path.isfile(self.path):
                if self.verbose:
                    print('Dataset is not loaded. Loading data from one single file: {}'.format(self.path))

                filename, extension = os.path.splitext(self.path)
                if extension == '.csv':
                    self.df = self.read_csv()
                elif extension == '.json':
                    self.df = self.read_json()
                elif extension == '.pkl':
                    self.df = self.read_pickle()
            elif os.path.isdir(self.path):
                if self.verbose:
                    print('Dataset is not loaded. Loading data from directory: {}'.format(self.path))

                files_names = self.get_files_in_path()
                self.df = self.read_batch(files_names)

            self.df = self.apply_options_df()

        return self.df

    def read_csv(self) -> pd.DataFrame:
        """
        Reads the CSV file specified in path param in class constructor with separator also specified in class
        constructor.

        Returns:
            DataFrame with data from the CSV.
        """
        return pd.read_csv(self.path, sep=self.separator)

    def read_json(self):
        """
        Reads the JSON file specified in path param in class constructor.

        Returns:
            DataFrame with data from the JSON.
        """
        return pd.read_json(self.path, orient='records')

    def read_pickle(self):
        """
        Reads the PICKLE file specified in path param in class constructor.

        Returns:
            DataFrame with data from the PICKLE.
        """
        return pd.read_pickle(path=self.path)

    def apply_options_df(self) -> pd.DataFrame:
        """
        Applies options (columns names, functions to apply to columns, ...) to a DataFrame loaded from a file.

        Returns:
            DataFrame with options applied.
        """
        df = self.df

        if self.columns:
            df.columns = self.columns

        if self.apply_func_col:
            df = self.apply_functions_cols(df)

        if self.index_column:
            df.set_index(self.index_column, inplace=True)

        return df

    def read_batch_csv(self, csv_files_names) -> pd.DataFrame:
        """
        This function loads several CSVs files stored in path param specified in class constructor.

        Args:
            csv_files_names (list): A list containing the names of CSV files to read.

        Returns:
            DataFrame with data from the batch of CSVs stored in path class attribute.
        """
        raise NotImplementedError('read_batch function is not implemented yet!')

    def get_files_in_path(self) -> list:
        """
        Gets the name of files stored in path attribute class. If batch_name_pattern is defined then the function will
        get files stored in path that matches the pattern.

        Returns:
            list of files stored in path.
        """
        pattern = self.path + '/'
        if self.batch_name_pattern is None:
            pattern += '*'
        else:
            pattern += self.batch_name_pattern

        return sorted(glob(pattern))

    def apply_functions_cols(self, df) -> pd.DataFrame:
        """
        Iterates over the apply_func_col dictionary specified in class constructor and applies the function to the
        column.

        Args:
            df (Pandas DataFrame): DataFrame where the function has to be applied.

        Returns:
            DataFrame with functions applied in columns.
        """
        for col, fn in self.apply_func_col.items():
            df[col] = df[col].apply(fn)

        return df

    def save_df(self, format='csv', path='./dataframe.csv', columns=None) -> None:
        """
        Saves the DataFrame in a file.

        Args:
            format (str): indicates format of file. Possible formats: csv, json, pickle. Default to 'csv'.
            path (str): path where file will be saved. Default to './dataframe.csv'
            columns (list): list of columns that will be saved. Default to all columns.
        """
        if self.verbose:
            print('Saving DataFrame with format {}'.format(format))

        if columns is None:
            columns = self.df.columns

        if format == 'csv':
            self.df.to_csv(path_or_buf=path, columns=columns, header=True)
        elif format == 'json':
            self.df.to_json(path_or_buf=path, orient='records', columns=columns)
        elif format == 'pickle':
            self.df.to_pickle(path=path)

    def generate_new_cols(self):
        raise NotImplementedError('generate_new_cols is not implemented!')

    def concat_batch(self) -> None:
        raise NotImplementedError('concat_batch function is not implemented yet!')