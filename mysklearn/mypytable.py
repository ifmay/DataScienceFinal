"""
Programmer: Drew Fitzpatrick
Class: CPSC 322, Fall 2024
Programming Assignment #7
11/7/2024

Description: This program implements methods of MyPyTable, 
            including an inner and outer join.
"""

import copy
import csv
from tabulate import tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col = []
        if isinstance(col_identifier, int):
            if col_identifier < 0 or col_identifier >= len(self.column_names):
                raise ValueError('Invalid column index')
            col_index = col_identifier
        else:
            col_index = self.column_names.index(col_identifier)
        if not include_missing_values:
            self.remove_rows_with_missing_values()
        for row in self.data:
            col.append(row[col_index])
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for i, value in enumerate((row)):
                try:
                    row[i] = float(value)
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        for index in sorted(row_indexes_to_drop, reverse=True):
            self.data.pop(index)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """

        with open(filename, newline='', encoding='utf8') as infile:
            reader = csv.reader(infile)
            for row in reader:
                self.data.append(row)
        self.column_names = self.data.pop(0)
        return self

    def load_txt_file(self, filename):
        """Load column names and data from a .txt file.

        Args:
            filename(str): relative path for the .txt file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            First row of .txt file is assumed to be the header.
            Calls convert_to_numeric() after load
        """

        with open(filename, 'r', newline='', encoding='utf8') as infile:
            reader = infile.readlines()
            self.column_names = reader[0].strip().split(',')
            for row in reader[1:]:
                self.data.append(row.strip().split(','))
        return self


    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, mode='w', newline='', encoding='utf8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def save_to_txt_file(self, filename):
        """Save column names and data to a .txt file.

        Args:
            filename(str): relative path for the .txt file to save the contents to.
        """
        with open(filename, 'w', encoding='utf8') as outfile:
            outfile.write(','.join(self.column_names) + '\n')
            for row in self.data:
                str_row = [str(element) for element in row]
                outfile.write(','.join(str_row) + '\n')

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        unique_vals = {}
        duplicates = []
        key_rows = []
        for key in key_column_names:
            key_rows.append(self.column_names.index(key))
        for i, row in enumerate(self.data):
            key = tuple(row[index] for index in key_rows)
            if key in unique_vals:
                duplicates.append(i)
            else:
                unique_vals[key] = i
        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        self.data = [row for row in self.data if "" not in row]

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        col = self.get_column(col_name)
        sum_col, count_not_na, avg = 0, 0, 0
        # compute average
        for value in col:
            if value != "NA":
                sum_col += value
                count_not_na += 1
        avg = sum_col / count_not_na
        # replace values
        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        min_col, max_col, mid, avg, median = 10000000, -10000000, 0, 0, 0
        sum_col, count = 0, 0
        attribute_names = ['attribute', 'min', 'max', 'mid', 'avg', 'median']
        stats_data = []
        stats_table = MyPyTable()

        # create a list of stats for each name in col_names
        for column in col_names:
            # check for empty list
            if len(self.get_column(column, include_missing_values=False)) == 0:
                return MyPyTable()
            for val in self.get_column(column, include_missing_values=False):
                sum_col += val
                count += 1
                if val < min_col:
                    min_col = val
                if val > max_col:
                    max_col = val

            # find median
            sorted_col = sorted(self.get_column(column, include_missing_values=False))
            mid_val = len(sorted_col)//2
            if len(sorted_col) % 2 ==0:
                median = (sorted_col[mid_val - 1] + sorted_col[mid_val]) /2
            else:
                median = sorted_col[mid_val]

            mid = (min_col + max_col) / 2
            avg = sum_col / count
            # add attributes
            stats_data.append([column, min_col, max_col, mid, avg, median])
            # reset values
            min_col, max_col, sum_col, count = 1000000, -1000000, 0, 0

        stats_table.column_names = attribute_names
        stats_table.data = stats_data
        return stats_table

    def extract_key_from_row(self, row, header, key_names):
        """Return the instance of key_name in the given row.
        
        Args:
            row (list of obj): the row to extract the key from
            header (list of str): attribute names for the given row
            key_names (list of str): name of the key to be extracted
        
        Returns:
            list of obj: the objects at the indexes of the given keys
        """
        keys = []
        for name in key_names:
            index = header.index(name)
            keys.append(row[index])
        return keys

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        joined_table = MyPyTable()
        joined_table.column_names = self.column_names + [col for col in other_table.column_names if col not in key_column_names]
        for row1 in self.data:
            for row2 in other_table.data:
                self_match_check = self.extract_key_from_row(row1, self.column_names, key_column_names)
                other_match_check = other_table.extract_key_from_row(row2, other_table.column_names, key_column_names)
                if self_match_check == other_match_check:
                    joined_row = row1 + [row2[other_table.column_names.index(col)] for col in other_table.column_names if col not in key_column_names]
                    joined_table.data.append(joined_row)

        return joined_table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        joined_table = MyPyTable()
        joined_table.column_names = self.column_names + [col for col in other_table.column_names if col not in key_column_names]
        matched_rows = []
        # process self.data rows
        for row1 in self.data:
            match_found = False
            for row2 in other_table.data:
                self_match_check = self.extract_key_from_row(row1, self.column_names, key_column_names)
                other_match_check = other_table.extract_key_from_row(row2, other_table.column_names, key_column_names)
                if self_match_check == other_match_check:
                    joined_row = row1 + [row2[i] for i in range(len(row2)) if other_table.column_names[i] not in key_column_names]
                    joined_table.data.append(joined_row)
                    matched_rows.append(self_match_check)
                    match_found = True
            # items in self with no matches in other_table
            if not match_found:
                joined_row = row1 + ['NA'] * (len(other_table.column_names) - len(key_column_names))
                joined_table.data.append(joined_row)
        # items in other_table with no matches in self
        for row2 in other_table.data:
            other_match_check = other_table.extract_key_from_row(row2, other_table.column_names, key_column_names)
            if other_match_check not in matched_rows:
                # format rows correctly
                joined_row = ['NA'] * len(self.column_names)
                for i, col in enumerate(other_table.column_names):
                    if col not in key_column_names:
                        joined_row.append(row2[i])
                    else:
                        joined_row[self.column_names.index(col)] = row2[i]
                joined_table.data.append(joined_row)

        return joined_table
