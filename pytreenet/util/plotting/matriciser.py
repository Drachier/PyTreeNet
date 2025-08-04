"""
Implements a class that allows for easy plotting of matrix like data, that
is not yet in a format suitable for plotting.
"""

import numpy as np
from matplotlib.axes import Axes

class Matriciser:
    """
    Handles result data to be turned into a matrix that can be used for
    plotting.
    """

    def __init__(self, data: list[tuple[int, int, float]] | None = None):
        """
        Initialize the Matriciser with a list of tuples.

        Each tuple should contain two integers and a float, representing
        the row index, column index, and value respectively.
        
        Args:
            data (list[tuple[int, int, float]]): The data to be converted
                into a matrix. The first integer describes the values of 
                corresponding to rows, the second integer describes the
                values corresponding to columns, and the float is the value
                at that position in the matrix.
        """
        if data is None:
            data = []
        self._data = data
        self._row_elements = []
        self._col_elements = []

    def _row_and_col_init(self) -> bool:
        """
        Determines if the row and column elements have been initialized.
        """
        return bool(self._row_elements and self._col_elements)

    def add_data_point(self, row: int, col: int, value: float):
        """
        Add a data point to the Matriciser.

        Args:
            row (int): The row index of the data point.
            col (int): The column index of the data point.
            value (float): The value at the specified row and column.
        """
        self._data.append((row, col, value))
        if self._row_and_col_init():
            if row not in self._row_elements:
                self._row_elements.append(row)
            if col not in self._col_elements:
                self._col_elements.append(col)

    def get_row_elements(self) -> list[int]:
        """
        Get the unique row elements from the data.

        Returns:
            list[int]: A sorted list of unique row elements.
        """
        if not self._row_and_col_init():
            return sorted(set(row_val for row_val, _, _ in self._data))
        return self._row_elements

    def row_map(self) -> dict[int,int]:
        """
        A mapping from row values to their indices in the matrix.
        """
        return {val: idx for idx, val in enumerate(self.get_row_elements())}

    def get_col_elements(self) -> list[int]:
        """
        Get the unique column elements from the data.

        Returns:
            list[int]: A sorted list of unique column elements.
        """
        if not self._row_and_col_init():
            return sorted(set(col_val for _, col_val, _ in self._data))
        return self._col_elements

    def col_map(self) -> dict[int,int]:
        """
        A mapping from column values to their indices in the matrix.
        
        Returns:
            dict[int,int]: A dictionary mapping column values to their indices.
        """
        return {val: idx for idx, val in enumerate(self.get_col_elements())}

    def data_min(self) -> float:
        """
        Get the minimum value from the data.

        Returns:
            float: The minimum value in the data.
        """
        return min(value for _, _, value in self._data)

    def data_max(self) -> float:
        """
        Get the maximum value from the data.

        Returns:
            float: The maximum value in the data.
        """
        return max(value for _, _, value in self._data)

    def set_y_tick_labels(self, ax: Axes):
        """
        Set the y-tick labels of the given axes to the row elements.

        Args:
            ax (Axes): The matplotlib Axes object to set the y-tick labels on.
        """
        if not ax.images:
            errstr = "No image found in the axes. Please run `ax.imshow()` first."
            raise RuntimeError(errstr)
        row_map = self.row_map()
        y_ticks = [int(round(t)) for t in ax.get_yticks()]
        y_tick_labels = [str(row_map[t]) for t in y_ticks]
        ax.set_yticklabels(y_tick_labels)

    def set_x_tick_labels(self, ax: Axes):
        """
        Set the x-tick labels of the given axes to the column elements.

        Args:
            ax (Axes): The matplotlib Axes object to set the x-tick labels on.
        """
        if not ax.images:
            errstr = "No image found in the axes. Please run `ax.imshow()` first."
            raise RuntimeError(errstr)
        col_map = self.col_map()
        x_ticks = [int(round(t)) for t in ax.get_xticks()]
        x_tick_labels = [str(col_map[t]) for t in x_ticks]
        ax.set_xticklabels(x_tick_labels)

    def to_matrix(self) -> np.ndarray:
        """
        Convert the data into a matrix.

        Returns:
            np.ndarray: A 2D numpy array representing the matrix.
        """
        row_elements = self.get_row_elements()
        col_elements = self.get_col_elements()
        matrix = np.zeros((len(row_elements), len(col_elements)))
        for row_val, col_val, value in self._data:
            row_idx = row_elements.index(row_val)
            col_idx = col_elements.index(col_val)
            matrix[row_idx, col_idx] = value
        return matrix