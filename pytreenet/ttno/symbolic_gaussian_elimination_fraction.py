from copy import deepcopy
import numpy as np
from fractions import Fraction

def row_swap(matrix,Op_l, row1, row2):
    _row_swap(matrix, row1, row2)
    _col_swap(Op_l, row1, row2)
    return matrix, Op_l

def col_swap(matrix, Op_r, col1, col2):
    _col_swap(matrix, col1, col2)
    _row_swap(Op_r, col1, col2)
    return matrix, Op_r

def _row_swap(matrix, row1, row2):
    matrix[row1], matrix[row2] = matrix[row2], matrix[row1]

def _col_swap(matrix, col1, col2):
    for row in matrix:
        row[col1], row[col2] = row[col2], row[col1]

def row_scale(matrix, Op_l, row, factor):
    _row_scale(matrix, row, factor)
    _col_scale(Op_l, row, 1/factor)
    return matrix, Op_l

def col_scale(matrix, Op_r, col, factor):
    _col_scale(matrix, col, factor)
    _row_scale(Op_r, col, 1/factor)
    return matrix, Op_r

def _row_scale(matrix, row, factor):
    for i in range(len(matrix[row])):
        entry = matrix[row][i]
        if isinstance(entry, tuple):
            coeff, var = entry
            new_coeff = coeff * factor
            matrix[row][i] = (new_coeff, var)
            
        else:

            matrix[row][i] *= factor

def _col_scale(matrix, col, factor):
    for row in matrix:
        entry = row[col]
        if isinstance(entry, tuple):
            coeff, var = entry
            new_coeff = coeff * factor
            row[col] = (new_coeff, var)
        else:
            row[col] *= factor

def row_add(matrix, Op_l, target_row, source_row, factor):
    if isinstance(factor, Fraction):
        is_zero, success = _row_add(matrix, target_row, source_row, factor)
        if success:
            _col_add_float(Op_l, source_row, target_row, -factor)
        return matrix, Op_l, is_zero
    else:
        return matrix, Op_l, False

def col_add(matrix, Op_r, target_col, source_col, factor):
    if isinstance(factor, Fraction):
        is_zero, success = _col_add(matrix, target_col, source_col, factor)
        if success:
            _row_add_float(Op_r, source_col, target_col, -factor)
        return matrix, Op_r, is_zero
    else:
        return matrix, Op_r, False

def _row_add(matrix, target_row, source_row, factor):
    new_row = deepcopy(matrix[target_row])
    for i in range(len(matrix[target_row])):
        source_entry = matrix[source_row][i]
        target_entry = matrix[target_row][i]
        
        if isinstance(source_entry, tuple):
            source_coeff, source_var = source_entry
            if isinstance(target_entry, tuple):
                target_coeff, target_var = target_entry
                if target_var == source_var:
                    # Combine if variables match
                    new_coeff = target_coeff + factor * source_coeff
                    if new_coeff == 0:
                        new_row[i] = 0
                    else:
                        new_row[i] = (new_coeff, source_var)
                else:
                    return ( False, False)

            else:
                if target_entry == 0:
                    new_row[i] = ( factor * source_coeff, source_var )
                else:
                    return  ( False , False)
        else:
            if not isinstance(target_entry,  tuple):
                new_row[i] += factor * source_entry
            elif source_entry != 0:
                return ( False, False ) 
    
    
    matrix[target_row] = new_row
    return (not any(new_row) , True)

def _row_add_float(matrix, target_row, source_row, factor):
    for i in range(len(matrix[target_row])):
        source_entry = matrix[source_row][i]
        matrix[target_row][i] += factor * source_entry
    
def _col_add(matrix, target_col, source_col, factor):
    new_col = [row[target_col] for row in matrix]    
    for i,row in enumerate(matrix):
        source_entry = row[source_col]
        target_entry = row[target_col]
        
        if isinstance(source_entry, tuple):
            source_coeff, source_var = source_entry

            if isinstance(target_entry, tuple):
                target_coeff, target_var = target_entry
                if target_var == source_var:
                    # Combine if variables match
                    new_coeff = target_coeff + factor * source_coeff
                    if new_coeff == 0:
                        new_col[i] = 0
                    else:
                        new_col[i] = (new_coeff, source_var)
                else:
                    return ( False, False)
            else:
                if target_entry == 0:
                    new_col[i] = ( factor * source_coeff, source_var )
                else:
                    return ( False, False)
        else:
            if not isinstance(target_entry, tuple):
                new_col[i] += factor * source_entry
            elif source_entry != 0:
                return ( False, False)
    for i, row in enumerate(matrix):
        row[target_col] = new_col[i]
    
    return (not any(new_col) , True)

def _col_add_float(matrix, target_col, source_col, factor):
    for row in matrix:
        source_entry = row[source_col]
        row[target_col] += factor * source_entry
           
def print_matrix(matrix):
    """Helper function to print the matrix nicely."""
    for row in matrix:
        print("\t".join([f"{entry[0]}*{entry[1]}" if isinstance(entry, tuple) else str(entry)
                         for entry in row]))
    print()

def gaussian_elimination(matrix):
    n_rows, n_rows_old = len(matrix), 0
    n_cols, n_cols_old = len(matrix[0]), 0


    Op_l = [[Fraction(1) if i == j else Fraction(0) for j in range(n_rows)] for i in range(n_rows)]
    Op_r = [[Fraction(1) if i == j else Fraction(0) for j in range(n_cols)] for i in range(n_cols)]
    
    deparallelize_rows(Op_l, matrix)
    deparallelize_cols(Op_r, matrix)


    while(n_rows != n_rows_old or n_cols != n_cols_old):
        n_rows_old, n_cols_old = n_rows, n_cols
        row_elimination(Op_l, matrix)
        column_elimination(Op_r, matrix)
        n_rows, n_cols = len(matrix), len(matrix[0])
    return  Op_l, matrix,  Op_r                

def deparallelize_rows(Op_l, matrix):
    zero_rows = []
    for i in range(len(matrix)):
        if i in zero_rows:
            continue
        for j in range(i+1, len(matrix)):
            if j in zero_rows:
                continue
            mult = are_parallel_row(matrix[i], matrix[j])
            
            if mult != 0:    
                _col_add_float(Op_l, i, j, mult)
                zero_rows.append(j)
            
    zero_rows = sorted(zero_rows, reverse=True)
    for row_0 in zero_rows:
        del matrix[row_0]
        for row in Op_l:
            del row[row_0]
    return Op_l, matrix

def are_parallel_row(row1, row2):
    """
    Check if two rows are parallel (Fraction multiples of each other).
    Handles fractions and tuples (coefficient, variable).
    """
    ratio = Fraction(0)  # Initialize the scalar ratio to None

    for a, b in zip(row1, row2):
        # Extract the value and variable for each element
        a_coeff, a_var = (a, '') if isinstance(a, Fraction) else a
        b_coeff, b_var = (b, '') if isinstance(b, Fraction) else b

        # If variables don't match, rows are not parallel
        if a_var != b_var:
            return 0

        # If both are zero, continue
        if a_coeff == 0 and b_coeff == 0:
            continue

        # If one is zero and the other isn't, rows are not parallel
        if a_coeff == 0 or b_coeff == 0:
            return 0

        # Calculate the ratio for this pair of elements
        current_ratio = b_coeff / a_coeff

        # Check if the ratio is consistent
        if ratio == 0:
            ratio = current_ratio  # Set the initial ratio
        elif current_ratio != ratio:
            return 0  # Ratios don't match, not parallel

    return ratio  # All elements are consistent

def deparallelize_cols(Op_r, matrix):
    zero_cols = []
    for i in range(len(matrix[0])):
        if i in zero_cols:
            continue
        for j in range(i+1, len(matrix[0])):
            if j in zero_cols:
                continue
            mult = are_parallel_col(matrix, i, j)
            if mult != 0:
                
                _row_add_float(Op_r, i, j, mult)
                zero_cols.append(j)
            
    zero_cols = sorted(zero_cols, reverse=True)
    for col_0 in zero_cols:
        for row in matrix:
            del row[col_0]
        del Op_r[col_0]
    return Op_r, matrix

def are_parallel_col(matrix, col1, col2):
    """
    Check if two columns are parallel (scalar multiples of each other).
    Handles scalars and tuples (coefficient, variable).
    """
    ratio = Fraction(0)  # Initialize the scalar ratio to None

    for row in matrix:
        # Extract the value and variable for each element
        a_coeff, a_var = (row[col1], '') if isinstance(row[col1], Fraction) else row[col1]
        b_coeff, b_var = (row[col2], '') if isinstance(row[col2], Fraction) else row[col2]

        # If variables don't match, columns are not parallel
        if a_var != b_var:
            return 0

        # If both are zero, continue
        if a_coeff == 0 and b_coeff == 0:
            continue

        # If one is zero and the other isn't, columns are not parallel
        if a_coeff == 0 or b_coeff == 0:
            return 0

        # Calculate the ratio for this pair of elements
        current_ratio = b_coeff / a_coeff

        # Check if the ratio is consistent
        if ratio == 0:
            ratio = current_ratio  # Set the initial ratio
        elif current_ratio != ratio:  # Allow for floating-point tolerance
            return 0  # Ratios don't match, not parallel

    return ratio  # All elements are consistent

def row_elimination(Op_l, matrix):
    i = 0  
    while i < min(len(matrix) , len(matrix[0])):


        if matrix[i][i] == 0:
            for j in range(i + 1, len(matrix)):
                if matrix[j][i] != 0:
                    row_swap(matrix, Op_l, i, j)
                    break

        pivot = matrix[i][i]
        if pivot == 0:
            i += 1
            continue

        #if isinstance(pivot, int) and pivot != 0:
        #    row_scale(matrix, Op_l, i, 1 / pivot)
            
        
        j = 0
        zero_rows = []
        while j < len(matrix):
            if j != i and matrix[j][i] != 0:
                is_zero = False
                   
                if isinstance(pivot, tuple) and isinstance(matrix[j][i], tuple) and pivot[1] == matrix[j][i][1]:
                    _,_, is_zero = row_add(matrix, Op_l, j, i, -matrix[j][i][0] / pivot[0])

                elif not isinstance(pivot, tuple) and not isinstance(matrix[j][i], tuple) :
                    _,_, is_zero = row_add(matrix, Op_l, j, i, -matrix[j][i] / pivot)
                if is_zero:
                    zero_rows.append(j)
            j += 1

        
        i += 1
        
        zero_rows = sorted(zero_rows, reverse=True)
        for row_0 in zero_rows:
            del matrix[row_0]
            for row in Op_l:
                del row[row_0]

def row_elimination_brute(Op_l, matrix):
    zero_rows = []
    for j in range(len(matrix[0])):
        for i in range(len(matrix)):
            pivot = matrix[i][j]
            if pivot == 0:
                continue

            
            for k in range(i+1,len(matrix)):
                if matrix[k][j] != 0:
                    is_zero = False
                    if isinstance(pivot,  (int,float)) and isinstance(matrix[k][j],  (int,float)):
                        _,_, is_zero = row_add(matrix, Op_l, k, j, -matrix[k][j] / pivot)
                        
                    elif isinstance(pivot, tuple) and isinstance(matrix[k][j], tuple) and pivot[1] == matrix[k][j][1]:
                        _,_, is_zero = row_add(matrix, Op_l, k, j, -matrix[k][j][0] / pivot[0])
                    if is_zero:
                        zero_rows.append(k)

    zero_rows = sorted(zero_rows, reverse=True)
    for row_0 in zero_rows:
        del matrix[row_0]
        for row in Op_l:
            del row[row_0]
              
def column_elimination(Op_r, matrix):
    j = 0
    while j < min(len(matrix) , len(matrix[0])):
        if matrix[j][j] == 0:
            for i in range(j + 1, len(matrix[0])):
                if matrix[j][i] != 0:
                    col_swap(matrix, Op_r, j, i)
                    break

        pivot = matrix[j][j]
        if pivot == 0:
            j += 1
            continue

        #if isinstance(pivot,  (int,float)) and pivot != 0:
        #    col_scale(matrix, Op_r, j, 1 / pivot)

        i = 0
        zero_cols = []
        while i < len(matrix[0]):
            if i != j and matrix[j][i] != 0:
                is_zero = False
                    
                if isinstance(pivot, tuple) and isinstance(matrix[j][i], tuple) and pivot[1] == matrix[j][i][1]:
                    _,_, is_zero = col_add(matrix, Op_r, i, j, -matrix[j][i][0] / pivot[0])
                elif not isinstance(pivot, tuple) and not isinstance(matrix[j][i], tuple) :
                    _,_, is_zero = col_add(matrix, Op_r, i, j, -matrix[j][i] / pivot)
                if is_zero:
                    zero_cols.append(i)
            i += 1
        j += 1

        zero_cols = sorted(zero_cols, reverse=True)
        for col_0 in zero_cols:

            for row in matrix:
                del row[col_0]
            del Op_r[col_0]
                
def column_elimination_brute(Op_r, matrix):
    zero_cols = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            pivot = matrix[i][j]
            if pivot == 0:
                continue
            
            for k in range(j+1,len(matrix[0])):
                if matrix[i][k] != 0:
                    is_zero = False
                    if isinstance(pivot,  (int,float)) and isinstance(matrix[i][k],  (int,float)):
                        _,_, is_zero = col_add(matrix, Op_r, k, j, -matrix[i][k] / pivot)
                    elif isinstance(pivot, tuple) and isinstance(matrix[i][k], tuple) and pivot[1] == matrix[i][k][1]:
                        _,_, is_zero = col_add(matrix, Op_r, k, j, -matrix[i][k][0] / pivot[0])
                    if is_zero:
                        zero_cols.append(k)
            
    zero_cols = sorted(zero_cols, reverse=True)
    for col_0 in zero_cols:

        for row in matrix:
            del row[col_0]
        del Op_r[col_0]
