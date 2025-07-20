"""
This module implements unittests to test the utility functions of `abc_model`.
"""

import pytest

from pytreenet.operators.models.abc_model import (generate_chain_indices,
                                                  generate_t_topology_indices,
                                                  generate_2d_indices)


testdata = [(0,[]),
            (1, ["site0"]),
            (2, ["site0", "site1"]),
            (3, ["site0", "site1", "site2"]),
            (4, ["site0", "site1", "site2", "site3"]),
            (5, ["site0", "site1", "site2", "site3", "site4"])]
@pytest.mark.parametrize("num_sites, expected", testdata)
def test_generate_chain_indices(num_sites, expected):
    """
    Test the generation of chain indices for a given number of sites.
    """
    assert generate_chain_indices(num_sites) == expected

testdata = [(0, []),
            (1, ["qubit0"]),
            (2, ["qubit0", "qubit1"]),
            (3, ["qubit0", "qubit1", "qubit2"]),
            (4, ["qubit0", "qubit1", "qubit2", "qubit3"]),
            (5, ["qubit0", "qubit1", "qubit2", "qubit3", "qubit4"])]
@pytest.mark.parametrize("num_sites, expected", testdata)
def test_generate_chain_indices_nonstandardprefix(num_sites, expected):
    """
    Test the generation of chain indices for a given number of sites with
    non-standard prefix.
    """
    assert generate_chain_indices(num_sites, site_ids="qubit") == expected

def test_generate_chain_indices_invalid():
    """
    Test the generation of chain indices with invalid input.
    """
    with pytest.raises(ValueError):
        generate_chain_indices(-1)

testdata = [(0, ([], [], [])),
            (1, (["site0"], ["site1"], ["site2"])),
            (2, (["site0", "site1"], ["site2", "site3"], ["site4", "site5"])),
            (3, (["site0", "site1", "site2"], ["site3", "site4", "site5"],
                  ["site6", "site7", "site8"]))]

@pytest.mark.parametrize("chain_length, expected", testdata)
def test_generate_t_topology_indices(chain_length, expected):
    """
    Test the generation of T-topology indices for a given chain length.
    """
    assert generate_t_topology_indices(chain_length) == expected

testdata = [(0, ([], [], [])),
            (1, (["qubit0"], ["qubit1"], ["qubit2"])),
            (2, (["qubit0", "qubit1"], ["qubit2", "qubit3"],
                  ["qubit4", "qubit5"])),
            (3, (["qubit0", "qubit1", "qubit2"], ["qubit3", "qubit4", "qubit5"],
                  ["qubit6", "qubit7", "qubit8"]))]

@pytest.mark.parametrize("chain_length, expected", testdata)
def test_generate_t_topology_indices_nonstandardprefix(chain_length, expected):
    """
    Test the generation of T-topology indices for a given chain length with
    non-standard prefix.
    """
    assert generate_t_topology_indices(chain_length, site_ids="qubit") == expected

def test_generate_t_topology_indices_invalid():
    """
    Test the generation of T-topology indices with invalid input.
    """
    with pytest.raises(ValueError):
        generate_t_topology_indices(-1)

testdata = [(0, []),
            (1, [["site0"]]),
            (2, [["site0", "site1"],
                 ["site2", "site3"]]),
            (3, [["site0", "site1", "site2"],
                 ["site3", "site4", "site5"],
                 ["site6", "site7", "site8"]]),
            (4, [["site0", "site1", "site2", "site3"],
                 ["site4", "site5", "site6", "site7"],
                 ["site8", "site9", "site10", "site11"],
                 ["site12", "site13", "site14", "site15"]])]
@pytest.mark.parametrize("num_rows, expected", testdata)
def test_generate_2d_indices_square(num_rows, expected):
    """
    Test the generation of 2D indices for a given number of rows, i.e. a
    square grid.
    """
    assert generate_2d_indices(num_rows) == expected

testdata = [(1,2, [["site0", "site1"]]),
            (2,3, [["site0", "site1", "site2"],
                   ["site3", "site4", "site5"]]),
            (3,4, [["site0", "site1", "site2", "site3"],
                   ["site4", "site5", "site6", "site7"],
                   ["site8", "site9", "site10", "site11"]])]

@pytest.mark.parametrize("num_rows, num_cols, expected", testdata)
def test_generate_2d_indices_rectangle(num_rows, num_cols, expected):
    """
    Test the generation of 2D indices where the number of rows is smaller
    than the number of columns.
    """
    assert generate_2d_indices(num_rows, num_cols) == expected

testdata = [(2,1, [["site0"],
                   ["site1"]]),
            (3,2, [["site0", "site1"],
                   ["site2", "site3"],
                   ["site4", "site5"]]),
            (4,3, [["site0", "site1", "site2"],
                   ["site3", "site4", "site5"],
                   ["site6", "site7", "site8"],
                   ["site9", "site10", "site11"]])]

@pytest.mark.parametrize("num_rows, num_cols, expected", testdata)
def test_generate_2d_indices_rectangle_transposed(num_rows, num_cols, expected):
    """
    Test the generation of 2D indices where the number of rows is larger
    than the number of columns.
    """
    assert generate_2d_indices(num_rows, num_cols) == expected

def test_generate_2d_indices_invalid_rows():
    """
    Test the generation of 2D indices with invalid number of rows.
    """
    with pytest.raises(ValueError):
        generate_2d_indices(-1)

def test_generate_2d_indices_invalid_cols():
    """
    Test the generation of 2D indices with invalid number of columns.
    """
    with pytest.raises(ValueError):
        generate_2d_indices(2, -1)
