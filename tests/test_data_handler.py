import pytest
import numpy as np
from SKA_assignment.data_handler import DataHandler


@pytest.fixture
def mock_data_handler(mocker):
    """Fixture to create a mock DataHandler instance."""
    # Mock the casacore.tables.table class
    mock_table = mocker.patch("casacore.tables.table")
    mock_table.return_value.getcol.side_effect = lambda col: {
        "TIME": np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
        "UVW": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        "DATA": np.array(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
            ]
        ),
    }[col]
    return DataHandler("mock_path")


def test_get_autocorr_filter(mock_data_handler):
    """Test the get_autocorr_filter method."""
    autocorr_filter = mock_data_handler.get_autocorr_filter()
    expected = np.array([True, False, True, False, True, False])
    assert np.array_equal(autocorr_filter, expected)


def test_get_visibilities(mock_data_handler):
    """Test the get_visibilities method."""
    visibilities = mock_data_handler.get_visibilities()
    expected = np.array([[3, 7], [11, 15], [19, 23], [3, 7], [11, 15], [19, 23]])
    assert np.array_equal(visibilities, expected)


def test_get_times(mock_data_handler):
    """Test the get_times method."""
    times = mock_data_handler.get_times()
    expected = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(times, expected)


def test_get_time_step(mock_data_handler):
    """Test the get_time_step method."""
    time_step = mock_data_handler.get_time_step()
    expected = 1.0
    assert time_step == expected
