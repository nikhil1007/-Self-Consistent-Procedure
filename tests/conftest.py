"""
The PyTest Configuration code
"""
import pytest
from main import main


@pytest.fixture
def mol_h2o():
    """
    Fixture that creates a specific water molecule
    """

    atom = "8 0.000000000000  -0.143225816552   0.000000000000;" \
        + "1 1.638036840407   1.136548822547  -0.000000000000;" \
        + "1 -1.638036840407   1.136548822547  -0.000000000000"

    return ""

# def get_Tuv():
#     return main.Tuv
#
# def get_Huv():
#     return main.Huv

