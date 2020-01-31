"""
Unit and regression test for the vibrav package.
"""

# Import package, test suite, and other packages as needed
import vibrav
import pytest
import sys

def test_vibrav_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "vibrav" in sys.modules
