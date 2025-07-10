import unittest
from unittest.mock import Mock
import numpy as xp
from wakis import FieldMonitor, Field


class TestFieldMonitor(unittest.TestCase):
    def setUp(self):
        self.field_monitor = FieldMonitor(frequencies=[1e9]) # TODO)

    def test___init__(self):
        pass  # is tested by setup

    def test_update(self):
        field = Mock(Field)
        field.Nx = 12
        field.Ny = 12
        field.Nz = 12
        field.xp = xp
        # todo
        self.field_monitor.update(E=field, dt=124342143213214.)

    def test_get_components(self):
        self.field_monitor.get_components()


if __name__ == "__main__":
    unittest.main()
