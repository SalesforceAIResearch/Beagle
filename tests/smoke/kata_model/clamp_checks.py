"""Copied as test_clamp.py in the image; not a host pytest test module."""
import unittest

from clamp import clamp


class ClampTests(unittest.TestCase):
    def test_below(self):
        self.assertEqual(clamp(-4, 0, 10), 0)

    def test_inside(self):
        self.assertEqual(clamp(6, 0, 10), 6)

    def test_above(self):
        self.assertEqual(clamp(14, 0, 10), 10)

    def test_negative_interval(self):
        self.assertEqual(clamp(0, -8, -2), -2)

    def test_float(self):
        self.assertEqual(clamp(3.75, 1.5, 2.5), 2.5)


if __name__ == "__main__":
    unittest.main()
