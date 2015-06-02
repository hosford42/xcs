__author__ = 'Aaron Hosford'

import logging
import unittest

import xcs
import xcs.bitstrings as bitstrings
import xcs.problems


# noinspection PyUnresolvedReferences
class TestBitString(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_numpy = False

    def run(self, result=None):
        if bitstrings.numpy_is_available():
            self.use_numpy = True
            super().run(result)
        self.use_numpy = False
        super().run(result)

    def setUp(self):
        if self.use_numpy:
            bitstrings.use_numpy()
        else:
            bitstrings.use_pure_python()

        self.bitstring = bitstrings.BitString('10010101')  # 149

    def test_using(self):
        if self.use_numpy:
            self.assertTrue(bitstrings.using_numpy())
            self.assertTrue('fast' in bitstrings.BitString.__module__)
            bitstrings.use_pure_python()
            self.assertFalse(bitstrings.using_numpy())
            self.assertTrue('slow' in bitstrings.BitString.__module__)
            bitstrings.use_numpy()
            self.assertTrue(bitstrings.using_numpy())
            self.assertTrue('fast' in bitstrings.BitString.__module__)
            logging.disable(logging.CRITICAL)
            try:
                xcs.test(problem=xcs.problems.MUXProblem(1000))
            finally:
                logging.disable(logging.NOTSET)
        else:
            self.assertFalse(bitstrings.using_numpy())
            self.assertTrue('slow' in bitstrings.BitString.__module__)
            logging.disable(logging.CRITICAL)
            try:
                xcs.test(problem=xcs.problems.MUXProblem(1000))
            finally:
                logging.disable(logging.NOTSET)

    def test_from_int(self):
        bitstring = bitstrings.BitString.from_int(149, 8)
        self.assertTrue(self.bitstring == bitstring)

    def test_from_string(self):
        self.assertTrue(bitstrings.BitString(str(self.bitstring)) == self.bitstring)

    def test_random(self):
        previous = bitstrings.BitString.random(len(self.bitstring), .5)
        for i in range(10):
            current = bitstrings.BitString.random(len(self.bitstring), 1 / (i + 2))
            if previous != current:
                break
            previous = current
        else:
            self.fail("Failed to produce distinct random bitstrings.")

    def test_crossover_template(self):
        previous = bitstrings.BitString.crossover_template(len(self.bitstring), 2)
        for i in range(10):
            current = bitstrings.BitString.crossover_template(len(self.bitstring), i + 1)
            if previous != current:
                break
            previous = current
        else:
            self.fail("Failed to produce distinct crossover templates.")

    def test_any(self):
        self.assertTrue(self.bitstring.any())
        self.assertFalse(bitstrings.BitString.from_int(0, len(self.bitstring)).any())
        self.assertTrue(bitstrings.BitString.from_int(-1, len(self.bitstring)).any())

    def test_count(self):
        self.assertTrue(self.bitstring.count() == 4)
        self.assertTrue(bitstrings.BitString.from_int(0, len(self.bitstring)).count() == 0)
        self.assertTrue(bitstrings.BitString.from_int(-1, len(self.bitstring)).count() == len(self.bitstring))


def main():
    unittest.main()

if __name__ == "__main__":
    main()
