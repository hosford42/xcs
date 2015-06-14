__author__ = 'Aaron Hosford'

import logging
import unittest

import xcs
import xcs.bitstrings as bitstrings
import xcs.scenarios


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
            self.assertTrue('numpy' in bitstrings.BitString.__module__)
            bitstrings.use_pure_python()
            self.assertFalse(bitstrings.using_numpy())
            self.assertTrue('python' in bitstrings.BitString.__module__)
            bitstrings.use_numpy()
            self.assertTrue(bitstrings.using_numpy())
            self.assertTrue('numpy' in bitstrings.BitString.__module__)
            logging.disable(logging.CRITICAL)
            try:
                xcs.test(scenario=xcs.scenarios.MUXProblem(1000))
            finally:
                logging.disable(logging.NOTSET)
        else:
            self.assertFalse(bitstrings.using_numpy())
            self.assertTrue('python' in bitstrings.BitString.__module__)
            logging.disable(logging.CRITICAL)
            try:
                xcs.test(scenario=xcs.scenarios.MUXProblem(1000))
            finally:
                logging.disable(logging.NOTSET)

    def test_from_int(self):
        bitstring = bitstrings.BitString(149, 8)
        self.assertTrue(self.bitstring == bitstring)

    def test_from_string(self):
        self.assertTrue(
            bitstrings.BitString(str(self.bitstring)) == self.bitstring
        )

    def test_random(self):
        previous = bitstrings.BitString.random(len(self.bitstring), .5)
        self.assertEqual(len(previous), len(self.bitstring))
        for i in range(10):
            current = bitstrings.BitString.random(
                len(self.bitstring),
                1 / (i + 2)
            )
            self.assertEqual(len(current), len(self.bitstring))
            if previous != current:
                break
            previous = current
        else:
            self.fail("Failed to produce distinct random bitstrings.")

    def test_crossover_template(self):
        previous = bitstrings.BitString.crossover_template(
            len(self.bitstring),
            2
        )
        self.assertEqual(len(previous), len(self.bitstring))
        for i in range(10):
            current = bitstrings.BitString.crossover_template(
                len(self.bitstring),
                i + 1
            )
            self.assertEqual(len(current), len(self.bitstring))
            if previous != current:
                break
            previous = current
        else:
            self.fail("Failed to produce distinct crossover templates.")

    def test_any(self):
        self.assertTrue(self.bitstring.any())
        self.assertFalse(
            bitstrings.BitString(0, len(self.bitstring)).any()
        )
        self.assertTrue(
            bitstrings.BitString(-1, len(self.bitstring)).any()
        )

    def test_count(self):
        self.assertTrue(self.bitstring.count() == 4)
        self.assertTrue(
            bitstrings.BitString(0, len(self.bitstring)).count() == 0
        )
        self.assertTrue(
            bitstrings.BitString(-1, len(self.bitstring)).count() ==
            len(self.bitstring)
        )

    def test_and(self):
        self.assertEqual(self.bitstring, self.bitstring & self.bitstring)
        self.assertEqual(
            self.bitstring & ~self.bitstring,
            bitstrings.BitString([0] * len(self.bitstring))
        )

    def test_or(self):
        self.assertEqual(self.bitstring, self.bitstring | self.bitstring)
        self.assertEqual(
            self.bitstring | ~self.bitstring,
            bitstrings.BitString([1] * len(self.bitstring))
        )

    def test_xor(self):
        mask = bitstrings.BitString.random(len(self.bitstring))
        self.assertEqual(
            mask ^ mask,
            bitstrings.BitString([0] * len(self.bitstring))
        )
        self.assertEqual(self.bitstring, (self.bitstring ^ mask) ^ mask)

    def test_invert(self):
        self.assertNotEqual(self.bitstring, ~self.bitstring)
        self.assertEqual(self.bitstring, ~~self.bitstring)

    def test_plus(self):
        self.assertEqual(
            self.bitstring + ~self.bitstring,
            bitstrings.BitString(list(self.bitstring) +
                                 list(~self.bitstring))
        )

    def test_slice(self):
        self.assertEqual(self.bitstring, self.bitstring[:])
        self.assertEqual(self.bitstring,
                         self.bitstring[:2] + self.bitstring[2:])
        self.assertEqual(self.bitstring,
                         self.bitstring[-len(self.bitstring):])
        self.assertEqual(self.bitstring,
                         self.bitstring[:len(self.bitstring)])
        self.assertEqual(
            self.bitstring,
            self.bitstring[0:3] + self.bitstring[3:len(self.bitstring)]
        )

    def test_index(self):
        self.assertEqual(
            list(self.bitstring),
            [self.bitstring[index] for index in range(len(self.bitstring))]
        )
        self.assertEqual(
            list(self.bitstring),
            [self.bitstring[index]
             for index in range(-len(self.bitstring), 0)]
        )


def main():
    unittest.main()

if __name__ == "__main__":
    main()
