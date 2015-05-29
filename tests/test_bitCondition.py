__author__ = 'Aaron Hosford'

import unittest

from xcs.bitstrings import BitString, BitCondition


class TestBitCondition(unittest.TestCase):

    def setUp(self):
        self.bitstring1 = BitString.from_string('101101')
        self.bitstring2 = BitString.from_string('100101')
        self.bitstring3 = BitString.from_string('100011')
        self.bitstring4 = BitString.from_string('010010')

    def test_init(self):
        condition1 = BitCondition.from_string('1###01')
        condition2 = BitCondition(self.bitstring2, self.bitstring3)
        condition3 = BitCondition(self.bitstring2, self.bitstring2)
        condition4 = BitCondition(self.bitstring1, self.bitstring3)
        condition5 = BitCondition(self.bitstring4, self.bitstring3)
        self.assertTrue(condition1 == condition2)
        self.assertTrue(condition1 != condition3)
        self.assertTrue(condition1 == condition4)
        self.assertTrue(condition1 != condition5)

    def test_matched(self):
        condition = BitCondition.from_string('###101')
        self.assertTrue(condition(self.bitstring1))

    def test_unmatched(self):
        condition = BitCondition.from_string('001###')
        self.assertFalse(condition(self.bitstring1))

    def test_cover(self):
        condition = BitCondition.from_string('1###01')
        covered = condition.cover(self.bitstring1, .5)
        self.assertTrue(covered(covered))
        self.assertTrue(covered(self.bitstring1))

    def test_bits(self):
        condition = BitCondition(self.bitstring1, self.bitstring2)
        self.assertTrue(condition(self.bitstring1))
        self.assertTrue(condition.bits & self.bitstring1 == condition.bits)

    def test_mask(self):
        condition = BitCondition(self.bitstring3, self.bitstring1)
        self.assertTrue(condition.mask == self.bitstring1)
        self.assertFalse(any(condition.bits & ~condition.mask))

    def test_count(self):
        condition = BitCondition(self.bitstring4, self.bitstring1)
        self.assertTrue(condition.count() == condition.mask.count())

    def test_crossover_with(self):
        parent1 = BitCondition(self.bitstring1, self.bitstring3)
        inbred1, inbred2 = parent1.crossover_with(parent1)
        self.assertTrue(inbred1 == inbred2 == parent1)

        parent2 = BitCondition(self.bitstring4, self.bitstring3)
        child1, child2 = parent1.crossover_with(parent2)
        self.assertTrue(child1.mask == child2.mask == parent1.mask == parent2.mask)
        self.assertFalse(child1.bits != ~child2.bits & child1.mask)


def main():
    unittest.main()

if __name__ == "__main__":
    main()
