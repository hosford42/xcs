__author__ = 'Aaron Hosford'

import unittest

from xcs.bitstrings import BitString, BitCondition


class TestBitCondition(unittest.TestCase):

    def setUp(self):
        self.bitstring1 = BitString('101101')
        self.bitstring2 = BitString('100101')
        self.bitstring3 = BitString('100011')
        self.bitstring4 = BitString('010010')

        self.all_combos1 = BitCondition('000111###')
        self.all_combos2 = BitCondition('01#01#01#')

    def test_init(self):
        condition1 = BitCondition('1###01')
        condition2 = BitCondition(self.bitstring2, self.bitstring3)
        condition3 = BitCondition(self.bitstring2, self.bitstring2)
        condition4 = BitCondition(self.bitstring1, self.bitstring3)
        condition5 = BitCondition(self.bitstring4, self.bitstring3)
        self.assertTrue(condition1 == condition2)
        self.assertTrue(condition1 != condition3)
        self.assertTrue(condition1 == condition4)
        self.assertTrue(condition1 != condition5)

    def test_matched(self):
        condition = BitCondition('###101')
        self.assertTrue(condition(self.bitstring1))

    def test_unmatched(self):
        condition = BitCondition('001###')
        self.assertFalse(condition(self.bitstring1))

    def test_cover(self):
        covered = BitCondition.cover(self.bitstring1, .5)
        self.assertTrue(covered(covered))
        self.assertTrue(covered(self.bitstring1))

        covered = BitCondition.cover(self.bitstring1, 0)
        self.assertTrue(covered.mask.count() == len(self.bitstring1))

        covered = BitCondition.cover(self.bitstring1, 1)
        self.assertTrue(covered.mask.count() == 0)

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
        self.assertTrue(
            child1.mask == child2.mask == parent1.mask == parent2.mask
        )
        self.assertFalse(child1.bits != ~child2.bits & child1.mask)

    def test_bitwise_and(self):
        # Provided the two conditions have compatible bits, their
        # intersection should be matched by both. If they don't have
        # compatible bits, all bets are off.
        result = self.all_combos1 & self.all_combos2
        self.assertEqual(result, BitCondition('00001101#'))

    def test_bitwise_or(self):
        # Provided the two conditions have compatible bits, their
        # union should match both of them. If they don't have
        # compatible bits, all bets are off.
        result = self.all_combos1 | self.all_combos2
        self.assertEqual(result, BitCondition('0###1####'))
        self.assertTrue(
            result(self.all_combos1.bits | self.all_combos2.bits)
        )

    def test_bitwise_invert(self):
        # Each unmasked bit gets inverted. The mask is unchanged.
        result = ~self.all_combos1
        self.assertEqual(result, BitCondition('111000###'))
        self.assertTrue(result(~self.all_combos1.bits))


def main():
    unittest.main()

if __name__ == "__main__":
    main()
