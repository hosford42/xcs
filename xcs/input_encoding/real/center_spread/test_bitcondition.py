__author__ = 'Luis Da Costa'

import unittest

from xcs.input_encoding.real.center_spread.bitstrings import BitConditionRealEncoding, BitString
from xcs.input_encoding.real.center_spread.util import EncoderDecoder

class TestBitCondition(unittest.TestCase):

    def setUp(self):
        self.real_translator = EncoderDecoder(min_value=0, max_value=255, encoding_bits=8)
        self.condition = BitConditionRealEncoding(encoder=self.real_translator, center_spreads=[(3, 2), (5, 1)], mutation_strength=.1)

    def test_iter(self):
        for (center, spread) in self.condition:
            print("(%.2f, %.2f)" % (center, spread))
        for value in BitString(encoder=self.real_translator, reals=[4, 5]):
            print("value: %.2f" % (value))

    def test_matched(self):
        situation = BitString(encoder=self.real_translator, reals=[4, 5])
        self.assertTrue(self.condition(situation))

    def test_not_matched(self):
        situation = BitString(encoder=self.real_translator, reals=[14, 5])
        self.assertFalse(self.condition(situation))

    def test_crossover(self):
        a_translator = EncoderDecoder(min_value=-255, max_value=255, encoding_bits=8)
        cond1 = BitConditionRealEncoding(
            encoder=a_translator,
            center_spreads=[(3, 2), (5, 1), (13, 12), (15, 11), (23, 22), (25, 21), (33, 32), (35, 31)], mutation_strength=.1)
        cond2 = BitConditionRealEncoding(
            encoder=a_translator,
            center_spreads=[(-3, -2), (-5, -1), (-13, -12), (-15, -11), (-23, -22), (-25, -21), (-33, -32), (-35, -31)], mutation_strength=.1)
        kid1, kid2 = cond1.crossover_with(cond2, -1, 2)
        print(kid1)
        print(kid2)
        self.assertEqual(len(kid1), len(kid2))
        self.assertEqual(len(kid1), len(cond1))

