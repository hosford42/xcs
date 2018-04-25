__author__ = 'Luis Da Costa'

import unittest

from random import randint

from xcs.input_encoding.real.center_spread.bitstrings import BitConditionRealEncoding, BitString
from xcs.input_encoding.real.center_spread.util import EncoderDecoder


class TestBitString(unittest.TestCase):

    def setUp(self):
        self.real_translator = EncoderDecoder(min_value=0, max_value=255, encoding_bits=8)
        # self.condition = BitConditionRealEncoding(encoder=self.real_translator, center_spreads=[(3, 2), (5, 1)], mutation_strength=.1)

    def test_cover(self):
        for _ in range(10):
            situation = BitString.random_from(self.real_translator, randint(2, 20))
            print("\n")
            print(situation)
            for _ in range(10):
                condition = situation.cover(wildcard_probability=.33)
                print(condition)
                self.assertTrue(condition(situation))
