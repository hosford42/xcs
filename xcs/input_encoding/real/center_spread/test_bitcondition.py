__author__ = 'Luis Da Costa'

import unittest
from random import randint, sample

from xcs.input_encoding.real.center_spread.bitstrings import BitConditionRealEncoding, BitString
from xcs.input_encoding.real.center_spread.util import EncoderDecoder


class TestBitCondition(unittest.TestCase):

    def setUp(self):
        self.real_translator = EncoderDecoder(min_value=0, max_value=255, encoding_bits=8)
        self.condition = BitConditionRealEncoding(encoder=self.real_translator, center_spreads=[(3, 2), (5, 1)], mutation_strength=.1)

    def test_init(self):
        for _ in range(10):
            condition = BitConditionRealEncoding.random(self.real_translator, mutation_strength=.1, length=2)
            comp_array = [
                condition.clip_center_spread((center, spread)) == (center, spread)
                for center, spread in condition.center_spreads
            ]
            self.assertTrue(all(comp_array), comp_array)

    def test_contains(self):
        for _ in range(10):
            condition = BitConditionRealEncoding.random(self.real_translator, mutation_strength=.1, length=randint(2, 20))
            for center_spread in condition.center_spreads:
                self.assertTrue(center_spread in condition)

    def test_iter(self):
        for (center, spread) in self.condition:
            print("(%.2f, %.2f)" % (center, spread))
        for value in BitString(encoder=self.real_translator, reals=[4, 5]):
            print("value: %.2f" % (value))

    def test_matched(self):
        # situation(s)
        situation = BitString(encoder=self.real_translator, reals=[4, 5])
        self.assertTrue(self.condition(situation))
        # condition(s)
        self.assertTrue(self.condition(self.condition))
        a_condition = BitConditionRealEncoding(encoder=self.real_translator, center_spreads=[(3, 1), (5, 0)], mutation_strength=.1)
        self.assertTrue(self.condition(a_condition))

    def test_not_matched(self):
        # situation(s)
        situation = BitString(encoder=self.real_translator, reals=[14, 5])
        self.assertFalse(self.condition(situation))
        # condition(s)
        a_condition = BitConditionRealEncoding(encoder=self.real_translator, center_spreads=[(1, 1), (6, 2)], mutation_strength=.1)
        self.assertFalse(self.condition(a_condition))

    def test_crossover(self):
        a_translator = EncoderDecoder(min_value=-255, max_value=255, encoding_bits=8)
        for a_length in sample(range(4, 20), 10):
            cond1 = BitConditionRealEncoding.random(encoder=a_translator, mutation_strength=.1, length=a_length)
            cond2 = BitConditionRealEncoding.random(encoder=a_translator, mutation_strength=.1, length=a_length)
            kid1, kid2 = cond1.crossover_with(cond2, 2)
            self.assertEqual(len(kid1), len(kid2))
            self.assertEqual(len(kid1), len(cond1))
            for center_spread in kid1:
                self.assertTrue((center_spread in cond1) or (center_spread in cond2))
            for center_spread in kid2:
                self.assertTrue((center_spread in cond1) or (center_spread in cond2))

    def test_mutate_interval_by_translation(self):
        for _ in range(10):
            situation = BitString.random_from(self.real_translator, randint(2, 20))
            for _ in range(10):
                a_condition = situation.cover()
                self.assertTrue(a_condition(situation)) # just to check.
                idx = randint(0, len(situation) - 1)
                an_interval = a_condition.center_spreads[idx]
                a_value = situation.as_reals[idx]
                another_interval=a_condition._mutate_interval_by_translation(interval=an_interval, value=a_value)
                self.assertLessEqual(a_value, another_interval[0] + another_interval[1])
                self.assertGreaterEqual(a_value, another_interval[0] - another_interval[1])
                # the only way the result is equal to the input is when the spread is 0.
                center, spread = an_interval
                bottom, top = center - spread, center + spread
                self.assertTrue((an_interval != another_interval) or (spread == 0) or (top - bottom == self.real_translator.extremes[1] - self.real_translator.extremes[0]))

    def test_mutate_interval_by_stretching(self):
        for _ in range(10):
            situation = BitString.random_from(self.real_translator, randint(2, 20))
            for _ in range(10):
                a_condition = situation.cover()
                self.assertTrue(a_condition(situation))  # just to be sure
                idx = randint(0, len(situation) - 1)
                a_value = situation.as_reals[idx]
                another_interval=a_condition._mutate_interval_by_stretching(interval=a_condition.center_spreads[idx], value=a_value)
                self.assertLessEqual(a_value, another_interval[0] + another_interval[1])
                self.assertGreaterEqual(a_value, another_interval[0] - another_interval[1])

    def test_mutate(self):
        for _ in range(10):
            situation = BitString.random_from(self.real_translator, randint(2, 20))
            # print("***************")
            # print(situation)
            for _ in range(10):
                a_condition = situation.cover()
                self.assertTrue(a_condition(situation)) # just to check.
                mutated_condition = a_condition.mutate(situation)
                # print(a_condition)
                # print(mutated_condition)
                self.assertTrue(mutated_condition(situation))
