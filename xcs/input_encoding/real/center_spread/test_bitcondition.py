__author__ = 'Luis Da Costa'

import unittest
from random import random, randint, sample

from xcs.input_encoding.real.center_spread.bitstrings import BitConditionRealEncoding, BitString
from xcs.input_encoding.real.center_spread.util import EncoderDecoder, random_in


class TestBitCondition(unittest.TestCase):

    def setUp(self):
        self.real_translator = EncoderDecoder(min_value=0, max_value=255, encoding_bits=8)
        self.condition = BitConditionRealEncoding(encoder=self.real_translator, center_spreads=[(3, 2), (5, 1)], mutation_strength=.1, mutation_prob=.2)

    def test_init(self):
        for _ in range(10):
            condition = BitConditionRealEncoding.random(self.real_translator, mutation_strength=.1, mutation_prob=.2, length=2)
            comp_array = [
                condition.clip_center_spread((center, spread)) == (center, spread)
                for center, spread in condition.center_spreads
            ]
            self.assertTrue(all(comp_array), comp_array)

    def test_contains(self):
        for _ in range(10):
            condition = BitConditionRealEncoding.random(self.real_translator, mutation_strength=.1, mutation_prob=.2, length=randint(2, 20))
            for center_spread in condition.center_spreads:
                self.assertTrue(center_spread in condition)

    def test_iter(self):
        for (center, spread) in self.condition:
            print("(%.2f, %.2f)" % (center, spread))
        for value in BitString(encoder=self.real_translator, reals=[4, 5]):
            print("value: %.2f" % (value))

    def test_matched(self):
        for _ in range(10):
            condition = BitConditionRealEncoding.random(self.real_translator, mutation_strength=.1, mutation_prob=.2, length=randint(2, 20))
            # situation(s)
            reals = [random_in(center - spread, center + spread) for center, spread in condition.center_spreads]
            situation = BitString(self.real_translator, reals)
            # print(condition)
            # print(situation)
            self.assertTrue(condition(situation))
        # condition(s)
        self.assertTrue(self.condition(self.condition))
        a_condition = BitConditionRealEncoding(encoder=self.real_translator, center_spreads=[(3, 1), (5, 0)], mutation_prob=.2, mutation_strength=.1)
        self.assertTrue(self.condition(a_condition))

    def test_not_matched(self):
        # situation(s)
        situation = BitString(encoder=self.real_translator, reals=[14, 5])
        self.assertFalse(self.condition(situation))
        # condition(s)
        a_condition = BitConditionRealEncoding(encoder=self.real_translator, center_spreads=[(1, 1), (6, 2)], mutation_prob=.2, mutation_strength=.1)
        self.assertFalse(self.condition(a_condition))

    def test_crossover(self):
        a_translator = EncoderDecoder(min_value=-255, max_value=255, encoding_bits=8)
        for a_length in sample(range(4, 50), 25):
            cond1 = BitConditionRealEncoding.random(encoder=a_translator, mutation_strength=.1, mutation_prob=.2, length=a_length)
            cond2 = BitConditionRealEncoding.random(encoder=a_translator, mutation_strength=.1, mutation_prob=.2, length=a_length)
            kid1, kid2 = cond1.crossover_with(cond2, randint(1, a_length - 2))
            self.assertEqual(len(kid1), len(kid2))
            self.assertEqual(len(kid1), len(cond1))
            self.assertTrue((kid1 != kid2) or (cond1 == cond2))
            for center_spread in kid1:
                self.assertTrue((center_spread in cond1) or (center_spread in cond2))
            for center_spread in kid2:
                self.assertTrue((center_spread in cond1) or (center_spread in cond2))

    def test_mutate_interval_by_translation(self):
        for _ in range(10):
            situation = BitString.random_from(self.real_translator, randint(2, 20))
            for _ in range(10):
                a_condition = situation.cover(.33)
                self.assertTrue(a_condition(situation)) # just to check.
                idx = randint(0, len(situation) - 1)
                an_interval = a_condition.center_spreads[idx]
                a_value = situation.as_reals[idx]
                another_interval=a_condition._mutate_interval_by_translation(interval=an_interval, value=a_value)
                self.assertLessEqual(a_value, another_interval[0] + another_interval[1])
                self.assertGreaterEqual(a_value, another_interval[0] - another_interval[1])
                #
                center, spread = an_interval
                bottom, top = center - spread, center + spread
                self.assertTrue(
                    (an_interval != another_interval) or  # either: the intervals are different, or
                    (spread == 0) or  # the interval is length = 0, or
                    (top - bottom == self.real_translator.extremes[1] - self.real_translator.extremes[0])) # the interval occupies the whole space

    def test_mutate_interval_by_stretching(self):
        for _ in range(10):
            situation = BitString.random_from(self.real_translator, randint(2, 20))
            for _ in range(10):
                a_condition = situation.cover(.33)
                self.assertTrue(a_condition(situation))  # just to be sure
                idx = randint(0, len(situation) - 1)
                a_value = situation.as_reals[idx]
                an_interval = a_condition.center_spreads[idx]
                another_interval=a_condition._mutate_interval_by_stretching(interval=an_interval, value=a_value)
                self.assertNotEqual(an_interval, another_interval, str(an_interval))
                self.assertLessEqual(a_value, another_interval[0] + another_interval[1])
                self.assertGreaterEqual(a_value, another_interval[0] - another_interval[1])

    def test_mutate(self):
        for _ in range(10):
            situation = BitString.random_from(self.real_translator, randint(2, 20))
            # print("***************")
            # print(situation)
            for _ in range(10):
                a_condition = situation.cover(.33)
                self.assertTrue(a_condition(situation)) # just to check.
                mutated_condition = a_condition.mutate(situation)
                # print(a_condition)
                # print(mutated_condition)
                self.assertTrue(mutated_condition(situation))

    @unittest.skip("On original paper they don't clip. Let's try.")
    def test_clip(self):
        encoder = EncoderDecoder(min_value=0, max_value=1, encoding_bits=4)
        for _ in range(1000):
            center = random()
            spread = random()
            nc, ns = BitConditionRealEncoding.clip_center_spread_class(encoder, (center, spread))
            if center - spread >= encoder.extremes[0]:
                if center + spread <= encoder.extremes[1]:
                    self.assertEqual(center, nc)
                    self.assertEqual(spread, ns)
                else:
                    self.assertEqual(ns, encoder.extremes[1] - nc)
            else:
                if center + spread <= encoder.extremes[1]:
                    self.assertEqual(ns, nc - encoder.extremes[0])
                else:
                    self.assertEqual(ns, min(encoder.extremes[1] - nc, nc - encoder.extremes[0]))

            self.assertEqual(center, nc)
            self.assertTrue(nc - ns >= encoder.extremes[0]) and (nc + ns <= encoder.extremes[1])
