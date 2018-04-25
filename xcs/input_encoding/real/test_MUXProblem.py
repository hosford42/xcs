__author__ = 'Aaron Hosford'

import unittest

from xcs.input_encoding.real.bitstrings import BitString
from xcs.input_encoding.real.scenarios import MUXProblem


class TestMUXProblem(unittest.TestCase):

    def setUp(self):
        self.scenario = MUXProblem(training_cycles=200, address_size=5)

    def test_get_possible_actions(self):
        actions = self.scenario.get_possible_actions()
        self.assertTrue(len(actions) == 2)
        self.assertTrue(True in actions)
        self.assertTrue(False in actions)

    def test_sense(self):
        situation_size = (self.scenario.address_size +
                          (1 << self.scenario.address_size))
        previous = self.scenario.sense()
        self.assertIsInstance(previous, BitString)
        self.assertTrue(len(previous) == situation_size)
        while self.scenario.more():
            current = self.scenario.sense()
            self.assertIsInstance(current, BitString)
            self.assertTrue(len(current) == situation_size)
            if current != previous:
                break
        else:
            self.fail("All situations are the same.")

    def test_execute(self):
        for _ in range(10):
            situation = self.scenario.sense()
            index = self.scenario.get_index_from_situation(situation)
            value = round(situation[self.scenario.address_size + index])
            self.assertEqual(self.scenario.reward_on_success, self.scenario.execute(value))
            self.assertEqual(0, self.scenario.execute(not value))

    def test_more(self):
        self.scenario.reset()
        for _ in range(self.scenario.initial_training_cycles):
            self.scenario.sense()
            self.assertTrue(self.scenario.more())
            self.scenario.execute(False)
        self.assertFalse(self.scenario.more())


def main():
    unittest.main()

if __name__ == "__main__":
    main()
