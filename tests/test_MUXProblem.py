__author__ = 'Aaron Hosford'

import unittest

from xcs.bitstrings import BitString
from xcs.scenarios import MUXProblem


class TestMUXProblem(unittest.TestCase):

    def setUp(self):
        self.scenario = MUXProblem(10)

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
        situation = self.scenario.sense()
        index = int(situation[:self.scenario.address_size])
        value = situation[self.scenario.address_size + index]
        self.assertEqual(1, self.scenario.execute(value))
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
