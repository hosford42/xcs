__author__ = 'Aaron Hosford'

import unittest

from xcs.bitstrings import BitString
from xcs.problems import MUXProblem


class TestMUXProblem(unittest.TestCase):

    def setUp(self):
        self.problem = MUXProblem(10)

    def test_get_possible_actions(self):
        actions = self.problem.get_possible_actions()
        self.assertTrue(len(actions) == 2)
        self.assertTrue(True in actions)
        self.assertTrue(False in actions)

    def test_sense(self):
        situation_size = self.problem.address_size + (1 << self.problem.address_size)
        previous = self.problem.sense()
        self.assertIsInstance(previous, BitString)
        self.assertTrue(len(previous) == situation_size)
        while self.problem.more():
            current = self.problem.sense()
            self.assertIsInstance(current, BitString)
            self.assertTrue(len(current) == situation_size)
            if current != previous:
                break
        else:
            self.fail("All situations are the same.")

    def test_execute(self):
        situation = self.problem.sense()
        index = int(situation[:self.problem.address_size])
        value = situation[self.problem.address_size + index]
        self.assertEqual(1, self.problem.execute(value))
        self.assertEqual(0, self.problem.execute(not value))

    def test_more(self):
        self.problem.reset()
        for _ in range(self.problem.initial_training_cycles):
            self.problem.sense()
            self.assertTrue(self.problem.more())
            self.problem.execute(False)
        self.assertFalse(self.problem.more())


def main():
    unittest.main()

if __name__ == "__main__":
    main()
