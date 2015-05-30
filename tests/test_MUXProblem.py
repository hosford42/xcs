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
        # TODO
        pass

    def test_more(self):
        # TODO
        pass


def main():
    unittest.main()

if __name__ == "__main__":
    main()