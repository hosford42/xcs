__author__ = 'Aaron Hosford'

import logging
import unittest

import xcs
from xcs.problems import MUXProblem, HaystackProblem


class TestXCS(unittest.TestCase):

    def test_against_MUX(self):
        problem = MUXProblem(training_cycles=10000, address_size=3)

        algorithm = xcs.XCSAlgorithm(problem.get_possible_actions())
        algorithm.exploration_probability = .1
        algorithm.discount_factor = 0
        algorithm.do_GA_subsumption = True
        algorithm.do_action_set_subsumption = True

        for _ in range(2):
            logging.disable(logging.CRITICAL)
            try:
                problem.reset()
                steps, total_reward, time_passed, population = xcs.test(algorithm, problem)
            finally:
                logging.disable(logging.NOTSET)

            average_reward = total_reward / steps
            self.assertGreater(average_reward, .49)
            self.assertLess(time_passed, 40)
            if average_reward >= .6:
                break
        else:
            self.fail("Failed to achieve expected average reward level.")

    def test_against_haystack(self):
        problem = HaystackProblem(training_cycles=10000, input_size=500)

        algorithm = xcs.XCSAlgorithm(problem.get_possible_actions())
        algorithm.exploration_probability = .1
        algorithm.discount_factor = 0
        algorithm.do_GA_subsumption = False
        algorithm.do_action_set_subsumption = False
        algorithm.wildcard_probability = .99
        algorithm.deletion_threshold = 10
        algorithm.mutation_probability = .0001

        for _ in range(3):
            logging.disable(logging.CRITICAL)
            try:
                problem.reset()
                steps, total_reward, time_passed, population = xcs.test(algorithm, problem)
            finally:
                logging.disable(logging.NOTSET)

            average_reward = total_reward / steps
            self.assertGreater(average_reward, .49)
            self.assertLess(time_passed, 20)
            if average_reward >= .6:
                break
        else:
            self.fail("Failed to achieve expected average reward level.")


def main():
    unittest.main()


if __name__ == "__main__":
    main()
