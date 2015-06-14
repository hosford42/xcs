__author__ = 'Aaron Hosford'

import logging
import unittest

import xcs
from xcs.scenarios import MUXProblem, HaystackProblem


class TestXCS(unittest.TestCase):

    def test_against_MUX(self):
        scenario = MUXProblem(training_cycles=10000, address_size=3)

        algorithm = xcs.XCSAlgorithm()
        algorithm.exploration_probability = .1
        algorithm.discount_factor = 0
        algorithm.do_ga_subsumption = True
        algorithm.do_action_set_subsumption = True

        best = None
        expected = .6

        for _ in range(2):
            logging.disable(logging.CRITICAL)
            try:
                scenario.reset()
                steps, total_reward, time_passed, population = xcs.test(
                    algorithm,
                    scenario
                )
            finally:
                logging.disable(logging.NOTSET)

            average_reward = total_reward / steps
            self.assertGreater(average_reward, .49)
            self.assertLess(time_passed, 40)
            if average_reward >= expected:
                break
            elif best is None or best < average_reward:
                best = average_reward
        else:
            self.fail("Failed to achieve expected average reward level. "
                      "(Missed by %f.)" % (expected - best))

    def test_against_haystack(self):
        scenario = HaystackProblem(training_cycles=10000, input_size=500)

        algorithm = xcs.XCSAlgorithm()
        algorithm.ga_threshold = 1
        algorithm.crossover_probability = .5
        algorithm.exploration_probability = .1
        algorithm.discount_factor = 0
        algorithm.do_ga_subsumption = False
        algorithm.do_action_set_subsumption = True
        algorithm.wildcard_probability = 1 - 1 / 500
        algorithm.deletion_threshold = 1
        algorithm.mutation_probability = 1 / 500

        best = None
        expected = .6

        for _ in range(2):
            logging.disable(logging.CRITICAL)
            try:
                scenario.reset()
                steps, total_reward, time_passed, population = xcs.test(
                    algorithm,
                    scenario
                )
            finally:
                logging.disable(logging.NOTSET)

            average_reward = total_reward / steps
            self.assertGreater(average_reward, .48)
            self.assertLess(time_passed, 100)
            if average_reward >= expected:
                break
            elif best is None or best < average_reward:
                best = average_reward
        else:
            self.fail("Failed to achieve expected average reward level. "
                      "(Missed by %f.)" % (expected - best))


def main():
    unittest.main()


if __name__ == "__main__":
    main()
