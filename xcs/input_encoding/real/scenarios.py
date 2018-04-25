# -*- coding: utf-8 -*-
# Some basic scenarios with real-space encoding.

__author__ = 'Luis Da Costa'

__all__ = [
    'MUXProblem',
]

import pathlib
import math
import queue
import os
from typing import Tuple

from xcs.scenarios import Scenario
from xcs.input_encoding.real.center_spread.util import EncoderDecoder
from xcs.input_encoding.real.center_spread.bitstrings import BitString as BitStringRealEncoded
import logging
from xcs.algorithms.xcs import XCSAlgorithm
from xcs.framework import ClassifierSet

from xcs.scenarios import ScenarioObserver

import time


class MUXProblem(Scenario):
    """Classic multiplexer problem. This scenario is static; each action
    affects only the immediate reward, and the environment is stateless.
    The address size indicates the number of bits used as an address/index
    into the remaining bits in the situations returned by sense(). The
    agent is expected to return the value of the indexed bit from the
    situation.

    Usage:
        scenario = MUXProblem()
        model = algorithm.run(scenario)

    Init Arguments:
        training_cycles: An int, the number of situations to produce;
            default is 10,000.
        address_size: An int, the number of bits devoted to addressing into
            the remaining bits of each situation. The total number of bits
            in each situation will be equal to address_size +
            2 ** address_size.
    """

    def __init__(self, training_cycles=10000, address_size=3, theta = 0.5):
        assert isinstance(training_cycles, int) and training_cycles > 0
        assert isinstance(address_size, int) and address_size > 0
        assert (theta >=0) and (theta <= 1)

        self.reward_on_success = 1000
        self.address_size = address_size
        self.current_situation = None
        self.possible_actions = (True, False)
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles
        # a pixel can take values in the [0,1] range, encoded with a certain number of bits.
        self.real_translator = EncoderDecoder(min_value=0, max_value=1, encoding_bits=16)
        self.theta = theta

    @property
    def is_dynamic(self):
        """A Boolean value indicating whether earlier actions from the same
        run can affect the rewards or outcomes of later actions."""
        return False

    def get_possible_actions(self):
        """Return a sequence containing the possible actions that can be
        executed within the environment.

        Usage:
            possible_actions = scenario.get_possible_actions()

        Arguments: None
        Return:
            A sequence containing the possible actions which can be
            executed within this scenario.
        """
        return self.possible_actions

    def reset(self):
        """Reset the scenario, starting it over for a new run.

        Usage:
            if not scenario.more():
                scenario.reset()

        Arguments: None
        Return: None
        """
        print("reset")
        self.remaining_cycles = self.initial_training_cycles

    def sense(self):
        """Return a situation, encoded as a bit string, which represents
        the observable state of the environment.

        Usage:
            situation = scenario.sense()
            assert isinstance(situation, BitString)

        Arguments: None
        Return:
            The current situation.
        """
        self.current_situation = [self.real_translator.random()
                                  for _ in range(self.address_size + int(math.pow(2, self.address_size)))]
        self.current_situation = BitStringRealEncoded(encoder=self.real_translator, reals=self.current_situation)
        return self.current_situation

    def get_index_from_situation(self, situation) -> int:
        index_as_bitstring = ''.join([ str(round(situation[i])) for i in range(self.address_size) ])
        return int(index_as_bitstring, 2)

    def execute(self, action):
        """Execute the indicated action within the environment and
        return the resulting immediate reward dictated by the reward
        program.

        Usage:
            immediate_reward = scenario.execute(selected_action)

        Arguments:
            action: The action to be executed within the current situation.
        Return:
            A float, the reward received for the action that was executed,
            or None if no reward is offered.
        """

        assert action in self.possible_actions

        self.remaining_cycles -= 1
        index_value = self.get_index_from_situation(self.current_situation)
        result = (self.current_situation[index_value + self.address_size] >= self.theta)
        return self.reward_on_success * float(result == action)

    def more(self):
        """Return a Boolean indicating whether additional actions may be
        executed, per the reward program.

        Usage:
            while scenario.more():
                situation = scenario.sense()
                selected_action = choice(possible_actions)
                reward = scenario.execute(selected_action)

        Arguments: None
        Return:
            A bool indicating whether additional situations remain in the
            current run.
        """
        return self.remaining_cycles > 0


class MUXScenarioObserver(ScenarioObserver):

    def __init__(self, wrapped, feedback_dir: str):
        # Ensure that the wrapped object implements the same interface
        assert isinstance(wrapped, Scenario)

        ScenarioObserver.__init__(self, wrapped)
        self.latest_feedback_time = None  # when was the last time I gave feedback to the user
        self.feedback_dir = feedback_dir
        pathlib.Path(self.feedback_dir).mkdir(parents=True, exist_ok=True)
        # let's keep 'number of success(es) on "exploit" problems' as a measure of performance
        self.window_length = 50  # in number of steps. 50 == taken in https://pdfs.semanticscholar.org/6777/624d3a7230742d1b8f6bc5f8a43d6daf065d.pdf?_ga=2.120249934.77113088.1523040926-378861627.1523040926
        self.latest_exploit_rewards = queue.Queue(maxsize=self.window_length)
        self.num_latest_success = 0  # number of successes in current window
        self.num_exploits = 0  # how many 'exploit'ation problems have I seen
        self.success_per_period = []  # history of successes per window.

    def _get_free_file_name(self, root: str, ext: str) -> str:
        fcounter = 1
        fname = os.path.join(self.feedback_dir, "%s_%d.%s" % (root, fcounter, ext))
        while os.path.isfile(fname):
            fcounter += 1
            fname = os.path.join(self.feedback_dir, "%s_%d.%s" % (root, fcounter, ext))
        return fname

    @property
    def exploit_successes_on_window(self) -> Tuple[int, float]:
        """Returns exploit successes on window, as an absolute value and as a proportion (in [0,1])"""
        if self.steps == 0:
            return (0,0)
        return self.num_latest_success, \
               self.num_latest_success / (self.steps if self.steps < self.window_length else self.window_length)

    def execute(self, action, **kwargs):
        self.logger.debug('Executing action: %s', action)
        if 'is_exploit' not in kwargs:
            raise RuntimeError("execute: need to know if this action came from exploitation or exploration")
        is_exploit = kwargs['is_exploit']
        reward = self.wrapped.execute(action)
        if reward is not None:
            self.total_reward += reward
            if is_exploit:
                self.num_exploits += 1
                if self.latest_exploit_rewards.full():
                    # dump this value
                    self.success_per_period.append(self.num_latest_success)
                    # update statistics
                    old_reward = self.latest_exploit_rewards.get()
                    self.num_latest_success -= 1 if old_reward > 0 else 0
                self.latest_exploit_rewards.put(reward)
                self.num_latest_success += 1 if reward > 0 else 0
        self.steps += 1

        self.logger.debug('Reward received on this step: %.5f',
                          reward or 0)
        self.logger.debug('Average reward per step: %.5f',
                          self.total_reward / self.steps)
        succ_abs, succ_perc = self.exploit_successes_on_window
        self.logger.debug('[exploit trials so far: %d] Num successes on latest tries: %d (perc success of %.5f)',
                          self.num_exploits, succ_abs, succ_perc)

        return reward

    def more(self):
        """Return a Boolean indicating whether additional actions may be
        executed, per the reward program.

        Usage:
            while scenario.more():
                situation = scenario.sense()
                selected_action = choice(possible_actions)
                reward = scenario.execute(selected_action)

        Arguments: None
        Return:
            A bool indicating whether additional situations remain in the
            current run.
        """
        more = (self.num_exploits <= 10000)  # self.wrapped.more()# TODO more
        current_time = time.time()
        if self.latest_feedback_time is None or (current_time - self.latest_feedback_time >= 5):  # seconds between feedback
            self.latest_feedback_time = current_time
            self.logger.info('Steps completed: %d', self.steps)
            self.logger.info('Average reward per step: %.5f',
                             self.total_reward / (self.steps or 1))
            succ_abs, succ_perc = self.exploit_successes_on_window
            self.logger.debug('[exploit trials so far: %d] Num successes on latest tries: %d (perc success of %.5f)',
                              self.num_exploits, succ_abs, succ_perc)
        if not more:
            self.logger.info('Run completed.')
            self.logger.info('Total steps: %d', self.steps)
            self.logger.info('Total reward received: %.5f',
                             self.total_reward)
            self.logger.info('Average reward per step: %.5f',
                             self.total_reward / (self.steps or 1))
            # save successes to disk:
            import pickle

            with open(self._get_free_file_name(root="exploit_successes", ext="txt"), 'wb') as fp:
                pickle.dump(self.success_per_period, fp)
            self.logger.info("Successes on 'exploit' problems saved on directory %s" % (self.feedback_dir))
        return more

if __name__ == "__main__":

    # Setup logging so we can see the test run as it progresses.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # or:
    # logging.root.setLevel(logging.INFO)

    # Create the scenario instance
    mux_problem = MUXProblem(training_cycles=14000, address_size=2)

    # Wrap the scenario instance in an observer so progress gets logged,
    mux_scenario = MUXScenarioObserver(mux_problem, feedback_dir="/tmp/luis/tests")
    # mux_scenario = ScenarioObserver(mux_problem)
    # and pass it on to the test() function.
    # xcs.test(scenario=bowling_scenario)
    # or do the whole thing in a detailed way:

    algorithm = XCSAlgorithm()
    # parameters as of original paper:
    algorithm.max_population_size = 800          # N
    algorithm.learning_rate = .2                # beta
    algorithm.accuracy_coefficient = .1          # alpha # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.error_threshold = 10              # epsilon_0
    algorithm.accuracy_power = 5                 # nu # TODO: is this 'n'?
    algorithm.ga_threshold = 12                  # theta_GA
    algorithm.crossover_probability = .8        # chi
    algorithm.mutation_probability = .04         # mu
    algorithm.deletion_threshold = 20            # theta_del # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.fitness_threshold = .1             # delta # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.subsumption_threshold = 20         # theta_sub # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.initial_prediction = 10        # p_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.initial_error = 0             # epsilon_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.initial_fitness = .01           # F_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.minimum_actions = 2             # theta_mna # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # algorithm.error_threshold = 1.0              # epsilon_0  # TODO: is this s_0 ?
    # we want that, on average, we present exploitation and exploration problems:
    algorithm.exploration_probability = .5       # p_exp

    # these parameters come from page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf:
    algorithm.do_ga_subsumption = True
    algorithm.do_action_set_subsumption = False


    # discount_factor = .71              # gamma
    # fitness_threshold = .1             # delta
    # subsumption_threshold = 20         # theta_sub
    # wildcard_probability = .33         # P_#

    # Create the classifier system from the algorithm.
    model = ClassifierSet(algorithm, mux_scenario.get_possible_actions())
    start_time = time.time()
    model.run(mux_scenario, learn=True)
    end_time = time.time()

    # logger.info('Classifiers:\n\n%s\n', model)
    logger.info("Total time: %.5f seconds", end_time - start_time)

    # return (
    #     scenario.steps,
    #     scenario.total_reward,
    #     end_time - start_time,
    #     model
    # )
