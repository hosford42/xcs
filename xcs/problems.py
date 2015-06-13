# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# xcs
# ---
# Accuracy-based Classifier Systems for Python 3
#
# http://hosford42.github.io/xcs/
#
# (c) Aaron Hosford 2015, all rights reserved
# Revised (3 Clause) BSD License
#
# Implements the XCS (Accuracy-based Classifier System) algorithm,
# as described in the 2001 paper, "An Algorithmic Description of XCS,"
# by Martin Butz and Stewart Wilson.
#
# -------------------------------------------------------------------------------

"""
xcs.problems
Description: The problem interface and a selection of predefined problem classes and wrappers.
"""

__author__ = 'Aaron Hosford'

__all__ = [
    'OnLineProblem',
    'MUXProblem',
    'HaystackProblem',
    'OnLineObserver',
    'ClassifiedDataAsOnLineProblem',
    'PredictionDataAsOnLineProblem',
]

import logging
import random
from abc import ABCMeta, abstractmethod

from . import numpy
from . import bitstrings


class OnLineProblem(metaclass=ABCMeta):
    """Abstract interface for on-line problems accepted by XCS. To create a new problem to which XCS can be applied,
    subclass OnLineProblem and implement the methods defined here. See MUXProblem for an example."""

    @abstractmethod
    def get_possible_actions(self):
        """Return a sequence containing the possible actions that can be executed within the environment."""
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """Reset the problem, starting it over for a new run."""
        raise NotImplementedError()

    @abstractmethod
    def sense(self):
        """Return a situation, encoded as a bit string, which represents the observable state of the environment."""
        raise NotImplementedError()

    @abstractmethod
    def execute(self, action):
        """Execute the indicated action(s) within the environment and return the resulting immediate reward dictated by
        the reward program."""
        raise NotImplementedError()

    @abstractmethod
    def more(self):
        """Return a Boolean indicating whether additional actions may be executed, per the reward program."""
        raise NotImplementedError()


class MUXProblem(OnLineProblem):
    """Classic multiplexer problem. This problem is static; each action affects only the immediate reward and the
    environment is stateless. The address size indicates the number of bits used as an address/index into the remaining
    bits in the situations returned by sense(). The agent is expected to return the value of the indexed bit from the
    situation."""

    def __init__(self, training_cycles=10000, address_size=3):

        assert isinstance(training_cycles, int) and training_cycles > 0
        assert isinstance(address_size, int) and address_size > 0

        self.address_size = address_size
        self.current_situation = None
        self.possible_actions = (True, False)
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles

    def get_possible_actions(self):
        """Return a sequence containing the possible actions that can be executed within the environment."""
        return self.possible_actions

    def reset(self):
        """Reset the problem, starting it over for a new run."""
        self.remaining_cycles = self.initial_training_cycles

    def sense(self):
        """Return a situation, encoded as a bit string, which represents the observable state of the environment."""
        self.current_situation = bitstrings.BitString([
            random.randrange(2)
            for _ in range(self.address_size + (1 << self.address_size))
        ])
        return self.current_situation

    def execute(self, action):
        """Execute the indicated action(s) within the environment and return the resulting immediate reward dictated by
        the reward program."""

        assert action in self.possible_actions

        self.remaining_cycles -= 1
        index = int(bitstrings.BitString(self.current_situation[:self.address_size]))
        bit = self.current_situation[self.address_size + index]
        return action == bit

    def more(self):
        """Return a Boolean indicating whether additional actions may be executed, per the reward program."""
        return int(self.remaining_cycles > 0)


class HaystackProblem(OnLineProblem):
    """This is the problem that appears in the tutorial in the section, "Defining a new problem type". This problem is
     designed to test the algorithm's ability to find a single important input bit (the "needle") from among a large
     number of irrelevant input bits (the "haystack")."""

    def __init__(self, training_cycles=10000, input_size=500):

        assert isinstance(training_cycles, int) and training_cycles > 0
        assert isinstance(input_size, int) and input_size > 0

        self.input_size = input_size
        self.possible_actions = (True, False)
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles
        self.needle_index = random.randrange(input_size)
        self.needle_value = None

    def get_possible_actions(self):
        """Return a sequence containing the possible actions that can be executed within the environment."""
        return self.possible_actions

    def reset(self):
        """Reset the problem, starting it over for a new run."""
        self.remaining_cycles = self.initial_training_cycles
        self.needle_index = random.randrange(self.input_size)

    def sense(self):
        """Return a situation, encoded as a bit string, which represents the observable state of the environment."""
        haystack = bitstrings.BitString.random(self.input_size)
        self.needle_value = haystack[self.needle_index]
        return haystack

    def execute(self, action):
        """Execute the indicated action within the environment and return the resulting immediate reward dictated by the
        reward program."""

        assert action in self.possible_actions

        self.remaining_cycles -= 1
        return action == self.needle_value

    def more(self):
        """Return a Boolean indicating whether additional actions may be executed, per the reward program."""
        return self.remaining_cycles > 0


class OnLineObserver(OnLineProblem):
    """Wrapper for other OnLineProblem instances which prints details of the agent/problem interaction as they take
    place, forwarding the actual work on to the wrapped instance."""

    def __init__(self, wrapped):
        # Ensure that the wrapped object implements the same interface
        assert isinstance(wrapped, OnLineProblem)

        self.logger = logging.getLogger(__name__)
        self.wrapped = wrapped
        self.total_reward = 0
        self.steps = 0

    def get_possible_actions(self):
        """Return a sequence containing the possible actions that can be executed within the environment."""
        possible_actions = self.wrapped.get_possible_actions()

        # Try to ensure that the possible actions are unique. Also, put them into a list so we can iterate over them
        # safely before returning them; this avoids accidentally exhausting an iterator, if the wrapped class happens
        # to return one.
        try:
            possible_actions = list(set(possible_actions))
        except TypeError:
            possible_actions = list(possible_actions)

        try:
            possible_actions.sort()
        except TypeError:
            pass

        self.logger.info('Possible actions:')
        for action in possible_actions:
            self.logger.info('    %s', action)

        return possible_actions

    def reset(self):
        """Reset the problem, starting it over for a new run."""
        self.logger.info('Resetting problem.')
        self.wrapped.reset()

    def sense(self):
        """Return a situation, encoded as a bit string, which represents the observable state of the environment."""
        situation = self.wrapped.sense()

        self.logger.debug('Situation: %s', situation)

        return situation

    def execute(self, action):
        """Execute the indicated action within the environment and return the resulting immediate reward dictated by the
        reward program."""

        self.logger.debug('Executing action: %s', action)

        reward = self.wrapped.execute(action)
        if reward:
            self.total_reward += reward
        self.steps += 1

        self.logger.debug('Reward received on this step: %.5f', reward)
        self.logger.debug('Average reward per step: %.5f', self.total_reward / self.steps)

        return reward

    def more(self):
        """Return a Boolean indicating whether additional actions may be executed, per the reward program."""
        more = self.wrapped.more()

        if not self.steps % 100:
            self.logger.info('Steps completed: %d', self.steps)
            self.logger.info('Average reward per step: %.5f', self.total_reward / (self.steps or 1))
        if not more:
            self.logger.info('Run completed.')
            self.logger.info('Total steps: %d', self.steps)
            self.logger.info('Total reward received: %.5f', self.total_reward)
            self.logger.info('Average reward per step: %.5f', self.total_reward / (self.steps or 1))

        return more


class ClassifiedDataAsOnLineProblem(OnLineProblem):
    """Wrap off-line (non-interactive) training/test data as an on-line (interactive) problem."""

    def __init__(self, situations, classifications, reward_function=None):
        if bitstrings.using_numpy() and isinstance(situations, numpy.ndarray):
            self.situations = []
            for situation_bits in situations:
                # This doesn't affect the original situations array.
                situation_bits.setflags(write=False)
                situation = bitstrings.BitString(situation_bits)
                self.situations.append(situation)
        else:
            self.situations = [bitstrings.BitString(situation_bits) for situation_bits in situations]

        if isinstance(classifications, (list, tuple)) or (bitstrings.using_numpy() and
                                                          isinstance(classifications, numpy.ndarray)):
            self.classifications = classifications
        else:
            self.classifications = list(classifications)

        assert len(self.situations) == len(self.classifications)

        self.reward_function = reward_function or (lambda action, target: float(action == target))
        self.possible_actions = set(self.classifications)
        self.steps = 0
        self.total_reward = 0

    def get_possible_actions(self):
        """Return a sequence containing the possible actions that can be executed within the environment."""
        return self.possible_actions

    def reset(self):
        """Reset the problem, starting it over for a new run."""
        self.steps = 0
        self.total_reward = 0

    def sense(self):
        """Return a situation, encoded as a bit string, which represents the observable state of the environment."""
        return self.situations[self.steps]

    def execute(self, action):
        """Execute the indicated action within the environment and return the resulting immediate reward dictated by the
        reward program."""
        reward = self.reward_function(action, self.classifications[self.steps])
        self.total_reward += reward
        self.steps += 1
        return reward

    def more(self):
        """Return a Boolean indicating whether additional actions may be executed, per the reward program."""
        return self.steps < len(self.situations)


class PredictionDataAsOnLineProblem(OnLineProblem):
    """Wrap off-line (non-interactive) prediction data as an on-line (interactive) problem."""

    def __init__(self, situations, possible_actions):
        if bitstrings.using_numpy() and isinstance(situations, numpy.ndarray):
            self.situations = []
            for situation_bits in situations:
                # This doesn't affect the original situations array.
                situation_bits.setflags(write=False)
                situation = bitstrings.BitString(situation_bits)
                self.situations.append(situation)
        else:
            self.situations = [bitstrings.BitString(situation_bits) for situation_bits in situations]

        self.possible_actions = frozenset(possible_actions)
        self.steps = 0
        self.classifications = []

    def get_possible_actions(self):
        """Return a sequence containing the possible actions that can be executed within the environment."""
        return self.possible_actions

    def reset(self):
        """Reset the problem, starting it over for a new run."""
        self.steps = 0
        self.classifications.clear()

    def sense(self):
        """Return a situation, encoded as a bit string, which represents the observable state of the environment."""
        return self.situations[self.steps]

    def execute(self, action):
        """Execute the indicated action within the environment and return the resulting immediate reward dictated by the
        reward program."""
        self.classifications.append(action)
        self.steps += 1
        return None  # This problem is not meant to be used for learning.

    def more(self):
        """Return a Boolean indicating whether additional actions may be executed, per the reward program."""
        return self.steps < len(self.situations)

    def get_classifications(self):
        """Return the classifications made by the algorithm for this problem."""
        if bitstrings.using_numpy():
            return numpy.array(self.classifications)
        else:
            return self.classifications
