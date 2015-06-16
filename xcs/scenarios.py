# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
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
# -------------------------------------------------------------------------

"""
Accuracy-based Classifier Systems for Python 3

This xcs submodule provides the scenario interface and a selection of
predefined scenarios and wrappers. Most widely used machine learning
algorithms operate on static, non-interactive data sets which have already
been collected, building a model from one data set and then using that
model to assess a new data set drawn from the same data source, predicting
unknown values from known ones. Unlike other machine learning algorithms,
reinforcement learning algorithms, of which XCS is one example, are capable
of dealing with not only non-interactive data, but interactive processes in
which classification decisions must be made on the fly that can potentially
affect which data will be gathered and classified in the future.

As a consequence, training a reinforcement learning algorithm such as XCS
is not always as simple as gathering a lot of data into a table or file and
then running the algorithm to build a model. The interactivity between the
algorithm and the environment from which the data is being collected must
be accounted for; this is the role of the Scenario class. The Scenario
class is an abstract base class that is designed to provide a standardized
interface through which reinforcement learning algorithms can assess the
environment and act upon it, building up models as they proceed.

If you wish to create a scenario of your own, subclass the Scenario class
and define the appropriate methods. To add logging to an existing scenario,
wrap your scenario in a ScenarioObserver. To treat non-interactive, pre-
collected data as a scenario, use PreClassifiedData for training and
testing, or UnclassifiedData for prediction. To get a full listing of the
classes provided by this module and see documentation on their appropriate
usage, use "help(xcs.scenarios)".




Copyright (c) 2015, Aaron Hosford
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of xcs nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

__author__ = 'Aaron Hosford'

__all__ = [
    'HaystackProblem',
    'MUXProblem',
    'Scenario',
    'ScenarioObserver',
    'PreClassifiedData',
    'UnclassifiedData',
]

import logging
import random
from abc import ABCMeta, abstractmethod

from . import numpy
from . import bitstrings


class Scenario(metaclass=ABCMeta):
    """Abstract interface for scenarios accepted by LCSAlgorithms. To
    create a new scenario to which an LCS algorithm like XCS can be
    applied, subclass Scenario and implement the methods defined here. See
    MUXProblem or HaystackProblem for examples.

    Usage:
        This is an abstract base class; it cannot be instantiated directly.
        You must create a subclass that defines the problem you expect the
        algorithm to solve, and instantiate that subclass instead.

    Init Arguments: n/a (See appropriate subclass.)
    """

    @property
    @abstractmethod
    def is_dynamic(self):
        """A Boolean value indicating whether earlier actions from the same
        run can affect the rewards or outcomes of later actions."""
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """Reset the scenario, starting it over for a new run.

        Usage:
            if not scenario.more():
                scenario.reset()

        Arguments: None
        Return: None
        """
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()


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

    def __init__(self, training_cycles=10000, address_size=3):
        assert isinstance(training_cycles, int) and training_cycles > 0
        assert isinstance(address_size, int) and address_size > 0

        self.address_size = address_size
        self.current_situation = None
        self.possible_actions = (True, False)
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles

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
        self.current_situation = bitstrings.BitString([
            random.randrange(2)
            for _ in range(self.address_size + (1 << self.address_size))
        ])
        return self.current_situation

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
        index = int(bitstrings.BitString(
            self.current_situation[:self.address_size]
        ))
        bit = self.current_situation[self.address_size + index]
        return action == bit

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
        return int(self.remaining_cycles > 0)


class HaystackProblem(Scenario):
    """This is the scenario that appears in the tutorial in the section,
    "Defining a new scenario". This scenario is designed to test the
    algorithm's ability to find a single important input bit (the "needle")
    from among a large number of irrelevant input bits (the "haystack").

    Usage:
        scenario = HaystackProblem()
        model = algorithm.run(scenario)

    Init Arguments:
        training_cycles: An int, the number of situations to produce;
            default is 10,000.
        input_size: An int, the number of bits in each situation.
    """

    def __init__(self, training_cycles=10000, input_size=500):

        assert isinstance(training_cycles, int) and training_cycles > 0
        assert isinstance(input_size, int) and input_size > 0

        self.input_size = input_size
        self.possible_actions = (True, False)
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles
        self.needle_index = random.randrange(input_size)
        self.needle_value = None

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
        self.remaining_cycles = self.initial_training_cycles
        self.needle_index = random.randrange(self.input_size)

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
        haystack = bitstrings.BitString.random(self.input_size)
        self.needle_value = haystack[self.needle_index]
        return haystack

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
        return action == self.needle_value

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


class ScenarioObserver(Scenario):
    """Wrapper for other Scenario instances which logs details of the
    agent/scenario interaction as they take place, forwarding the actual
    work on to the wrapped instance.

    Usage:
        model = algorithm.run(ScenarioObserver(scenario))

    Input Args:
        wrapped: The Scenario instance to be observed.
    """

    def __init__(self, wrapped):
        # Ensure that the wrapped object implements the same interface
        assert isinstance(wrapped, Scenario)

        self.logger = logging.getLogger(__name__)
        self.wrapped = wrapped
        self.total_reward = 0
        self.steps = 0

    @property
    def is_dynamic(self):
        """A Boolean value indicating whether earlier actions from the same
        run can affect the rewards or outcomes of later actions."""
        return self.wrapped.is_dynamic

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
        possible_actions = self.wrapped.get_possible_actions()

        if len(possible_actions) <= 20:
            # Try to ensure that the possible actions are unique. Also, put
            # them into a list so we can iterate over them safely before
            # returning them; this avoids accidentally exhausting an
            # iterator, if the wrapped class happens to return one.
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
        else:
            self.logger.info("%d possible actions.", len(possible_actions))

        return possible_actions

    def reset(self):
        """Reset the scenario, starting it over for a new run.

        Usage:
            if not scenario.more():
                scenario.reset()

        Arguments: None
        Return: None
        """
        self.logger.info('Resetting scenario.')
        self.wrapped.reset()

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
        situation = self.wrapped.sense()

        self.logger.debug('Situation: %s', situation)

        return situation

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

        self.logger.debug('Executing action: %s', action)

        reward = self.wrapped.execute(action)
        if reward:
            self.total_reward += reward
        self.steps += 1

        self.logger.debug('Reward received on this step: %.5f',
                          reward or 0)
        self.logger.debug('Average reward per step: %.5f',
                          self.total_reward / self.steps)

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
        more = self.wrapped.more()

        if not self.steps % 100:
            self.logger.info('Steps completed: %d', self.steps)
            self.logger.info('Average reward per step: %.5f',
                             self.total_reward / (self.steps or 1))
        if not more:
            self.logger.info('Run completed.')
            self.logger.info('Total steps: %d', self.steps)
            self.logger.info('Total reward received: %.5f',
                             self.total_reward)
            self.logger.info('Average reward per step: %.5f',
                             self.total_reward / (self.steps or 1))

        return more


class PreClassifiedData(Scenario):
    """Wrap off-line (non-interactive) training/test data as an on-line
    (interactive) scenario.

    Simple Usage:
        X = features
        y = classifications
        assert len(X) == len(y)
        scenario = PreClassifiedData(X, y)
        model = algorithm.run(scenario)

    Alternate Usage:
        def grade_classification(actual, target):
            '''Return a floating point value in the range [0, 1] that
            indicates how desirable the actual classification was in
            terms of how it compares to the correct classification.'''

            # In this case we are dealing with a binary classification
            # problem, where the classification targets (possible actions)
            # are either True or False and false negatives are highly
            # undesirable.
            if actual == target:
                # Correct classifications get maximum reward.
                return 1.0
            if actual:
                # False positives get lower but not minimal reward; these
                # types of errors are not good, but they are better than
                # false negatives for this problem.
                return .25
            else:
                # False negatives are really expensive to us so give them
                # the lowest reward level.
                return 0.0

        X = features
        y = classifications
        assert len(X) == len(y)
        scenario = PreClassifiedData(
            X,
            y,
            reward_function=grade_classification
        )
        model = algorithm.run(scenario)

    Init Arguments:
        situations: An iterable sequence containing the situations.
        classifications: An iterable sequence containing the correct
            action for each situation, appearing in the same order as the
            situations.
        reward_function: None, or a function of two arguments, the actual
            and target actions, which returns a float value indicating the
            reward that should be received for the actual action. The
            default is None, which causes the reward to be 1.0 when the
            actual is the target and 0.0 otherwise.
    """

    def __init__(self, situations, classifications, reward_function=None):
        if (bitstrings.using_numpy() and
                isinstance(situations, numpy.ndarray)):
            self.situations = []
            for situation_bits in situations:
                # This doesn't affect the original situations array.
                situation_bits.setflags(write=False)
                situation = bitstrings.BitString(situation_bits)
                self.situations.append(situation)
        else:
            self.situations = [
                bitstrings.BitString(situation_bits)
                for situation_bits in situations
            ]

        if (isinstance(classifications, (list, tuple)) or
                (bitstrings.using_numpy() and
                 isinstance(classifications, numpy.ndarray))):
            self.classifications = classifications
        else:
            self.classifications = list(classifications)

        assert len(self.situations) == len(self.classifications)

        self.reward_function = (
            reward_function or
            (lambda actual, target: float(actual == target))
        )
        self.possible_actions = set(self.classifications)
        self.steps = 0
        self.total_reward = 0

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
        self.steps = 0
        self.total_reward = 0

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
        return self.situations[self.steps]

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
        reward = self.reward_function(
            action,
            self.classifications[self.steps]
        )
        self.total_reward += reward
        self.steps += 1
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
        return self.steps < len(self.situations)


class UnclassifiedData(Scenario):
    """Wrap off-line (non-interactive) prediction data as an on-line
    (interactive) scenario.

    Usage:
        X = features
        scenario = UnclassifiedData(X)
        model.run(scenario, learn=False)
        y = scenario.get_classifications()

    Init Arguments:
        situations: An iterable sequence of situations for which actions
            (classifications) are desired.
        possible_actions: None, or the possible actions (classifications)
            which the model might generate.
    """

    def __init__(self, situations, possible_actions=None):
        if (bitstrings.using_numpy() and
                isinstance(situations, numpy.ndarray)):
            self.situations = []
            for situation_bits in situations:
                # This doesn't affect the original situations array.
                situation_bits.setflags(write=False)
                situation = bitstrings.BitString(situation_bits)
                self.situations.append(situation)
        else:
            self.situations = [
                bitstrings.BitString(situation_bits)
                for situation_bits in situations
            ]

        if possible_actions is None:
            self.possible_actions = None
        else:
            self.possible_actions = frozenset(possible_actions)
        self.steps = 0
        self.classifications = []

    @property
    def is_dynamic(self):
        """A Boolean value indicating whether earlier actions from the same
        run can affect the rewards or outcomes of later actions."""
        return False

    def get_possible_actions(self):
        """Return a sequence containing the possible actions that can be
        executed within the environment. Raise a ValueError if possible
        actions were not provided at initialization.

        Usage:
            possible_actions = scenario.get_possible_actions()

        Arguments: None
        Return:
            A sequence containing the possible actions which can be
            executed within this scenario.
        """
        # UnclassifiedData instances are typically not asked for possible
        # actions because they are used with pre-existing models that
        # already know the possible actions; thus for this one case we
        # make an exception... literally.
        if self.possible_actions is None:
            raise ValueError("Possible actions were not provided.")
        return self.possible_actions

    def reset(self):
        """Reset the scenario, starting it over for a new run.

        Usage:
            if not scenario.more():
                scenario.reset()

        Arguments: None
        Return: None
        """
        self.steps = 0
        self.classifications.clear()

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
        return self.situations[self.steps]

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
        self.classifications.append(action)
        self.steps += 1
        return None  # This scenario is not meant to be used for learning.

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
        return self.steps < len(self.situations)

    def get_classifications(self):
        """Return the classifications made by the algorithm for this
        scenario.

        Usage:
            model.run(scenario, learn=False)
            classifications = scenario.get_classifications()

        Arguments: None
        Return:
            An indexable sequence containing the classifications made by
            the model for each situation, in the same order as the original
            situations themselves appear.
        """
        if bitstrings.using_numpy():
            return numpy.array(self.classifications)
        else:
            return self.classifications
