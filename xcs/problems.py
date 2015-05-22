# -------------------------------------------------------------------------------
# Name:     problems
# Purpose:  Implements the problem interface, used by problems the XCS algorithm
#           knows how to drive.
#
# Author:       Aaron Hosford
#
# Created:      5/5/2015
# Copyright:    (c) Aaron Hosford 2015, all rights reserved
# Licence:      Revised (3 Clause) BSD License
# -------------------------------------------------------------------------------

"""
xcs/problems.py
(c) Aaron Hosford 2015, all rights reserved
Revised BSD License

Implements the problem interface, used by problems the XCS algorithm knows how to drive.

This file is part of the public API of the xcs package.
"""

__author__ = 'Aaron Hosford'
__all__ = [
    'OnLineProblem',
    'MUXProblem',
    'ObservedOnLineProblem',
]

import random
from abc import ABCMeta, abstractmethod

# noinspection PyUnresolvedReferences
from xcs.bitstrings import BitString, BitCondition


class OnLineProblem(metaclass=ABCMeta):
    """Abstract interface for on-line problems accepted by XCS. To create a new problem to which XCS can be applied,
    subclass OnLineProblem and implement the methods defined here. See MUXProblem for an example."""

    @abstractmethod
    def get_possible_actions(self):
        """Return a sequence containing the possible actions that can be executed within the environment."""
        raise NotImplementedError()

    @abstractmethod
    def sense(self):
        """Return a situation, encoded as a bit string, which represents the observable state of the environment."""
        raise NotImplementedError()

    @abstractmethod
    def execute(self, action):
        """Execute the indicated action within the environment and return the resulting immediate reward dictated by the
        reward program."""
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

    def __init__(self, training_cycles=1000, address_size=3):
        self.address_size = address_size
        self.current_situation = None
        self.possible_actions = (True, False)
        self.remaining_cycles = training_cycles

    def get_possible_actions(self):
        """Return a sequence containing the possible actions that can be executed within the environment."""
        return self.possible_actions

    def sense(self):
        """Return a situation, encoded as a bit string, which represents the observable state of the environment."""
        self.current_situation = BitString([
            random.randrange(2)
            for _ in range(self.address_size + (1 << self.address_size))
        ])
        return self.current_situation

    def execute(self, action):
        """Execute the indicated action within the environment and return the resulting immediate reward dictated by the
        reward program."""
        self.remaining_cycles -= 1
        index = int(BitString(self.current_situation[:self.address_size]))
        bit = self.current_situation[self.address_size + index]
        return action == bit

    def more(self):
        """Return a Boolean indicating whether additional actions may be executed, per the reward program."""
        return int(self.remaining_cycles > 0)


# TODO: Use logging instead of print. This has the dual advantage of making the code more Python 2.x friendly, if
#       anyone down the road decides to make it compatible, and of giving the client greater control over reporting.
class ObservedOnLineProblem(OnLineProblem):
    """Wrapper for other OnLineProblem instances which prints details of the agent/problem interaction as they take
    place, forwarding the actual work on to the wrapped instance."""

    def __init__(self, wrapped):
        # Ensure that the wrapped object implements the same interface
        if not isinstance(wrapped, OnLineProblem):
            raise TypeError(wrapped)

        self.wrapped = wrapped
        self.reward = 0
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

        print()
        print('Possible actions:')
        for action in possible_actions:
            print('    ' + str(action))
        print()

        return possible_actions

    def sense(self):
        """Return a situation, encoded as a bit string, which represents the observable state of the environment."""
        situation = self.wrapped.sense()

        print()
        print('Situation:', situation)
        print()

        return situation

    def execute(self, action):
        """Execute the indicated action within the environment and return the resulting immediate reward dictated by the
        reward program."""
        print()
        print("Executing action:", action)

        reward = self.wrapped.execute(action)
        self.reward += reward
        self.steps += 1

        print("Reward:", reward)
        print("Average reward:", self.reward / self.steps)
        print()

        return reward

    def more(self):
        """Return a Boolean indicating whether additional actions may be executed, per the reward program."""
        more = self.wrapped.more()

        print()
        print("Steps:", self.steps)
        print("Next iteration" if more else "Terminated")
        print()

        return more
