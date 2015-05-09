#!/usr/bin/env python
# -------------------------------------------------------------------------------
# Name:     xcs.py
# Purpose:  Implements the XCS (Accuracy-based Classifier System) algorithm,
#           roughly according to the description provided in the paper, "An
#           Algorithmic Description of XCS," by Martin Butz and Stewart Wilson.
#
# Butz, M. and Wilson, S. (2001). An algorithmic description of XCS. In Lanzi, P., 
#   Stolzmann, W., and Wilson, S., editors, Advances in Learning Classifier Systems:
#   Proceedings of the Third International Workshop, volume 1996 of Lecture Notes in 
#   Artificial Intelligence, pages 253–272. Springer-Verlag Berlin Heidelberg.
#
# Author:       Aaron Hosford
#
# Created:      5/5/2015
# Copyright:    (c) Aaron Hosford 2015, all rights reserved
# Licence:      Revised (3 Clause) BSD License
# -------------------------------------------------------------------------------

"""
xcs.py
(c) Aaron Hosford 2015, all rights reserved
Revised BSD License

Implements the XCS (Accuracy-based Classifier System) algorithm,
roughly according to the description provided in the paper, "An
Algorithmic Description of XCS," by Martin Butz and Stewart Wilson.

Butz, M. and Wilson, S. (2001). An algorithmic description of XCS. In Lanzi, P.,
    Stolzmann, W., and Wilson, S., editors, Advances in Learning Classifier Systems:
    Proceedings of the Third International Workshop, volume 1996 of Lecture Notes in
    Artificial Intelligence, pages 253–272. Springer-Verlag Berlin Heidelberg.
"""

__author__ = 'Aaron Hosford'

import numpy
import random
from abc import ABCMeta, abstractmethod


class BitString:
    """A hashable, immutable sequence of bits (Boolean values).

    In addition to operations for indexing and iteration, implements standard bitwise operations, including & (bitwise
    and), | (bitwise or), ^ (bitwise xor), and ~ (bitwise not). Also implements the + operator, which acts like string
    concatenation.

    A bit string can also be cast as an integer or an ordinary string.
    """

    @classmethod
    def from_int(cls, value, length=None):
        """Create a bit string from an integer value. If the length parameter is provided, it determines the number of
        bits in the bit string. Otherwise, the minimum length required to represent the value is used."""

        bits = []
        while value:
            if length is not None and len(bits) >= length:
                break
            bits.append(value % 2)
            value >>= 1

        if length:
            if len(bits) < length:
                bits.extend([0] * (length - len(bits)))
            elif len(bits) > length:
                bits = bits[:length]

        bits.reverse()

        return cls(bits)

    def __init__(self, bits):
        if isinstance(bits, numpy.ndarray) and bits.dtype == numpy.bool:
            if bits.flags.writeable:
                self._bits = bits.copy()
                self._bits.writeable = False
            else:
                self._bits = bits
        elif isinstance(bits, int):
            self._bits = numpy.zeros(bits, bool)
            self._bits.flags.writeable = False
        elif isinstance(bits, BitString):
            # No need to make a copy because we use immutable bit arrays
            self._bits = bits._bits
        else:
            self._bits = numpy.array(bits, bool)
            self._bits.flags.writeable = False

        self._hash = None

    @property
    def bits(self):
        """The numpy array containing the actual bits of the bit string. Note that the array is immutable."""
        # Safe because we use immutable bit arrays
        return self._bits

    def __str__(self):
        return ''.join('1' if bit else '0' for bit in self._bits)

    def __repr__(self):
        return type(self).__name__ + '(' + repr([int(bit) for bit in self._bits]) + ')'

    def __int__(self):
        value = 0
        for bit in self._bits:
            value <<= 1
            value += int(bit)
        return value

    def __len__(self):
        return len(self._bits)

    def __iter__(self):
        return iter(self._bits)

    def __getitem__(self, index):
        return self._bits[index]

    def __hash__(self):
        if self._hash is None:
            self._hash = len(self._bits) + (hash(int(self)) << 13)
        return self._hash

    def __eq__(self, other):
        return isinstance(other, BitString) and numpy.array_equal(self._bits, other._bits)

    def __ne__(self, other):
        return not self == other

    def __and__(self, other):
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = numpy.bitwise_and(self._bits, other._bits)
        bits.flags.writeable = False
        return type(self)(bits)

    def __or__(self, other):
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = numpy.bitwise_or(self._bits, other._bits)
        bits.flags.writeable = False
        return type(self)(bits)

    def __xor__(self, other):
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = numpy.bitwise_xor(self._bits, other._bits)
        bits.flags.writeable = False
        return type(self)(bits)

    def __invert__(self):
        bits = numpy.bitwise_not(self._bits)
        bits.flags.writeable = False
        return type(self)(bits)

    def __add__(self, other):
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = numpy.concatenate((self._bits, other._bits))
        bits.flags.writeable = False
        return type(self)(bits)


class BitCondition:
    """A pair of bit strings, one indicating the bit values, and the other indicating the bit mask, which together act
    as a matching template for bit strings. Like bit strings, bit conditions are hashable and immutable."""

    @classmethod
    def cover(cls, bits, wildcard_probability):
        """Create a new bit condition that matches the provided bit string, with the indicated per-index wildcard
         probability."""

        if not isinstance(bits, BitString):
            bits = BitString(bits)

        mask = BitString([random.random() < wildcard_probability for _ in range(len(bits))])
        return cls(bits, mask)

    def __init__(self, bits, mask):
        if not isinstance(bits, BitString):
            bits = BitString(bits)

        if isinstance(mask, int):
            mask = BitString.from_int(mask, len(bits))
        elif not isinstance(mask, BitString):
            mask = BitString(mask)

        if len(bits) != len(mask):
            raise ValueError("Length mismatch between bits and mask")

        self._bits = bits & mask
        self._mask = mask
        self._hash = None

    @property
    def bits(self):
        """The bit string indicating the bit values of this bit condition. Indices that are wildcarded will have a
        value of False."""
        return self._bits

    @property
    def mask(self):
        """The bit string indicating the bit mask. A value of True for a bit indicates it must match the value bit
        string. A value of False indicates it is masked/wildcarded."""
        return self._mask

    def __str__(self):
        return ''.join('1' if bit else ('#' if bit is None else '0') for bit in self)

    def __repr__(self):
        return type(self).__name__ + repr((self._bits, self._mask))

    def __len__(self):
        return len(self._bits)

    def __iter__(self):
        for bit, mask in zip(self._bits, self._mask):
            yield bit if mask else None

    def __getitem__(self, index):
        return self._bits[index] if self._mask[index] else None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(tuple(self))
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, BitCondition) or len(self._bits) != len(other._bits):
            return False
        return self._bits == other._bits and self._mask == other._mask

    def __ne__(self, other):
        return not self == other

    def __and__(self, other):
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)((self._bits | ~self._mask) & (other._bits | ~other._mask), self._mask | other._mask)

    def __or__(self, other):
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)(self._bits | other._bits, self._mask & other._mask)

    def __xor__(self, other):
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)(self._bits ^ other._bits, self._mask & other._mask)

    def __invert__(self):
        return type(self)(~self._bits, self._mask)

    def __add__(self, other):
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)(self._bits + other._bits, self._mask + other._mask)

    def __floordiv__(self, other):
        if isinstance(other, BitCondition):
            return ((self._bits ^ other._bits) | ~other._mask) & self._mask

        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)

        return (self._bits ^ other) & self._mask

    def __call__(self, other):
        mismatches = self // other
        return not mismatches.bits.any()

    def crossover_with(self, other):
        """Perform 2-point crossover on this bit condition and another of the same length, returning the two resulting
        children."""

        if not isinstance(other, BitCondition):
            raise TypeError(other)
        if len(self) != len(other):
            raise ValueError(other)

        # TODO: Revamp this to take advantage of numpy array speeds

        point1 = random.randrange(len(self._bits))
        point2 = random.randrange(len(self._bits))
        if point1 > point2:
            point1, point2 = point2, point1

        bits1 = list(self._bits)
        bits2 = list(other._bits)

        mask1 = list(self._mask)
        mask2 = list(other._mask)

        for index in range(point1, point2):
            bits1[index], bits2[index] = bits2[index], bits1[index]
            mask1[index], mask2[index] = mask2[index], mask1[index]

        return type(self)(bits1, mask1), type(self)(bits2, mask2)


class RuleMetadata:
    """Metadata used by the XCS algorithm to track the rules (classifiers) in a population."""

    def __init__(self, time_stamp, parameters):
        self.time_stamp = time_stamp
        self.prediction = parameters.initial_prediction
        self.error = parameters.initial_error
        self.fitness = parameters.initial_fitness
        self.experience = 0
        self.action_set_size = 1
        self.numerosity = 1

    def __iadd__(self, other):
        self.numerosity += other.numerosity
        return self


class ActionSet:
    """A set of rules (classifiers) with the same action that matched the same situation."""

    def __init__(self, action, rules, population):
        self._action = action
        self._rules = rules  # {condition : metadata}
        self._prediction = None
        self._population = population
        self._parameters = population.parameters

    @property
    def action(self):
        """The action shared by the matching rules."""
        return self._action

    @property
    def population(self):
        """The population from which the action set was taken."""
        return self._population

    @property
    def prediction(self):
        """The predicted payoff for this action set."""
        if self._prediction is None:
            total_fitness = 0
            total_prediction = 0
            for metadata in self._rules.values():
                total_fitness += metadata.fitness
                total_prediction += metadata.prediction * metadata.fitness
            self._prediction = total_prediction / (total_fitness or 1)
        return self._prediction

    def _update_fitness(self):
        """Update the fitness of the rules belonging to this action set."""
        total_accuracy = 0
        accuracies = {}
        for condition, metadata in self._rules.items():
            if metadata.error < self._parameters.error_threshold:
                accuracy = 1
            else:
                accuracy = (
                    self._parameters.accuracy_coefficient *
                    (metadata.error / self._parameters.error_threshold) ** -self._parameters.accuracy_power
                )
            accuracies[condition] = accuracy
            total_accuracy += accuracy * metadata.numerosity
        for condition, metadata in self._rules.items():
            accuracy = accuracies[condition]
            metadata.fitness += (
                self._parameters.learning_rate *
                (accuracy * metadata.numerosity / total_accuracy - metadata.fitness)
            )

    def update(self, payoff):
        """Update the rule metadata for the rules belonging to this action set, based on the payoff received."""
        action_set_size = sum(metadata.numerosity for metadata in self._rules.values())
        for metadata in self._rules.values():
            metadata.experience += 1

            update_rate = max(self._parameters.learning_rate, 1 / metadata.experience)

            metadata.prediction += (payoff - metadata.prediction) * update_rate
            metadata.error += (abs(payoff - metadata.prediction) - metadata.error) * update_rate
            metadata.action_set_size += (action_set_size - metadata.action_set_size) * update_rate
        self._update_fitness()
        if self._parameters.do_action_set_subsumption:
            pass  # TODO: DO ACTION SET SUBSUMPTION

    def get_average_time_stamp(self):
        """Return the average time stamp for the rules in this action set."""
        return (
            sum(metadata.time_stamp * metadata.numerosity for metadata in self._rules.values()) /
            sum(metadata.numerosity for metadata in self._rules.values())
        )

    def set_timestamps(self, time_stamp):
        """Set the time stamp of each rule in this action set to the given value."""
        for metadata in self._rules.values():
            metadata.time_stamp = time_stamp

    def select_parent(self):
        """Select a rule from this action set, with probability proportionate to its fitness, to act as a parent for a
        new rule in the population. Return its bit condition."""
        total_fitness = sum(metadata.fitness for metadata in self._rules.values())
        selector = random.uniform(0, total_fitness)
        for condition, metadata in self._rules.items():
            selector -= metadata.fitness
            if selector <= 0:
                return condition
        return random.choice(list(self._rules))


class MatchSet:
    """A collection of coincident action sets."""

    def __init__(self, action_sets):
        self._action_sets = {action_set.action: action_set for action_set in action_sets}

    def select_action_set(self, explore=False):
        """Select an action set from among those belonging to this match set. If the explore parameter is provided, it
        is used as the probability of exploration, i.e. uniform action set selection. Otherwise the action set with the
        best predicted payoff is selected with probability 1."""

        if explore and (explore >= 1 or random.random() < explore):
            return random.choice(list(self._action_sets.values()))
        best_prediction = max(action_set.prediction for action_set in self._action_sets.values())
        best_action_sets = [
            action_set
            for action_set in self._action_sets.values()
            if action_set.prediction >= best_prediction
        ]
        if len(best_action_sets) == 1:
            return best_action_sets[0]
        return random.choice(best_action_sets)


class ClassifierSetParameters:
    """The parameters used by the XCS algorithm. For a detailed explanation of each parameter, please see the original
    paper by Martin Butz and Stewart Wilson."""

    max_population_size = 200                   # N
    learning_rate = .15                         # beta
    accuracy_coefficient = .1                   # alpha
    error_threshold = .01                       # epsilon_naught
    accuracy_power = 5                          # nu
    discount_factor = .71                       # gamma
    GA_threshold = 35                           # theta_GA
    crossover_probability = .75                 # chi
    mutation_probability = .03                  # mu
    deletion_threshold = 20                     # theta_del
    fitness_threshold = .1                      # delta
    subsumption_threshold = 20                  # theta_sub
    wildcard_probability = .33                  # P_#
    initial_prediction = .00001                 # p_I
    initial_error = .00001                      # epsilon_I
    initial_fitness = .00001                    # F_I
    exploration_probability = .5                # p_explr
    minimum_actions = 2                         # theta_mna
    do_GA_subsumption = False                   # doGASubsumption
    do_action_set_subsumption = False           # doActionSetSubsumption

    def __init__(self, possible_actions):
        self.possible_actions = frozenset(possible_actions)
        self.minimum_actions = len(self.possible_actions)


class Population:
    """A set of rules (classifiers), together with their associated metadata."""

    def __init__(self, parameters):
        self._population = {}
        self._parameters = parameters
        self._time_stamp = 0

    @property
    def parameters(self):
        """The parameter settings used by this population."""
        return self._parameters

    def get_match_set(self, situation):
        """Accept a situation, encoded as a bit string. Return the set of matching rules (classifiers) for the given
        situation."""

        if not isinstance(situation, (BitString, BitCondition)):
            raise TypeError(situation)

        by_action = {}
        while not by_action:
            for condition, actions in self._population.items():
                if not condition(situation):
                    continue
                for action, rule_metadata in actions.items():
                    if action in by_action:
                        by_action[action][condition] = rule_metadata
                    else:
                        by_action[action] = {condition: rule_metadata}
            if len(by_action) < self._parameters.minimum_actions:
                condition, action, metadata = self.cover(situation, by_action)
                self.add(condition, action, metadata)
                self.prune()
                by_action.clear()

        return MatchSet(ActionSet(action, rules, self) for action, rules in by_action.items())

    def add(self, condition, action, metadata):
        """Add a new rule to the population."""
        if condition not in self._population:
            self._population[condition] = {}
        if action in self._population[condition]:
            self._population[condition][action] += metadata
        else:
            self._population[condition][action] = metadata

    def subsume(self, general_condition, specific_condition, action):
        """NOTE: THIS IS NOT IMPLEMENTED YET.
        Determine whether the more specific condition can be subsumed by the more general one. If so, remove the
        specific condition and add its numerosity to the general condition."""
        if not general_condition(specific_condition):
            return

        # TODO: DO ACTION SET SUBSUMPTION

    def cover(self, situation, existing_actions):
        """Create a new rule that matches the given situation and return it. Preferentially choose an action that is
        not present in the existing actions, if possible."""

        condition = BitCondition.cover(situation, self._parameters.wildcard_probability)
        action_candidates = ((set(existing_actions) - self._parameters.possible_actions) or
                             self._parameters.possible_actions)
        action = random.choice(list(action_candidates))
        metadata = RuleMetadata(self._time_stamp, self._parameters)
        return condition, action, metadata

    def mutate(self, condition, situation):
        """Create a new condition from the given one by probabilistically applying point-wise mutations. Bits that were
        originally wildcarded in the parent condition acquire their values from the provided situation, to ensure the
        child condition continues to match it."""

        # TODO: Revamp to take advantage of numpy array speeds

        bits = []
        mask = []
        for index, value in enumerate(condition):
            if random.random() < self._parameters.mutation_probability:
                if value is None:
                    value = situation[index]
                else:
                    value = None
            if value is None:
                bits.append(False)
                mask.append(False)
            else:
                bits.append(value)
                mask.append(True)

        return BitCondition(bits, mask)

    def get_metadata(self, condition, action):
        """Return the metadata associated with the given rule (classifier)."""

        if condition not in self._population or action not in self._population[condition]:
            return None
        return self._population[condition][action]

    def run_ga(self, action_set, situation):
        """Update the time stamp. If sufficient time has passed, apply the genetic algorithm's operators to update the
         population."""

        self._time_stamp += 1
        if self._time_stamp - action_set.get_average_time_stamp() <= self._parameters.GA_threshold:
            return

        action_set.set_timestamps(self._time_stamp)

        parent1 = action_set.select_parent()
        parent2 = action_set.select_parent()

        if random.random() < self._parameters.crossover_probability:
            child1, child2 = parent1.crossover_with(parent2)
        else:
            child1, child2 = parent1, parent2

        child1 = self.mutate(child1, situation)
        child2 = self.mutate(child2, situation)

        new_children = []
        for child in child1, child2:
            if self._parameters.do_GA_subsumption:
                subsumed = False
                for parent in parent1, parent2:
                    if parent(child):
                        self._population[child][action_set.action].numerosity += 1
                        subsumed = True
                        break
                if subsumed:
                    continue

            if child in self._population:
                if action_set.action in self._population[child]:
                    self._population[child][action_set.action].numerosity += 1
                    continue
            else:
                self._population[child] = {}

            new_children.append(child)

        if new_children:
            parent1_metadata = (self.get_metadata(parent1, action_set.action) or
                                RuleMetadata(self._time_stamp, self._parameters))
            parent2_metadata = (self.get_metadata(parent1, action_set.action) or
                                RuleMetadata(self._time_stamp, self._parameters))

            prediction = (parent1_metadata.prediction + parent2_metadata.prediction) / 2
            error = (parent1_metadata.error + parent2_metadata.error) / 2
            fitness = (parent1_metadata.fitness + parent2_metadata.fitness) / 2 * .1

            for child in new_children:
                metadata = RuleMetadata(self._time_stamp, self._parameters)
                metadata.prediction = prediction
                metadata.error = error
                metadata.fitness = fitness
                self.add(child, action_set.action, metadata)

        self.prune()

    def prune(self):
        """Reduce the population size, if necessary, to ensure that it does not exceed the maximum population size set
        out in the parameters."""

        total_numerosity = sum(
            metadata.numerosity
            for actions in self._population.values()
            for metadata in actions.values()
        )
        if total_numerosity <= self._parameters.max_population_size:
            return

        total_fitness = sum(metadata.fitness for actions in self._population.values() for metadata in actions.values())
        average_fitness = total_fitness / total_numerosity

        total_votes = 0
        deletion_votes = {}
        for condition, pop_by_action in self._population.items():
            votes_by_action = {}
            deletion_votes[condition] = votes_by_action
            for action, metadata in pop_by_action.items():
                vote = metadata.action_set_size * metadata.numerosity
                if (metadata.experience > self._parameters.deletion_threshold and
                        metadata.fitness / metadata.numerosity < self._parameters.fitness_threshold * average_fitness):
                    vote *= average_fitness / (metadata.fitness / metadata.numerosity)
                votes_by_action[action] = vote
                total_votes += vote

        selector = random.uniform(0, total_votes)
        for condition, votes_by_action in deletion_votes.items():
            for action, vote in votes_by_action.items():
                selector -= vote
                if selector <= 0:
                    metadata = self._population[condition][action]
                    if metadata.numerosity > 1:
                        metadata.numerosity -= 1
                    else:
                        del self._population[condition][action]
                        if not self._population[condition]:
                            del self._population[condition]
                    return


class OnLineProblem(metaclass=ABCMeta):
    """Abstract interface for on-line problems accepted by XCS."""

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


class ObservedOnLineProblem(OnLineProblem):
    """Wrapper for other OnLineProblem instances which prints details of the agent/problem interaction as they take
    place."""

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.reward = 0
        self.steps = 0

    def get_possible_actions(self):
        """Return a sequence containing the possible actions that can be executed within the environment."""
        possible_actions = self.wrapped.get_possible_actions()
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


class XCS:
    """The XCS algorithm. Create the parameters and (optionally) a population, passing them in to initialize the XCS
    algorithm. Then create a problem instance and pass it to drive()."""

    def __init__(self, parameters, population=None):
        self._parameters = parameters
        self._population = population or Population(parameters)

    @property
    def parameters(self):
        """The parameter settings used by this instance of the XCS algorithm."""
        return self._parameters

    @property
    def population(self):
        """The population used by this instance of the XCS algorithm."""
        return self._population

    def drive(self, problem):
        """The main loop/entry point of the XCS algorithm. Create a problem instance and pass it in to this method to
        perform the algorithm and optimize the rule set. Problem instances must implement the OnLineProblem interface.
        """
        previous_situation = None
        previous_reward = 0
        previous_action_set = None
        while problem.more():
            situation = problem.sense()
            match_set = self._population.get_match_set(situation)
            action_set = match_set.select_action_set(self._parameters.exploration_probability)
            reward = problem.execute(action_set.action)
            if previous_action_set:
                payoff = previous_reward + self._parameters.discount_factor * action_set.prediction
                previous_action_set.update(payoff)
                self._population.run_ga(previous_action_set, previous_situation)
            previous_situation = situation
            previous_reward = reward
            previous_action_set = action_set
        if previous_action_set:
            previous_action_set.update(previous_reward)
            self._population.run_ga(previous_action_set, previous_situation)


def test():
    """A quick test of the XCS algorithm, demonstrating how to use it in client code."""
    problem = MUXProblem(10000)
    problem = ObservedOnLineProblem(problem)
    parameters = ClassifierSetParameters(problem.get_possible_actions())
    parameters.exploration_probability = .1
    xcs = XCS(parameters)
    xcs.drive(problem)

if __name__ == '__main__':
    test()
