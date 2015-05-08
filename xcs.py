# -------------------------------------------------------------------------------
# Name:     xcs.py
# Purpose:  Implements the XCS (Accuracy-based Classifier System) algorithm,
#           per the description provided in:
#           http://opim.wharton.upenn.edu/~sok/papers/b/XCSAlgDesc01202001.pdf
#
# Author:       Aaron Hosford
#
# Created:      5/5/2015
# Copyright:    (c) Aaron Hosford 2015, all rights reserved
# Licence:      Revised (3 Clause) BSD License
# -------------------------------------------------------------------------------

import numpy
import random
from abc import ABCMeta, abstractmethod


class BitString:

    @classmethod
    def from_int(cls, value, length=None):
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

    @classmethod
    def cover(cls, bits, wildcard_probability):
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
        return self._bits

    @property
    def mask(self):
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
        if not isinstance(other, BitCondition):
            raise TypeError(other)
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

    def __init__(self, action, rules, classifier_set):
        self._action = action
        self._rules = rules  # {condition : metadata}
        self._prediction = None
        self._classifier_set = classifier_set
        self._parameters = classifier_set.parameters

    @property
    def action(self):
        return self._action

    @property
    def classifier_set(self):
        return self._classifier_set

    @property
    def prediction(self):
        if self._prediction is None:
            total_fitness = 0
            total_prediction = 0
            for metadata in self._rules.values():
                total_fitness += metadata.fitness
                total_prediction += metadata.prediction * metadata.fitness
            self._prediction = total_prediction / (total_fitness or 1)
        return self._prediction

    def _update_fitness(self):
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
        return (
            sum(metadata.time_stamp * metadata.numerosity for metadata in self._rules.values()) /
            sum(metadata.numerosity for metadata in self._rules.values())
        )

    def set_timestamps(self, timestamp):
        for metadata in self._rules.values():
            metadata.timestamp = timestamp

    def select_parent(self):
        total_fitness = sum(metadata.fitness for metadata in self._rules.values())
        selector = random.uniform(0, total_fitness)
        for condition, metadata in self._rules.items():
            selector -= metadata.fitness
            if selector <= 0:
                return condition
        return random.choice(list(self._rules))


class MatchSet:

    def __init__(self, action_sets):
        self._action_sets = {action_set.action: action_set for action_set in action_sets}

    def select_action_set(self, explore=False):
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

    def __init__(self, parameters):
        self._population = {}
        self._parameters = parameters
        self._time_stamp = 0

    @property
    def parameters(self):
        return self._parameters

    def get_match_set(self, situation):
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
        if condition not in self._population:
            self._population[condition] = {}
        if action in self._population[condition]:
            self._population[condition][action] += metadata
        else:
            self._population[condition][action] = metadata

    def subsume(self, general_condition, specific_condition, action):
        if not general_condition(specific_condition):
            return

        # TODO: DO ACTION SET SUBSUMPTION

    def cover(self, situation, existing_actions):
        condition = BitCondition.cover(situation, self._parameters.wildcard_probability)
        action_candidates = ((set(existing_actions) - self._parameters.possible_actions) or
                             self._parameters.possible_actions)
        action = random.choice(list(action_candidates))
        metadata = RuleMetadata(self._time_stamp, self._parameters)
        return condition, action, metadata

    def mutate(self, condition, situation):
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
        if condition not in self._population or action not in self._population[condition]:
            return None
        return self._population[condition][action]

    def run_ga(self, action_set, situation):
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

    @abstractmethod
    def get_possible_actions(self):
        raise NotImplementedError()

    @abstractmethod
    def sense(self):
        raise NotImplementedError()

    @abstractmethod
    def execute(self, action):
        raise NotImplementedError()

    @abstractmethod
    def more(self):
        raise NotImplementedError()


class MUXProblem(OnLineProblem):

    def __init__(self, training_cycles=1000, address_size=3):
        self.address_size = address_size
        self.current_situation = None
        self.possible_actions = (True, False)
        self.remaining_cycles = training_cycles

    def get_possible_actions(self):
        return self.possible_actions

    def sense(self):
        self.current_situation = BitString([
            random.randrange(2)
            for _ in range(self.address_size + (1 << self.address_size))
        ])
        return self.current_situation

    def execute(self, action):
        self.remaining_cycles -= 1
        index = int(BitString(self.current_situation[:self.address_size]))
        bit = self.current_situation[self.address_size + index]
        return action == bit

    def more(self):
        return self.remaining_cycles > 0


class ObservedOnLineProblem(OnLineProblem):

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.reward = 0
        self.steps = 0

    def get_possible_actions(self):
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
        situation = self.wrapped.sense()

        print()
        print('Situation:', situation)
        print()

        return situation

    def execute(self, action):
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
        more = self.wrapped.more()

        print()
        print("Steps:", self.steps)
        print("Next iteration" if more else "Terminated")
        print()

        return more


class XCS:

    def __init__(self, parameters, population=None):
        self._parameters = parameters
        self._population = population or Population(parameters)

    @property
    def parameters(self):
        return self._parameters

    @property
    def population(self):
        return self._population

    def drive(self, problem):
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


def main():
    problem = MUXProblem(10000)
    problem = ObservedOnLineProblem(problem)
    parameters = ClassifierSetParameters(problem.get_possible_actions())
    parameters.exploration_probability = .1
    xcs = XCS(parameters)
    xcs.drive(problem)

if __name__ == '__main__':
    main()
