# -------------------------------------------------------------------------------
# Name:     xcs
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
xcs/__init__.py
(c) Aaron Hosford 2015, all rights reserved
Revised BSD License

Implements the XCS (Accuracy-based Classifier System) algorithm,
roughly according to the description provided in the paper, "An
Algorithmic Description of XCS," by Martin Butz and Stewart Wilson.

Butz, M. and Wilson, S. (2001). An algorithmic description of XCS. In Lanzi, P.,
    Stolzmann, W., and Wilson, S., editors, Advances in Learning Classifier Systems:
    Proceedings of the Third International Workshop, volume 1996 of Lecture Notes in
    Artificial Intelligence, pages 253–272. Springer-Verlag Berlin Heidelberg.


A quick explanation of the XCS algorithm:
    The XCS algorithm attempts to solve the reinforcement learning problem, which is to maximize a reward signal by
    learning the optimal mapping from inputs to outputs, where inputs are represented as sequences of bits and
    outputs are selected from a finite set of predetermined actions. It does so by using a genetic algorithm to
    evolve a population of rules of the form

        condition => action => prediction

    where the condition is a bit template (a string of 1s, 0s, and wildcards, represented as #s) which matches
    against one or more inputs, and the prediction is a floating point value that indicates the observed
    reward level when the condition matches the input and the indicated action is selected. The fitness of
    each rule in the population is determined not by the size of the prediction, but by its observed accuracy,
    as well as by the degree to which the rule fills a niche that many other rules do not already fill. The
    reason for using accuracy rather than reward is that it was found that using reward destabilizes the
    population.
"""

# TODO: Finish refactoring the code to isolate the algorithm-specific code from the algorithm-agnostic code. When all
#       is said and done, we should be able to implement a new algorithm with minimal effort or rewritten code, and
#       zero change to the interface.

# TODO: Clean up docstrings and comments with obsolete references to pre-refactoring code.

__author__ = 'Aaron Hosford'
__version__ = '1.0.0a5'
__all__ = [
    '__author__',
    '__version__',
    'RuleMetadata',
    'LCSAlgorithm',
    'ActionSet',
    'MatchSet',
    'Population',
    'XCSRuleMetadata',
    'XCSAlgorithm',
    'LCS',
    'test',
]

import random
from abc import ABCMeta, abstractmethod

# noinspection PyUnresolvedReferences
from xcs.bitstrings import BitString, BitCondition
from xcs.problems import MUXProblem, ObservedOnLineProblem


class RuleMetadata(metaclass=ABCMeta):
    """Abstract base class for metadata used by LCS algorithms to track the rules (classifiers) in a population."""

    numerosity = 1

    @abstractmethod
    def __str__(self):
        raise NotImplementedError()

    # This is here strictly for sorting purposes in calls to Population.__str__
    @abstractmethod
    def __lt__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def prediction(self):
        """The prediction made by this rule."""
        raise NotImplementedError()

    @abstractmethod
    def prediction_weight(self):
        """The weight of this rule's prediction as compared to others in the same action set."""
        raise NotImplementedError()


class LCSAlgorithm(metaclass=ABCMeta):
    """Abstract interface for LCS algorithms. To create a new algorithm that can be used to create an LCS, inherit from
    this class."""

    @abstractmethod
    def get_exploration_probability(self, time_stamp):
        """Return the probability of exploration for the given time stamp."""
        raise NotImplementedError()

    @abstractmethod
    def get_discount_factor(self, time_stamp):
        """Return the future reward discount factor for the given time stamp."""
        raise NotImplementedError()

    @abstractmethod
    def covering_is_required(self, matches_by_action):
        """Return a Boolean indicating whether covering is required given the current matches."""
        raise NotImplementedError()

    @abstractmethod
    def cover(self, time_stamp, situation, existing_actions):
        """Return a tuple (condition, action, metadata) representing a new rule that matches the given situation and
        attempts to avoid duplication of actions already contained in the current matches."""
        raise NotImplementedError()

    @abstractmethod
    def distribute_payoff(self, action_set, payoff):
        """Accept a payoff and distribute it among the rules in the action set which deserve credit for it."""
        raise NotImplementedError()

    @abstractmethod
    def update(self, action_set, situation):
        """Update the population, e.g. by applying GA, based on the situation and action set."""
        raise NotImplementedError()

    @abstractmethod
    def prune(self, population):
        """Reduce the population size, if necessary, by removing lower-quality rules."""
        raise NotImplementedError()


class ActionSet:
    """Abstract base class for a set of rules (classifiers) with the same action that matched the same situation."""

    def __init__(self, action, rules, population):
        self._action = action
        self._rules = rules  # {condition: metadata}
        self._prediction = None  # We'll calculate this later if it is needed
        self._population = population

    @property
    def conditions(self):
        """An iterator over the conditions in the action set."""
        return iter(self._rules)

    @property
    def metadata(self):
        """An iterator over the metadata of the rules in the action set."""
        return self._rules.values()

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
        # If the action set's prediction has not already been computed, do so by taking the weighted
        # average of the individual rules' predictions.
        if self._prediction is None:
            total_weight = 0
            total_prediction = 0
            for metadata in self._rules.values():
                total_weight += metadata.prediction_weight
                total_prediction += metadata.prediction * metadata.prediction_weight
            self._prediction = total_prediction / (total_weight or 1)
        return self._prediction

    def get_metadata(self, condition):
        """Return the metadata of the condition appearing in this action set."""
        return self._rules[condition]

    def remove_condition(self, condition):
        """Remove the condition from the action set."""
        del self._rules[condition]

    def accept_payoff(self, payoff):
        """Accept the given payoff and distribute it across the action set."""
        self._population.algorithm.distribute_payoff(self, payoff)


class MatchSet:
    """A collection of coincident action sets. This represents the collection of all rules that matched
    within the same situation, organized into groups according to which action each rule recommends."""

    def __init__(self, action_sets):
        self._action_sets = {action_set.action: action_set for action_set in action_sets}

    def select_action_set(self, explore=False):
        """Select an action set from among those belonging to this match set. If the explore parameter is provided, it
        is used as the probability of exploration, i.e. uniform action set selection. Otherwise the action set with the
        best predicted payoff is selected with probability 1."""

        # If an exploration probability has been provided, then with that probability,
        # select an action uniformly rather than proportionally to the prediction
        # for each action.
        if explore and (explore >= 1 or random.random() < explore):
            return random.choice(list(self._action_sets.values()))

        # Otherwise, determine the collectively predicted reward for each possible action based
        # on the individual predictions of each rule suggesting that action, and choose the
        # action having the highest predicted reward.
        best_prediction = max(action_set.prediction for action_set in self._action_sets.values())
        best_action_sets = [
            action_set
            for action_set in self._action_sets.values()
            if action_set.prediction >= best_prediction
        ]

        # If only one action has the maximum predicted reward, return it. Otherwise,
        # choose uniformly from among the actions sharing the maximum predicted reward.
        if len(best_action_sets) == 1:
            return best_action_sets[0]
        return random.choice(best_action_sets)


class Population:
    """A set of rules (classifiers), together with their associated metadata, which the XCS algorithm
    attempts to evolve using a genetic algorithm. This population represents the accumulated experience
    of the XCS algorithm, in the form of a set of accurate rules that map classes of situations
    (identified by bit conditions) to actions and the expected rewards if those actions are taken
    within those classes of situations."""

    def __init__(self, algorithm):
        if not isinstance(algorithm, LCSAlgorithm):
            raise TypeError(algorithm)

        self._population = {}
        self._algorithm = algorithm
        self._time_stamp = 0

    @property
    def algorithm(self):
        """The algorithm managing this population."""
        return self._algorithm

    @property
    def time_stamp(self):
        """The number of steps completed since the population was initialized."""
        return self._time_stamp

    def __iter__(self):
        for condition, by_action in self._population.items():
            for action in by_action:
                yield condition, action

    def __len__(self):
        return sum(len(by_action) for by_action in self._population.values())

    def __contains__(self, condition_action):
        condition, action = condition_action
        return action in self._population.get(condition, ())

    def __str__(self):
        return '\n'.join(
            str(condition) + ' => ' + str(action) + '\n    ' +
            str(self.get_metadata(condition, action)).replace('\n', '\n    ')
            for condition, action in sorted(self, key=lambda condition_action: self.get_metadata(*condition_action))
        )

    def get_match_set(self, situation):
        """Accept a situation, encoded as a bit string. Return the set of matching rules (classifiers) for the given
        situation."""

        if not isinstance(situation, (BitString, BitCondition)):
            raise TypeError(situation)

        # Find the conditions that match against the current situation, and group them according to which
        # action(s) they recommend.
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
            # If an insufficient number of actions are recommended, create some new rules (condition/action pairs)
            # until there are enough actions being recommended.
            if self._algorithm.covering_is_required(by_action):
                condition, action, metadata = self._algorithm.cover(self._time_stamp, situation, by_action)
                assert condition(situation)
                self.add(condition, action, metadata)
                self._algorithm.prune(self)
                by_action.clear()

        # Return a match set which succinctly represents the information we just gathered.
        return MatchSet(ActionSet(action, rules, self) for action, rules in by_action.items())

    def add(self, condition, action, metadata=1):
        """Add a new rule to the population."""
        if condition not in self._population:
            self._population[condition] = {}

        # If the rule already exists in the population, then we virtually add the rule
        # by incrementing the existing rule's numerosity. This prevents redundancy in
        # the rule set.
        if isinstance(metadata, int):
            if condition not in self._population or action not in self._population[condition]:
                raise ValueError("Metadata must be supplied for new members of population.")
            self._population[condition][action].numerosity += 1
        elif isinstance(metadata, RuleMetadata):
            if action in self._population[condition]:
                self._population[condition][action].numerosity += metadata.numerosity
            else:
                self._population[condition][action] = metadata
        else:
            raise TypeError(metadata)

    def remove(self, condition, action, count=1):
        """Remove one or more instances of a rule in the population."""
        metadata = self.get_metadata(condition, action)
        if metadata is None:
            return

        # Only actually remove the rule if its numerosity drops below 1.
        metadata.numerosity -= count
        if metadata.numerosity <= 0:
            del self._population[condition][action]
            if not self._population[condition]:
                del self._population[condition]

    def get_metadata(self, condition, action):
        """Return the metadata associated with the given rule (classifier). If the rule is not present in the
        population, return None."""
        if condition not in self._population or action not in self._population[condition]:
            return None
        return self._population[condition][action]

    def update_time_stamp(self):
        """Update the time stamp, indicating another completed cycle of the algorithm."""
        self._time_stamp += 1


class XCSRuleMetadata(RuleMetadata):
    """Metadata used by the XCS algorithm to track the rules (classifiers) in a population."""

    def __init__(self, time_stamp, algorithm):
        self.time_stamp = time_stamp  # The iteration of the algorithm at which this rule was last updated
        self.average_reward = algorithm.initial_prediction  # The predicted (averaged) reward for this rule
        self.error = algorithm.initial_error  # The observed error in this rule's prediction
        self.fitness = algorithm.initial_fitness  # The fitness of this rule within the GA
        self.experience = 0  # The number of times this rule has been evaluated
        self.action_set_size = 1  # The average number of rules sharing the same niche as this rule
        self.numerosity = 1  # The number of instances of this rule in the population, used to eliminate redundancy

    def __str__(self):
        return '\n'.join(
            key.replace('_', ' ').title() + ': ' + str(getattr(self, key))
            for key in (
                'time_stamp',
                'average_reward',
                'error',
                'fitness',
                'experience',
                'action_set_size',
                'numerosity'
            )
        )

    # This is here strictly for sorting purposes in calls to Population.__str__
    def __lt__(self, other):
        if not isinstance(other, XCSRuleMetadata):
            return NotImplemented
        attribute_order = (
            'numerosity',
            'fitness',
            'error',
            'average_reward',
            'experience',
            'action_set_size',
            'time_stamp'
        )
        for attribute in attribute_order:
            my_key = getattr(self, attribute)
            other_key = getattr(other, attribute)
            if my_key < other_key:
                return attribute != 'error'
            if my_key > other_key:
                return attribute == 'error'
        return False

    @property
    def prediction(self):
        """The prediction made by this rule. For XCS, this is the average reward received when the condition matches
        and the action is taken."""
        return self.average_reward

    @property
    def prediction_weight(self):
        """The weight of this rule's prediction as compared to others in the same action set. For XCS, this is the
        fitness of the rule."""
        return self.fitness


class XCSAlgorithm(LCSAlgorithm):
    """The XCS algorithm."""

    # For a detailed explanation of each parameter, please see the original
    # paper by Martin Butz and Stewart Wilson.
    max_population_size = 200                   # N
    learning_rate = .15                         # beta
    accuracy_coefficient = .1                   # alpha
    error_threshold = .01                       # epsilon_0
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

    def get_exploration_probability(self, time_stamp):
        """Return the probability of exploration for the given time stamp."""
        return self.exploration_probability

    def get_discount_factor(self, time_stamp):
        """Return the future reward discount factor for the given time stamp."""
        return self.discount_factor

    def covering_is_required(self, actions):
        """Return a Boolean value indicating whether covering is required based on the contents of the current action
        set."""
        return len(actions) < self.minimum_actions

    def cover(self, time_stamp, situation, existing_actions):
        """Create a new rule that matches the given situation and return it. Preferentially choose an action that is
        not present in the existing actions, if possible."""

        # Create a new condition that matches the situation.
        condition = BitCondition.cover(situation, self.wildcard_probability)

        # Pick a random action that (preferably) isn't already suggested by some
        # other rule for this situation.
        action_candidates = ((set(existing_actions) - self.possible_actions) or
                             self.possible_actions)
        action = random.choice(list(action_candidates))

        # Create metadata for the new rule.
        metadata = XCSRuleMetadata(time_stamp, self)

        # The actual rule is just a condition/action/metadata triple
        return condition, action, metadata

    def distribute_payoff(self, action_set, payoff):
        """Update the rule metadata for the rules belonging to this action set, based on the payoff received."""
        action_set_size = sum(metadata.numerosity for metadata in action_set.metadata)

        # Update the average reward, error, and action set size of each rule participating in the
        # action set.
        for metadata in action_set.metadata:
            metadata.experience += 1

            update_rate = max(self.learning_rate, 1 / metadata.experience)

            metadata.average_reward += (payoff - metadata.average_reward) * update_rate
            metadata.error += (abs(payoff - metadata.average_reward) - metadata.error) * update_rate
            metadata.action_set_size += (action_set_size - metadata.action_set_size) * update_rate

        # Update the fitness of the rules.
        self._update_fitness(action_set)

        # If the parameters so indicate, perform action set subsumption.
        if self.do_action_set_subsumption:
            self._action_set_subsumption(action_set)

    def update(self, action_set, situation):
        """Update the time stamp. If sufficient time has passed, apply the genetic algorithm's operators to update the
         population."""

        # Increment the iteration counter.
        action_set.population.update_time_stamp()

        # If the average number of iterations since the last update for each rule in the action set
        # is too small, return early instead of applying the GA.
        if action_set.population.time_stamp - self._get_average_time_stamp(action_set) <= self.GA_threshold:
            return

        # Update the time step for each rule to indicate that they were updated by the GA.
        self._set_timestamps(action_set)

        # Select two parents from the action set, with probability proportionate to their fitness.
        parent1 = self._select_parent(action_set)
        parent2 = self._select_parent(action_set)

        parent1_metadata = (action_set.population.get_metadata(parent1, action_set.action) or
                            XCSRuleMetadata(action_set.population.time_stamp, self))
        parent2_metadata = (action_set.population.get_metadata(parent1, action_set.action) or
                            XCSRuleMetadata(action_set.population.time_stamp, self))

        # With the probability specified in the parameters, apply the crossover operator
        # to the parents. Otherwise, just take the parents unchanged.
        if random.random() < self.crossover_probability:
            child1, child2 = parent1.crossover_with(parent2)
        else:
            child1, child2 = parent1, parent2

        # Apply the mutation operator to each child, randomly flipping their mask bits with a small probability.
        child1 = self._mutate(child1, situation)
        child2 = self._mutate(child2, situation)

        # If the newly generated children are already present in the population (or if they
        # should be subsumed due to GA subsumption) then simply increment the numerosities
        # of the existing rules in the population.
        new_children = []
        for child in child1, child2:
            # If the parameters specify that GA subsumption should be performed, look for an
            # accurate parent that can subsume the new child.
            if self.do_GA_subsumption:
                subsumed = False
                for parent, metadata in (parent1, parent1_metadata), (parent2, parent2_metadata):
                    if (metadata.experience > self.subsumption_threshold and
                            metadata.error < self.error_threshold and
                            parent(child)):
                        # print(parent, child)
                        action_set.population.add(parent, action_set.action)
                        subsumed = True
                        break
                if subsumed:
                    continue

            # Provided the child has not already been subsumed and it is present in the
            # population, just increment its numerosity. Otherwise, if the child has
            # neither been subsumed nor does it already exist, remember it so we can
            # add it to the population in just a moment.
            if (child, action_set.action) in action_set.population:
                action_set.population.add(child, action_set.action)
            else:
                new_children.append(child)

        # If there were any children which weren't subsumed and weren't already present
        # in the population, add them.
        if new_children:
            average_reward = (parent1_metadata.average_reward + parent2_metadata.average_reward) / 2
            error = (parent1_metadata.error + parent2_metadata.error) / 2
            fitness = (parent1_metadata.fitness + parent2_metadata.fitness) / 2 * .1

            for child in new_children:
                metadata = XCSRuleMetadata(action_set.population.time_stamp, self)
                metadata.average_reward = average_reward
                metadata.error = error
                metadata.fitness = fitness
                action_set.population.add(child, action_set.action, metadata)

        # Prune the population down if its size exceeds the maximum permitted.
        self.prune(action_set.population)

    def prune(self, population):
        """Reduce the population size, if necessary, to ensure that it does not exceed the maximum population size set
        out in the parameters."""

        # Determine the virtual population size.
        total_numerosity = sum(
            population.get_metadata(condition, action).numerosity
            for condition, action in population
        )

        # If the virtual population size is already small enough, just return early.
        if total_numerosity <= self.max_population_size:
            return

        # Determine the average fitness of the rules in the virtual population.
        total_fitness = sum(
            population.get_metadata(condition, action).fitness
            for condition, action in population
        )
        average_fitness = total_fitness / total_numerosity

        # Determine the probability of deletion, as a function of both accuracy and niche sparsity.
        total_votes = 0
        deletion_votes = {}
        for condition, action in population:
            metadata = population.get_metadata(condition, action)
            vote = metadata.action_set_size * metadata.numerosity
            if (metadata.experience > self.deletion_threshold and
                    metadata.fitness / metadata.numerosity < self.fitness_threshold * average_fitness):
                vote *= average_fitness / (metadata.fitness / metadata.numerosity)
            deletion_votes[condition, action] = vote
            total_votes += vote

        # Choose a rule to delete based on the probabilities just computed.
        selector = random.uniform(0, total_votes)
        for (condition, action), vote in deletion_votes.items():
            selector -= vote
            if selector <= 0:
                population.remove(condition, action)
                return

    def _update_fitness(self, action_set):
        """Update the fitness of the rules belonging to this action set."""
        # Compute the accuracy of each rule. Accuracy is inversely proportional to error. Below a certain error
        # threshold, accuracy becomes constant. Accuracy values range over (0, 1].
        total_accuracy = 0
        accuracies = {}
        for condition in action_set.conditions:
            metadata = action_set.get_metadata(condition)
            if metadata.error < self.error_threshold:
                accuracy = 1
            else:
                accuracy = (
                    self.accuracy_coefficient *
                    (metadata.error / self.error_threshold) ** -self.accuracy_power
                )
            accuracies[condition] = accuracy
            total_accuracy += accuracy * metadata.numerosity

        # Use the relative accuracies of the rules to update their fitness
        for condition in action_set.conditions:
            metadata = action_set.get_metadata(condition)
            accuracy = accuracies[condition]
            metadata.fitness += (
                self.learning_rate *
                (accuracy * metadata.numerosity / total_accuracy - metadata.fitness)
            )

    def _action_set_subsumption(self, action_set):
        """Perform action set subsumption."""
        # Select a condition with maximum bit count among those having sufficient experience and
        # sufficiently low error.
        selected_condition = None
        selected_bit_count = None
        for condition in action_set.conditions:
            metadata = action_set.get_metadata(condition)
            if not (metadata.experience > self.subsumption_threshold and
                    metadata.error < self.error_threshold):
                continue
            bit_count = condition.count()
            if (selected_condition is None or
                    bit_count > selected_bit_count or
                    (bit_count == selected_bit_count and random.randrange(2))):
                selected_condition = condition
                selected_bit_count = bit_count

        # If no condition was found satisfying the requirements, return early.
        if selected_condition is None:
            return

        selected_metadata = action_set.get_metadata(selected_condition)

        # Subsume each rule which the selected rule generalizes. When a rule is subsumed, all
        # instances of the subsumed rule are replaced with instances of the more general one
        # in the population.
        to_remove = []
        for condition in action_set.conditions:
            metadata = action_set.get_metadata(condition)
            if selected_condition is not condition and selected_condition(condition):
                selected_metadata.numerosity += metadata.numerosity
                action_set.population.remove(condition, action_set.action, metadata.numerosity)
                to_remove.append(condition)
        for condition in to_remove:
            action_set.remove_condition(condition)

    @staticmethod
    def _get_average_time_stamp(action_set):
        """Return the average time stamp for the rules in this action set."""
        # This is the average value of the iteration counter upon the most
        # recent update of each rule in this action set.
        return (
            sum(metadata.time_stamp * metadata.numerosity for metadata in action_set.metadata) /
            (sum(metadata.numerosity for metadata in action_set.metadata) or 1)
        )

    @staticmethod
    def _set_timestamps(action_set):
        """Set the time stamp of each rule in this action set to the given value."""
        # Indicate that each rule has been updated at the given iteration.
        for metadata in action_set.metadata:
            metadata.time_stamp = action_set.population.time_stamp

    @staticmethod
    def _select_parent(action_set):
        """Select a rule from this action set, with probability proportionate to its fitness, to act as a parent for a
        new rule in the population. Return its bit condition."""
        total_fitness = sum(metadata.fitness for metadata in action_set.metadata)
        selector = random.uniform(0, total_fitness)
        for condition in action_set.conditions:
            metadata = action_set.get_metadata(condition)
            selector -= metadata.fitness
            if selector <= 0:
                return condition
        # If for some reason a case slips through the above loop, perhaps due to floating point error,
        # we fall back on uniform selection.
        return random.choice(list(action_set.conditions))

    def _mutate(self, condition, situation):
        """Create a new condition from the given one by probabilistically applying point-wise mutations. Bits that were
        originally wildcarded in the parent condition acquire their values from the provided situation, to ensure the
        child condition continues to match it."""

        # TODO: Revamp to take advantage of numpy array speeds

        # Go through each position in the condition, randomly flipping whether
        # the position is a value (0 or 1) or a wildcard (#). We do this in
        # a new list because the original condition's mask is immutable.
        mask = list(condition.mask)
        for index in range(len(mask)):
            if random.random() < self.mutation_probability:
                mask[index] = not mask[index]

        # The bits that aren't wildcards always have the same value as the situation,
        # which ensures that the mutated condition still matches the situation.
        return BitCondition(situation, mask)


class LCS:
    """The XCS algorithm. Create the parameters and (optionally) a population, passing them in to initialize the XCS
    algorithm. Then create a problem instance and pass it to drive()."""

    def __init__(self, algorithm, population=None):
        if not isinstance(algorithm, LCSAlgorithm):
            raise TypeError(algorithm)
        if population is not None and not isinstance(population, Population):
            raise TypeError(population)

        self._algorithm = algorithm
        self._population = population or Population(algorithm)

    @property
    def algorithm(self):
        """The parameter settings used by this instance of the XCS algorithm."""
        return self._algorithm

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

        # Repeat until the problem has run its course.
        while problem.more():
            # Gather information about the current state of the environment.
            situation = problem.sense()

            # Determine which rules match the current situation.
            match_set = self._population.get_match_set(situation)

            # Select the best action for the current situation (or a random one,
            # if we are on an exploration step).
            exploration_probability = self._algorithm.get_exploration_probability(self._population.time_stamp)
            action_set = match_set.select_action_set(exploration_probability)

            # Perform the selected action and find out what the received reward was.
            reward = problem.execute(action_set.action)

            # Don't immediately apply the reward; instead, wait until the next iteration and
            # factor in not only the reward that was received on the previous step, but the
            # (discounted) reward that is expected going forward given the resulting situation
            # observed after the action was taken. This is a classic feature of reinforcement
            # learning algorithms, which acts to stitch together a general picture of the
            # future expected reward without actually waiting the full duration to find out
            # what it will be.
            if previous_action_set:
                discount_factor = self._algorithm.get_discount_factor(self._population.time_stamp)
                payoff = previous_reward + discount_factor * action_set.prediction
                previous_action_set.accept_payoff(payoff)
                # TODO: Should the action set remember the situation, making this second argument obsolete?
                self._algorithm.update(previous_action_set, previous_situation)
            previous_situation = situation
            previous_reward = reward
            previous_action_set = action_set

        # This serves to tie off the final stitch. The last action taken gets only the
        # immediate reward; there is no future reward expected.
        if previous_action_set:
            previous_action_set.accept_payoff(previous_reward)
            self._algorithm.update(previous_action_set, previous_situation)


def test():
    """A quick test of the XCS algorithm, demonstrating how to use it in client code."""
    import time

    # Define the problem.
    problem = MUXProblem(10000)

    # Put the problem into a wrapper that will report things back to us for visibility.
    problem = ObservedOnLineProblem(problem)

    # Define the algorithm.
    algorithm = XCSAlgorithm(problem.get_possible_actions())
    algorithm.exploration_probability = .1
    algorithm.do_GA_subsumption = True
    algorithm.do_action_set_subsumption = True

    # Create the classifier system from the algorithm.
    xcs = LCS(algorithm)

    start_time = time.time()

    # Run the algorithm on the problem. This does two things simultaneously:
    #   1. Learns a model of the problem space from experience.
    #   2. Attempts to maximize the reward received.
    # Since initially the algorithm's model has no experience incorporated
    # into it, performance will be poor, but it will improve over time as
    # the algorithm continues to be exposed to the problem.
    xcs.drive(problem)

    print()
    print("Population:")
    print(xcs.population)
    print()

    end_time = time.time()
    print("Total time:", end_time - start_time, "seconds")
