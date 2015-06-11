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
xcs/__init__.py
(c) Aaron Hosford 2015, all rights reserved
Revised BSD License

Implements the XCS (Accuracy-based Classifier System) algorithm,
as described in the 2001 paper, "An Algorithmic Description of
XCS," by Martin Butz and Stewart Wilson.

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


__author__ = 'Aaron Hosford'
__version__ = '1.0.0a9'
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
    'bitstrings',
    'problems',
]


import random
from abc import ABCMeta, abstractmethod


# Attempt to import numpy. If unsuccessful, set numpy = None.
try:
    # noinspection PyUnresolvedReferences
    import numpy
except ImportError:
    numpy = None
else:
    # This is necessary because sometimes the empty numpy folder is left in place when it is uninstalled.
    try:
        numpy.ndarray
    except AttributeError:
        numpy = None


from . import bitstrings, problems


class ActionSelectionStrategy(metaclass=ABCMeta):
    """Abstract base class defining the minimal interface for action selection policies. The action selection
    strategy is responsible for governing the trade-off between exploration (acquiring new experience) and
    exploitation (utilizing existing experience to maximize reward). Specify a new action selection strategy
    by subclassing this interface, or by defining a function which accepts a MatchSet as its sole argument
    and returns one of the actions suggested by that match set."""

    # Defining this allows the object to be used like a function.
    @abstractmethod
    def __call__(self, match_set):
        raise NotImplementedError()


class EpsilonGreedySelectionStrategy(ActionSelectionStrategy):
    """The epsilon-greedy action selection strategy. With probability epsilon, an action is chosen uniformly from all
     possible actions regardless of predicted payoff. The rest of the time, the action with the highest predicted
     payoff is chosen. The probability of exploration, epsilon, does not change as time passes."""

    def __init__(self, epsilon=.1):
        assert 0 <= epsilon <= 1
        self.epsilon = epsilon

    def __call__(self, match_set):
        assert isinstance(match_set, MatchSet)

        # With probability epsilon, select an action uniformly from all the actions in the match set.
        if random.random() < self.epsilon:
            return random.choice(list(match_set))

        # Otherwise, return (one of) the best action(s)
        best_actions = match_set.best_actions
        if len(best_actions) == 1:
            return best_actions[0]
        else:
            return random.choice(best_actions)


class RuleMetadata(metaclass=ABCMeta):
    """Abstract base class defining the minimal interface for metadata used by LCS algorithms to track the
    rules, aka classifiers, in a population. A classifier consists of a condition and an action taken as a
    pair. Each unique classifier in the population has an associated metadata instance. If there are multiple
    instances of a single classifier, this is indicated by incrementing the numerosity attribute of the
    associated metadata instance, rather than adding another copy of that classifier."""

    # The number of instances of the rule (classifier) with which this metadata instance is associated.
    numerosity = 1

    # Defining this determines the behavior of str(instance)
    @abstractmethod
    def __str__(self):
        raise NotImplementedError()

    # Defining this determines the behavior of instance1 < instance2
    # This is here strictly for sorting purposes in calls to Population.__str__
    @abstractmethod
    def __lt__(self, other):
        raise NotImplementedError()

    @property
    @abstractmethod
    def prediction(self):
        """The reward prediction made by this rule. This value represents the reward expected if the rule's action
        is taken when its condition matches."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def prediction_weight(self):
        """The weight of this rule's predictions. This is used to resolve conflicting predictions made by
        multiple classifiers appearing in the same action set. The combined reward prediction for the
        entire action set is the weighted average of the predictions made by each classifier appearing in
        that action set."""
        raise NotImplementedError()


class LCSAlgorithm(metaclass=ABCMeta):
    """Abstract base class defining the minimal interface for LCS algorithms. To create a new algorithm that can
    be used to initialize an LCS, inherit from this class. An LCS algorithm is responsible for managing the
    LCS's rule (aka classifier) population, distributing reward to the appropriate rules, and determining the
    action selection strategy that is used."""

    @property
    @abstractmethod
    def action_selection_strategy(self):
        """The action selection strategy used to govern the trade-off between exploration (acquiring new experience)
        and exploitation (utilizing existing experience to maximize reward)."""
        raise NotImplementedError()

    @abstractmethod
    def get_future_expectation(self, match_set):
        """Return a numerical value representing the expected future consequences (in terms of discounted reward) of
        the previously selected action(s), given only the current match set. The match_set argument is a MatchSet
        instance representing the current match set."""
        raise NotImplementedError()

    @abstractmethod
    def covering_is_required(self, match_set):
        """Return a Boolean indicating whether covering is required for the current match set. The match_set
        argument is a MatchSet instance representing the current match set before covering is applied."""
        raise NotImplementedError()

    @abstractmethod
    def cover(self, match_set):
        """Return a tuple (condition, action, metadata) representing a new rule that can be added to the
        match set, which matches the situation of the match set and attempts to avoid duplication of the
        actions already contained therein. The match_set argument is a MatchSet instance representing
        the match set to which the returned rule will be added."""
        raise NotImplementedError()

    @abstractmethod
    def distribute_payoff(self, match_set, payoff):
        """Accept a payoff in response to the selected action of the given match set, and distribute it among the
        rules in the action set which deserve credit for recommending the action. The match_set argument is the
        MatchSet instance which suggested the selected action which earned the payoff, and the payoff argument
        is a numerical value that represents the payoff received in response to the selected action."""
        raise NotImplementedError()

    @abstractmethod
    def update(self, match_set):
        """Update the classifier population from which the match set was drawn, e.g. by applying a genetic algorithm.
        The match_set argument is the MatchSet instance whose source population should be updated. The source
        population which is to be updated can be accessed through its population property."""
        raise NotImplementedError()

    @abstractmethod
    def prune(self, population):
        """Reduce the population size, if necessary, by removing lower-quality rules. Return a sequence containing
        the rules whose numerosities dropped to zero as a result of this call. The population argument is a
        Population instance which utilizes this algorithm."""
        raise NotImplementedError()


class ActionSet:
    """A set of rules (classifiers) drawn from the same population, all suggesting the same action and having
    conditions which matched the same situation, together with information as to the conditions under which the
    rules matched together."""

    def __init__(self, population, situation, action, rules):
        assert isinstance(population, Population)
        assert isinstance(rules, dict)
        assert all(isinstance(metadata, RuleMetadata) for metadata in rules.values())

        self._population = population
        self._situation = situation
        self._action = action
        self._rules = rules  # {condition: metadata}

        self._prediction = None  # We'll calculate this later if it is needed
        self._prediction_weight = None

        # Capture the time stamp of the population at which the action set was created
        self._time_stamp = population.time_stamp

    @property
    def population(self):
        """The population from which the classifiers in the action set were drawn."""
        return self._population

    @property
    def situation(self):
        """The common situation against which all the classifiers' conditions matched."""
        return self._situation

    @property
    def action(self):
        """The common action suggested by all the classifiers in the action set."""
        return self._action

    @property
    def conditions(self):
        """An iterator over the conditions of the classifiers in the action set."""
        return iter(self._rules)

    @property
    def metadata(self):
        """An iterator over the metadata of the classifiers in the action set."""
        return self._rules.values()

    @property
    def time_stamp(self):
        """The population time stamp at which this action set was generated."""
        return self._time_stamp

    def _compute_prediction(self):
        """Compute the combined prediction and prediction weight for this action set. The combined prediction is the
        weighted average of the individual predictions of the classifiers. The combined prediction weight is the sum
        of the individual prediction weights of the classifiers."""
        total_weight = 0
        total_prediction = 0
        for metadata in self._rules.values():
            total_weight += metadata.prediction_weight
            total_prediction += metadata.prediction * metadata.prediction_weight
        self._prediction = total_prediction / (total_weight or 1)
        self._prediction_weight = total_weight

    @property
    def prediction(self):
        """The combined prediction of expected payoff for taking the suggested action given the situation. This
        is the weighted average of the individual predictions of the classifiers constituting this action set."""
        if self._prediction is None:
            self._compute_prediction()
        return self._prediction

    @property
    def prediction_weight(self):
        """The total weight of the combined prediction made by this action set. This is the sum of the weights of
        the individual predictions made by the classifiers constituting this action set."""
        if self._prediction_weight is None:
            self._compute_prediction()
        return self._prediction_weight

    def get_metadata(self, condition):
        """Return the metadata of the classifier having this condition and appearing in this action set."""
        return self._rules[condition]

    def remove_condition(self, condition):
        """Remove the classifier having this condition from the action set."""
        del self._rules[condition]


class MatchSet:
    """A collection of coincident action sets. This represents the collection of all rules that matched
    within the same situation, organized into groups according to which action each rule recommends."""

    def __init__(self, population, situation, by_action):
        assert isinstance(population, Population)
        assert isinstance(by_action, dict)

        self._action_sets = {
            action: ActionSet(population, situation, action, rules)
            for action, rules in by_action.items()
        }
        self._best_actions = None
        self._best_prediction = None
        self._situation = situation
        self._population = population
        self._time_stamp = population.time_stamp
        self._selected_action = None

    @property
    def population(self):
        return self._population

    @property
    def situation(self):
        return self._situation

    @property
    def time_stamp(self):
        return self._time_stamp

    # Defining this determines the behavior of this class with respect to iteration, including the "iter(instance)"
    # and "for item in instance:" constructs.
    def __iter__(self):
        return iter(self._action_sets)

    # Defining this determines the behavior of len(instance)
    def __len__(self):
        return len(self._action_sets)

    # Defining this determines the behavior of instance[key]
    def __getitem__(self, action):
        return self._action_sets[action]

    def get(self, action, default=None):
        """Return the action set, if any, associated with this action. If no action set is associated with this
        action, return the default. If no default is provided, None is used."""
        return self._action_sets.get(action, default)

    @property
    def best_prediction(self):
        """The highest value from among the predictions made by the action sets in this match set."""
        if self._best_prediction is None and self._action_sets:
            self._best_prediction = max(action_set.prediction for action_set in self._action_sets.values())
        return self._best_prediction

    @property
    def best_actions(self):
        """A tuple containing the actions whose action sets have the best prediction."""
        if self._best_actions is None:
            best_prediction = self.best_prediction
            self._best_actions = tuple(
                action
                for action, action_set in self._action_sets.items()
                if action_set.prediction == best_prediction
            )
        return self._best_actions

    def _get_selected_action(self):
        """Getter method for the selected_action property."""
        return self._selected_action

    def _set_selected_action(self, action):
        """Setter method for the selected_action property."""
        assert action in self._action_sets

        if self._selected_action is not None:
            raise ValueError("The action(s) have already been selected.")
        self._selected_action = action

    selected_action = property(
        _get_selected_action,
        _set_selected_action,
        doc="The action which was selected for execution and which deserves credit for whatever payoff is received."
    )

    @property
    def prediction(self):
        """The prediction associated with the selected action."""
        assert self._selected_action is not None
        return self._action_sets[self._selected_action].prediction


class Population:
    """A set of rules (aka classifiers), together with their associated metadata, which the LCS algorithm
    attempts to optimize using an evolutionary algorithm. This population represents the accumulated experience
    of the LCS algorithm. Each rule in the population consists of a condition which identifies which situations
    its suggestions apply to and an action which represents the suggested course of action by that rule. Each
    rule has its own associated metadata which the algorithm uses to determine how much weight should be given
    to that rule's suggestions, as well as how the population should be modified to improve future performance."""

    def __init__(self, algorithm, possible_actions):
        assert isinstance(algorithm, LCSAlgorithm)

        # The population is stored as a tiered dictionary structure of the form {condition: {action: metadata}}.
        # Storing it in this form allows the conditions to be iterated over and tested against each situation
        # exactly once, rather than repeatedly (once for each unique occurrence in a classifier).
        self._population = {}

        self._algorithm = algorithm
        self._possible_actions = frozenset(possible_actions)
        self._time_stamp = 0

    @property
    def algorithm(self):
        """The algorithm in charge of managing this population."""
        return self._algorithm

    @property
    def possible_actions(self):
        """The possible actions that can potentially be suggested by the rules in this population."""
        return self._possible_actions

    @property
    def time_stamp(self):
        """The number of iterations completed since the population was initialized."""
        return self._time_stamp

    # Defining this determines the behavior of instances of this class with respect to iteration
    # constructs such as "iter(instance)" and "for item in instance:"
    def __iter__(self):
        for condition, by_action in self._population.items():
            for action in by_action:
                yield condition, action

    # Defining this determines the behavior of len(instance)
    def __len__(self):
        return sum(len(by_action) for by_action in self._population.values())

    # Defining this determines the behavior of "item in instance"
    def __contains__(self, condition_action):
        condition, action = condition_action
        return action in self._population.get(condition, ())

    # Defining this determines the behavior of str(instance)
    def __str__(self):
        return '\n'.join(
            str(condition) + ' => ' + str(action) + '\n    ' +
            str(self.get_metadata(condition, action)).replace('\n', '\n    ')
            for condition, action in sorted(self, key=lambda condition_action: self.get_metadata(*condition_action))
        )

    def get_match_set(self, situation):
        """Accept a situation (input) and return a MatchSet containing the rules (classifiers) whose conditions
        match the situation."""

        # Find the conditions that match against the current situation, and group them according to which
        # action(s) they recommend.
        by_action = {}
        for condition, actions in self._population.items():
            if not condition(situation):
                continue

            for action, rule_metadata in actions.items():
                if action in by_action:
                    by_action[action][condition] = rule_metadata
                else:
                    by_action[action] = {condition: rule_metadata}

        # Construct the match set.
        match_set = MatchSet(self, situation, by_action)

        # If an insufficient number of actions are recommended, create some new rules (condition/action pairs)
        # until there are enough actions being recommended.
        if self._algorithm.covering_is_required(match_set):
            # Ask the algorithm to provide a new condition/action pair to add to the population, together with
            # its associated metadata.
            condition, action, metadata = self._algorithm.cover(match_set)

            # Ensure that the condition provided by the algorithm does indeed match the situation.
            assert condition(situation)

            # Add the new classifier, getting back a list of the rule(s) which had to be removed to make room for it.
            replaced = self.add(condition, action, metadata)

            # Remove the rules that were removed the population from the action set, as well. Note that
            # they may not appear in the action set, in which case nothing is done.
            for replaced_condition, replaced_action in replaced:
                if replaced_action in by_action and replaced_condition in by_action[replaced_action]:
                    del by_action[replaced_action][replaced_condition]
                    if not by_action[replaced_action]:
                        del by_action[replaced_action]

            # Add the new classifier to the action set. This is done after the replaced rules are removed,
            # just in case the algorithm provided us with a rule that was already present and was displaced.
            if action not in by_action:
                by_action[action] = {}
            by_action[action][condition] = metadata

            # Reconstruct the match set with the modifications we just made.
            match_set = MatchSet(self, situation, by_action)

        # Return the newly created match set.
        return match_set

    def add(self, condition, action, metadata=1):
        """Add a new rule to the population. The metadata argument should either be a non-negative integer or
        an instance of RuleMetadata. If it is an integer value, the rule must already exist in the population,
        and the integer value is treated as an increment to its numerosity. If it is a RuleMetadata instance,
        then behavior depends on whether the rule already exists in the population. When a rule is already
        present, the metadata's numerosity is added to that of the rule already present in the population.
        Otherwise, it is assigned as the metadata for the newly added rule. Note that this means that for
        rules already present in the population, metadata is not overwritten by newly provided values."""

        assert isinstance(metadata, RuleMetadata) or (isinstance(metadata, int) and metadata >= 0)

        if condition not in self._population:
            self._population[condition] = {}

        # If the rule already exists in the population, then we virtually add the rule
        # by incrementing the existing rule's numerosity. This prevents redundancy in
        # the rule set. Otherwise we capture the metadata and associate it with the
        # newly added rule.
        if isinstance(metadata, int):
            if condition not in self._population or action not in self._population[condition]:
                raise ValueError("Metadata must be supplied for new members of population.")
            self._population[condition][action].numerosity += metadata
        else:
            assert isinstance(metadata, RuleMetadata)
            if action in self._population[condition]:
                self._population[condition][action].numerosity += metadata.numerosity
            else:
                self._population[condition][action] = metadata

        # Any time we add a rule, we need to call this to keep the population size under control.
        return self._algorithm.prune(self)

    def remove(self, condition, action, count=1):
        """Remove one or more instances of a rule from the population. Return a Boolean indicating whether the rule's
        numerosity dropped to zero. (If the rule's numerosity was already zero, do nothing and return False.)"""

        assert isinstance(count, int) and count >= 0

        metadata = self.get_metadata(condition, action)
        if metadata is None:
            return False

        # Only actually remove the rule if its numerosity drops below 1.
        metadata.numerosity -= count
        if metadata.numerosity <= 0:
            del self._population[condition][action]
            if not self._population[condition]:
                del self._population[condition]
            return True

        return False

    def get_metadata(self, condition, action):
        """Return the metadata associated with the given classifier. If the rule is not present in the
        population, return None."""
        if condition not in self._population or action not in self._population[condition]:
            return None
        return self._population[condition][action]

    def update_time_stamp(self):
        """Update the time stamp to indicate another completed iteration of the algorithm."""
        self._time_stamp += 1


class XCSRuleMetadata(RuleMetadata):
    """Metadata used by the XCS algorithm to track the rules (classifiers) in a population. The metadata stored by
    the XCS algorithm consists of a time stamp indicating the last time the rule participated in a GA population
    update, an average reward indicating the payoff expected for this rule's suggestions, an error value
    indicating how inaccurate the reward prediction is on average, a fitness computed through a complex set of
    equations specific to XCS which is used both as the prediction weight and by the GA to determine probabilities
    of reproduction and deletion, an experience value which represents the number of times the rule's suggestion
    has been taken and its parameters have been subsequently updated, and action set size which is the average
    number of other classifiers appearing in an action set with the rule and thereby competing for the same niche,
    and a numerosity value which represents the number of (virtual) occurrences of the rule in the population."""

    def __init__(self, time_stamp, algorithm):
        assert isinstance(time_stamp, int)
        assert isinstance(algorithm, XCSAlgorithm)

        self.time_stamp = time_stamp  # The iteration of the algorithm at which this rule was last updated
        self.average_reward = algorithm.initial_prediction  # The predicted (averaged) reward for this rule
        self.error = algorithm.initial_error  # The observed error in this rule's prediction
        self.fitness = algorithm.initial_fitness  # The fitness of this rule within the GA
        self.experience = 0  # The number of times this rule has been evaluated
        self.action_set_size = 1  # The average number of rules sharing the same niche as this rule
        self.numerosity = 1  # The number of instances of this rule in the population, used to eliminate redundancy

    # Defining this sets the behavior for str(instance)
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

    # Defining this sets the behavior for instance1 < instance2
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
    """The XCS algorithm. This class defines how a population is managed by the XCS algorithm to optimize for expected
    reward and descriptive brevity. There are numerous parameters which can be modified to control the behavior of
    the algorithm:

        accuracy_coefficient (default: .1, range: (0, 1])
            Affects the size of the "cliff" between measured accuracies of inaccurate versus accurate
            classifiers. A smaller value results in a larger "cliff". The default value is good for a
            wide array of problems; only modify this if you know what you are doing.

        accuracy_power (default: 5, range: (0, +inf))
            Affects the rate at which measured accuracies of inaccurate classifiers taper off. A larger value
            results in more rapid decline in accuracy as prediction error rises. The default value is good for
            a wide array of problems; only modify this if you know what you are doing.

        crossover_probability (default: .75, range: [0, 1])
            The probability that crossover will be applied to the selected parents in a GA selection step. The
            default value is good for a wide array of problems.

        deletion_threshold (default: 20, range: [0, +inf))
            The minimum experience of a classifier before its fitness is considered in its probability
            of deletion in a GA deletion step. The higher the value, the longer new classifiers are given
            to optimize their reward predictions before being held accountable.

        discount_factor (default: .71, range: [0, 1))
            The rate at which future expected reward is discounted before being added to the current
            reward to produce the payoff value that is used to update classifier parameters such as
            reward prediction, error, and fitness. Larger values produce more far-sighted behavior;
            smaller values produce more hedonistic behavior. For problems in which the current action
            does not affect future rewards beyond the current step, set this to 0. For problems in
            which actions do affect rewards beyond the immediate iteration, set this to higher values.
            Do not set this to 1 as it will produce undefined behavior.

        do_action_set_subsumption (default: False, range: {True, False})
            Subsumption is the replacement of a classifier with another classifier, already existing
            in the population, which is a generalization of it and is considered accurate. (See notes
            on the error_threshold parameter.) This parameter determines whether subsumption is
            applied to action sets after they are selected and receive a payoff.

        do_ga_subsumption (default: False, range: {True, False})
            Subsumption is the replacement of a classifier with another classifier, already existing
            in the population, which is a generalization of it and is considered accurate. (See notes
            on the error_threshold parameter.) This parameter controls whether subsumption is applied
            to eliminate newly added classifiers in a GA selection step.

        error_threshold (default: .01, range: [0, maximum reward])
            Determines how much prediction error is tolerated before a classifier is classified as inaccurate.
            This parameter is typically set to approximately 1% of the expected range of possible rewards
            received for each action. A range of [0, 1] for reward is assumed, and the default value is
            calculated as 1% of 1 - 0, or .01. If your problem is going to have a significantly wider or
            narrower range of potential rewards, you should set this parameter appropriately for that range.

        exploration_probability (default: .5, range: [0, 1])
            The probability of choosing a suboptimal action from the suggestions made by the match set
            rather than choosing an optimal one. It is advisable to set this above 0 for all problems
            to ensure that the reward predictions converge. For problems in which on-line performance
            is important, set this value closer to 0 to focus on optimizing reward. For problems in
            which the final expression of the solution in terms of the evolved population of classifiers
            is of greater significance than optimizing on-line reward, set this to a larger value.

        exploration_strategy (default: None, range: ActionSelectionStrategy instances)
            If this is set, exploration_probability will be ignored and the action selection strategy
            provided will be used. (This is not a canonical parameter of the XCS algorithm.)

        fitness_threshold (default: .1, range: [0, 1])
            The fraction of the mean fitness of the population below which a classifier's fitness is
            considered in its probability of deletion in a GA deletion step.

        ga_threshold (default: 35, range: [0, +inf))
            Determines how often the genetic algorithm is applied to an action set. The average time
            since each classifier in the action set last participated in a GA step is computed and
            compared against this threshold; if the value is higher than the threshold, the GA is
            applied. Set this value higher to apply the GA less often, or lower to apply it more
            often.

        idealization_factor (default: 0, range: [0, 1])
            When payoff is computed, the expected future reward is multiplied by discount_factor
            and added to the actual reward before being passed to the action set. This makes XCS
            a member of the TD (temporal differences) family of reinforcement learning algorithms.
            There are two major subfamilies of algorithms for TD-based reinforcement learning, called
            SARSA and Q-learning, which estimate expected future rewards differently. SARSA uses the
            prediction of the action selected for execution on the next step as its estimate, even on
            exploration steps, while Q-learning uses the highest prediction of any of the candidate
            actions in an attempt to eliminate the negative bias introduced by exploration steps in
            learning the optimal action. Each strategy has is pros and cons. The idealization_factor
            parameter allows for mixtures of these two approaches, with 0 representing a purely SARSA-
            like approach and 1 representing a purely Q-learning-like approach. (This is not a
            canonical parameter of the XCS algorithm.)

        initial_error (default: .00001, range: (0, +inf))
            The value used to initialize the error parameter in the metadata for new classifiers. It
            is recommended that this value be a positive number close to zero. The default value is
            good for a wide variety of problems.

        initial_fitness (default: .00001, range: (0, +inf))
            The value used to initialize the fitness parameter in the metadata for new classifiers.
            It is recommended that this value be a positive number close to zero. The default value
            is good for a wide variety of problems.

        initial_prediction (default: .00001, range: [0, +inf))
            The value used to initialize the reward prediction in the metadata for new classifiers.
            It is recommended that this value be a number slightly above the minimum possible reward
            for the problem. It is assumed that the minimum reward is 0; if your problem's minimum
            reward is significantly different, this value should be set appropriately.

        learning_rate (default: .15, range: (0, 1))
            The minimum update rate for time-averaged classifier metadata parameters. A small non-
            zero value is recommended.

        max_population_size (default: 200, range: [1, +inf))
            The maximum number of classifiers permitted in the population. A larger population may
            converge to a better solution and reach a higher level of performance faster, but will
            take longer to converge to an optimal classifier set.

        minimum_actions (default: None, range: [1, +inf))
            The minimum number of actions required in a match set, below which covering occurs.
            (Covering is the random generation of a classifier that matches the current situation.)
            When this is set to None, the number of possible actions dictated by the problem is used.
            If this is set to a value greater than that, covering will occur on every step no matter
            what.

        mutation_probability (default: .03, range: [0, 1])
            XCS operates on bit-strings as its input values, and uses bit-conditions which act as
            templates to match against those bit-strings. This parameter represents the probability
            of a mutation at any location along the condition, converting a wildcard to a non-
            wildcard or vice versa, in the condition of a new classifier generated by the genetic
            algorithm. Each position in the bit condition is evaluated for mutation independently
            of the others.

        subsumption_threshold (default: 20, range: [0, +inf))
            The minimum experience of a classifier before it can subsume another classifier. This
            value should be high enough that the accuracy measure has had time to converge.

        wildcard_probability (default: .33, range: [0, 1])
            The probability of any given bit being a wildcard in the conditions of the randomly
            generated classifiers produced during covering. If this value is too low, it can
            cause perpetual churn in the population as classifiers are displaced by new ones
            with a low probability of matching future situations. If it is too high, it can
            result in the sustained presence of overly general classifiers in the population.
    """

    # TODO: Verify these before publishing v1.0.0
    # For a detailed explanation of each parameter, please see the original
    # paper by Martin Butz and Stewart Wilson.
    max_population_size = 200                               # N
    learning_rate = .15                                     # beta
    accuracy_coefficient = .1                               # alpha
    error_threshold = .01                                   # epsilon_0
    accuracy_power = 5                                      # nu
    discount_factor = .71                                   # gamma
    ga_threshold = 35                                       # theta_GA
    crossover_probability = .75                             # chi
    mutation_probability = .03                              # mu
    deletion_threshold = 20                                 # theta_del
    fitness_threshold = .1                                  # delta
    subsumption_threshold = 20                              # theta_sub
    wildcard_probability = .33                              # P_#
    initial_prediction = .00001                             # p_I
    initial_error = .00001                                  # epsilon_I
    initial_fitness = .00001                                # F_I
    exploration_probability = .5                            # p_explr
    minimum_actions = None                                  # theta_mna; None indicates total number of possible actions
    do_ga_subsumption = False                               # doGASubsumption
    do_action_set_subsumption = False                       # doActionSetSubsumption

    # If this is None, epsilon-greedy selection with epsilon == exploration_probability is used.
    # Otherwise, exploration_probability is ignored.
    exploration_strategy = None

    # This is the ratio that determines how much of the discounted future reward comes from the best prediction
    # versus the actual prediction for the next match set. For canonical XCS, this is not an available parameter
    # and should be set to 0 in that case.
    idealization_factor = 0

    @property
    def action_selection_strategy(self):
        """The action selection strategy used to govern the trade-off between exploration (acquiring new experience)
        and exploitation (utilizing existing experience to maximize reward)."""
        return self.exploration_strategy or EpsilonGreedySelectionStrategy(self.exploration_probability)

    def get_future_expectation(self, match_set):
        """Return the future reward expectation for the given match set."""

        assert isinstance(match_set, MatchSet)

        return self.discount_factor * (
            self.idealization_factor * match_set.best_prediction +
            (1 - self.idealization_factor) * match_set.prediction
        )

    def covering_is_required(self, match_set):
        """Return a Boolean value indicating whether covering is required based on the contents of the current match
        set."""

        assert isinstance(match_set, MatchSet)

        if self.minimum_actions is None:
            return len(match_set) < len(match_set.population.possible_actions)
        else:
            return len(match_set) < self.minimum_actions

    def cover(self, match_set):
        """Create a new rule that matches the given situation and return it. Preferentially choose an action that is
        not present in the existing actions, if possible."""

        assert isinstance(match_set, MatchSet)

        # Create a new condition that matches the situation.
        condition = bitstrings.BitCondition.cover(match_set.situation, self.wildcard_probability)

        # Pick a random action that (preferably) isn't already suggested by some
        # other rule for this situation.
        action_candidates = frozenset(match_set.population.possible_actions) - frozenset(match_set)
        if not action_candidates:
            action_candidates = match_set.population.possible_actions
        action = random.choice(list(action_candidates))

        # Create metadata for the new rule.
        metadata = XCSRuleMetadata(match_set.time_stamp, self)

        # The actual rule is just a condition/action/metadata triple
        return condition, action, metadata

    def distribute_payoff(self, match_set, payoff):
        """Update the rule metadata for the rules belonging to this action set, based on the payoff received."""

        assert isinstance(match_set, MatchSet)
        assert match_set.population.algorithm is self
        assert match_set.selected_action is not None

        action_set = match_set[match_set.selected_action]
        payoff = float(payoff)

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

    def update(self, match_set):
        """Update the time stamp. If sufficient time has passed, apply the genetic algorithm's operators to update the
         population."""

        assert isinstance(match_set, MatchSet)
        assert match_set.population.algorithm is self
        assert match_set.selected_action is not None

        # Increment the iteration counter.
        match_set.population.update_time_stamp()

        action_set = match_set[match_set.selected_action]

        # If the average number of iterations since the last update for each rule in the action set
        # is too small, return early instead of applying the GA.
        if match_set.population.time_stamp - self._get_average_time_stamp(action_set) <= self.ga_threshold:
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
        child1 = self._mutate(child1, action_set.situation)
        child2 = self._mutate(child2, action_set.situation)

        # If the newly generated children are already present in the population (or if they
        # should be subsumed due to GA subsumption) then simply increment the numerosities
        # of the existing rules in the population.
        new_children = []
        for child in child1, child2:
            # If the parameters specify that GA subsumption should be performed, look for an
            # accurate parent that can subsume the new child.
            if self.do_ga_subsumption:
                subsumed = False
                for parent, metadata in (parent1, parent1_metadata), (parent2, parent2_metadata):
                    if (metadata.experience > self.subsumption_threshold and
                            metadata.error < self.error_threshold and
                            parent(child)):
                        if (parent, action_set.action) in action_set.population:
                            action_set.population.add(parent, action_set.action)
                        else:
                            # Sometimes the parent is removed from a previous subsumption
                            metadata.numerosity = 1
                            action_set.population.add(parent, action_set.action, metadata)
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
                # noinspection PyTypeChecker
                action_set.population.add(child, action_set.action, metadata)

    def prune(self, population):
        """Reduce the population size, if necessary, to ensure that it does not exceed the maximum population size set
        out in the parameters."""

        assert isinstance(population, Population)
        assert population.algorithm is self

        # Determine the virtual population size.
        total_numerosity = sum(
            population.get_metadata(condition, action).numerosity
            for condition, action in population
        )

        # If the virtual population size is already small enough, just return early.
        if total_numerosity <= self.max_population_size:
            return []  # No rule's numerosity dropped to zero.

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

            if not metadata.numerosity:
                # I am a little concerned because I'm not sure how this is happening.
                # It doesn't seem to affect anything, though.
                # In all likelihood, it is just a timing issue of some sort.
                continue

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
                assert (condition, action) in population
                if population.remove(condition, action):
                    return [(condition, action)]
                else:
                    return []

        return []  # No rule's numerosity dropped to zero.

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

        # On rare occasions we have zero total accuracy. This avoids a div by zero
        total_accuracy = total_accuracy or 1

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

        # Go through each position in the condition, randomly flipping whether
        # the position is a value (0 or 1) or a wildcard (#). We do this in
        # a new list because the original condition's mask is immutable.
        mutation_points = bitstrings.BitString.random(len(condition.mask), self.mutation_probability)
        mask = condition.mask ^ mutation_points

        # The bits that aren't wildcards always have the same value as the situation,
        # which ensures that the mutated condition still matches the situation.
        if isinstance(situation, bitstrings.BitCondition):
            mask &= situation.mask
            return bitstrings.BitCondition(situation.bits, mask)
        return bitstrings.BitCondition(situation, mask)


class LCS:
    """An Learning Classifier System model instance. Create the algorithm and (optionally) a population, passing them
    in to initialize the instance. Then create a problem instance and pass it to learn()."""

    def __init__(self, algorithm, population):
        assert isinstance(algorithm, LCSAlgorithm)
        assert isinstance(population, Population)

        self._algorithm = algorithm
        self._population = population

    @property
    def algorithm(self):
        """The parameter settings used by this instance of the algorithm."""
        return self._algorithm

    @property
    def population(self):
        """The population used by this instance of the algorithm."""
        return self._population

    def run(self, problem, learn=True):
        """Run the algorithm, utilizing the population to choose the most appropriate action for each situation produced
        by the problem. If learn is True, improve the situation/action mapping to maximize reward. Otherwise,
        ignore any reward received.

        Usage:
        Create a problem instance and pass it in to this method. Problem instances must implement the OnLineProblem
        interface.
        """
        assert isinstance(problem, problems.OnLineProblem)

        previous_reward = None
        previous_match_set = None

        # Repeat until the problem has run its course.
        while problem.more():
            # Gather information about the current state of the environment.
            situation = problem.sense()

            # Determine which rules match the current situation.
            match_set = self._population.get_match_set(situation)

            # Select the best action for the current situation (or a random one,
            # if we are on an exploration step).
            match_set.selected_action = self._algorithm.action_selection_strategy(match_set)

            # Perform the selected action and find out what the received reward was.
            reward = problem.execute(match_set.selected_action)

            # Don't immediately apply the reward; instead, wait until the next iteration and
            # factor in not only the reward that was received on the previous step, but the
            # (discounted) reward that is expected going forward given the resulting situation
            # observed after the action was taken. This is a classic feature of reinforcement
            # learning algorithms, which acts to stitch together a general picture of the
            # future expected reward without actually waiting the full duration to find out
            # what it will be.
            if previous_reward is not None and learn:
                payoff = previous_reward + self._algorithm.get_future_expectation(match_set)
                self._algorithm.distribute_payoff(previous_match_set, payoff)
                self._algorithm.update(previous_match_set)
            previous_reward = reward
            previous_match_set = match_set

        # This serves to tie off the final stitch. The last action taken gets only the
        # immediate reward; there is no future reward expected.
        if previous_reward is not None and learn:
            self._algorithm.distribute_payoff(previous_match_set, previous_reward)
            self._algorithm.update(previous_match_set)


def test(algorithm=None, problem=None):
    """A quick test of the XCS algorithm, demonstrating how to use it in client code."""

    assert algorithm is None or isinstance(algorithm, LCSAlgorithm)
    assert problem is None or isinstance(problem, problems.OnLineProblem)

    import logging
    import time

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if problem is None:
        # Define the problem.
        problem = problems.MUXProblem(10000)

    if not isinstance(problem, problems.OnLineObserver):
        # Put the problem into a wrapper that will report things back to us for visibility.
        problem = problems.OnLineObserver(problem)

    if algorithm is None:
        # Define the algorithm.
        algorithm = XCSAlgorithm()
        algorithm.exploration_probability = .1
        algorithm.discount_factor = 0
        algorithm.do_ga_subsumption = True
        algorithm.do_action_set_subsumption = True

    # Create the population.
    population = Population(algorithm, problem.get_possible_actions())

    # Create the classifier system from the algorithm.
    lcs = LCS(algorithm, population)

    start_time = time.time()

    # Run the algorithm on the problem. This does two things simultaneously:
    #   1. Learns a model of the problem space from experience.
    #   2. Attempts to maximize the reward received.
    # Since initially the algorithm's model has no experience incorporated
    # into it, performance will be poor, but it will improve over time as
    # the algorithm continues to be exposed to the problem.
    lcs.run(problem)

    logger.info('Population:\n\n%s\n', lcs.population)

    end_time = time.time()

    logger.info("Total time: %.5f seconds", end_time - start_time)

    return problem.steps, problem.total_reward, end_time - start_time, lcs.population
