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

This module implements the XCS (Accuracy-based Classifier System)
algorithm, as described in the 2001 paper, "An Algorithmic Description of
XCS," by Martin Butz and Stewart Wilson.[1] The module also provides a
framework for implementing and experimenting with learning classifier
systems in general.

Usage:
    import logging
    from xcs import XCSAlgorithm
    from xcs.scenarios import MUXProblem, ScenarioObserver

    # Create a scenario instance, either by instantiating one of the
    # predefined scenarios provided in xcs.scenarios, or by creating your
    # own subclass of the xcs.scenarios.Scenario base class and
    # instantiating it.
    scenario = MUXProblem(training_cycles=50000)

    # If you want to log the process of the run as it proceeds, set the
    # logging level with the built-in logging module, and wrap the
    # scenario with an OnLineObserver.
    logging.root.setLevel(logging.INFO)
    scenario = ScenarioObserver(scenario)

    # Instantiate the algorithm and set the parameters to values that are
    # appropriate for the scenario. Calling help(XCSAlgorithm) will give
    # you a description of each parameter's meaning.
    algorithm = XCSAlgorithm()
    algorithm.exploration_probability = .1
    algorithm.do_ga_subsumption = True
    algorithm.do_action_set_subsumption = True

    # Create a classifier set from the algorithm, tailored for the
    # scenario you have selected.
    model = algorithm.new_model(scenario)

    # Run the classifier set in the scenario, optimizing it as the
    # scenario unfolds.
    model.run(scenario, learn=True)

    # Use the built-in pickle module to save/reload your model for reuse.
    import pickle
    pickle.dump(model, open('model.bin', 'wb'))
    reloaded_model = pickle.load(open('model.bin', 'rb'))

    # Or just print the results out.
    print(model)

    # Or get a quick list of the best classifiers discovered.
    for rule in model:
        if rule.fitness <= .5 or rule.experience < 10:
            continue
        print(rule.condition, '=>', rule.action, ' [%.5f]' % rule.fitness)


A quick explanation of the XCS algorithm:

    The XCS algorithm attempts to solve the reinforcement learning
    problem, which is to maximize a reward signal by learning the optimal
    mapping from inputs to outputs, where inputs are represented as
    sequences of bits and outputs are selected from a finite set of
    predetermined actions. It does so by using a genetic algorithm to
    evolve a competing population of classifier rules, of the form

        condition => action => prediction

    where the condition is a bit template (a string of 1s, 0s, and
    wildcards, represented as #s) which matches against one or more
    possible inputs, and the prediction is a floating point value that
    indicates the observed reward level when the condition matches the
    input and the indicated action is selected. The fitness of each rule in
    the classifier set is determined not by the size of the prediction, but
    by its observed accuracy, as well as by the degree to which the rule
    fills a niche that many other rules do not already fill. The reason for
    using accuracy rather than reward is that it was found that using r
    eward destabilizes the population.


More extensive help is available online at https://pythonhosted.org/xcs/


References:

[1] Butz, M. and Wilson, S. (2001). An algorithmic description of XCS. In
    Lanzi, P., Stolzmann, W., and Wilson, S., editors, Advances in
    Learning Classifier Systems: Proceedings of the Third International
    Workshop, volume 1996 of Lecture Notes in Artificial Intelligence,
    pages 253–272. Springer-Verlag Berlin Heidelberg.




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
__version__ = '1.0.0'

__all__ = [
    # Module Metadata
    '__author__',
    '__version__',

    # Preloaded Submodules
    'bitstrings',
    'scenarios',

    # Classes
    'ActionSet',
    'ClassifierRule',
    'XCSClassifierRule',
    'ClassifierSet',
    'LCSAlgorithm',
    'XCSAlgorithm',
    'MatchSet',

    # Functions
    'test',
]


import random
from abc import ABCMeta, abstractmethod


# Attempt to import numpy. If unsuccessful, set numpy = None.
try:
    import numpy
except ImportError:
    numpy = None
else:
    # This is necessary because sometimes the empty numpy folder is left in
    # place when it is uninstalled.
    try:
        numpy.ndarray
    except AttributeError:
        numpy = None


from . import bitstrings, scenarios


class ActionSelectionStrategy(metaclass=ABCMeta):
    """Abstract base class defining the minimal interface for action
    selection strategies. The action selection strategy is responsible for
    governing the trade-off between exploration (acquiring new experience)
    and exploitation (utilizing existing experience to maximize reward).
    Define a new action selection strategy by subclassing this interface,
    or by defining a function which accepts a MatchSet as its sole argument
    and returns one of the actions suggested by that match set.

    Usage:
        This is an abstract base class. Use a subclass, such as
        EpsilonGreedySelectionStrategy, to create an instance.

    Init Arguments: n/a (See appropriate subclass.)

    Callable Instance:
        Arguments:
            match_set: A MatchSet instance.
        Return:
            An action from among those suggested by match_set.
    """

    @abstractmethod
    def __call__(self, match_set):
        """Defining this allows the object to be used like a function."""
        raise NotImplementedError()


class EpsilonGreedySelectionStrategy(ActionSelectionStrategy):
    """The epsilon-greedy action selection strategy. With probability
    epsilon, an action is chosen uniformly from all possible actions
    regardless of predicted payoff. The rest of the time, the action with
    the highest predicted payoff is chosen. The probability of exploration,
    epsilon, does not change as time passes.

    Usage:
        # Define a new strategy with probability of exploration .05
        strategy = EpsilonGreedySelectionStrategy(epsilon=.05)

        # Create an algorithm instance, and tell it to use the new
        # strategy we just created.
        algorithm = XCSAlgorithm()
        algorithm.exploration_strategy = strategy

    Init Arguments:
        epsilon: The probability of exploration; default is .1.

    Callable Instance:
        Arguments:
            match_set: A MatchSet instance.
        Return:
            An action from among those suggested by match_set.
    """

    def __init__(self, epsilon=.1):
        assert 0 <= epsilon <= 1
        self.epsilon = epsilon

    def __call__(self, match_set):
        """Defining this allows the object to be used like a function."""
        assert isinstance(match_set, MatchSet)

        # With probability epsilon, select an action uniformly from all the
        # actions in the match set.
        if random.random() < self.epsilon:
            return random.choice(list(match_set))

        # Otherwise, return (one of) the best action(s)
        best_actions = match_set.best_actions
        if len(best_actions) == 1:
            return best_actions[0]
        else:
            return random.choice(best_actions)


class ClassifierRule(metaclass=ABCMeta):
    """Abstract base class defining the minimal interface for classifier
    rules appearing in classifier sets. A classifier rule consists of a
    condition and an action taken as a pair, together with associated
    metadata as determined by the algorithm. If there are multiple
    instances of a single classifier rule in a classifier set, this is
    indicated by incrementing the numerosity attribute of classifier rule,
    rather than adding another copy of it.

    Usage:
        This is an abstract base class. Use a subclass, such as
        XCSClassifierRule, to create an instance.

    Init Arguments: n/a (See appropriate subclass.)
    """

    # The number of instances of the classifier rule in its classifier set.
    numerosity = 1

    @abstractmethod
    def __str__(self):
        """Defining this determines the behavior of str(instance)"""
        raise NotImplementedError()

    @abstractmethod
    def __lt__(self, other):
        """Defining this determines the behavior of instance1 < instance2.
        This is here strictly for sorting purposes in calls to
        ClassifierSet.__str__."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def algorithm(self):
        """The algorithm associated with this classifier rule."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def condition(self):
        """The match condition for this classifier rule."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def action(self):
        """The action suggested by this classifier rule."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def prediction(self):
        """The reward prediction made by this rule. This value represents
        the reward expected if the rule's action is taken when its
        condition matches."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def prediction_weight(self):
        """The weight of this rule's predictions. This is used to resolve
        conflicting predictions made by multiple classifiers appearing in
        the same action set. The combined reward prediction for the entire
        action set is the weighted average of the predictions made by each
        classifier appearing in that action set."""
        raise NotImplementedError()


class LCSAlgorithm(metaclass=ABCMeta):
    """Abstract base class defining the minimal interface for LCS
    algorithms. To create a new algorithm that can be used to initialize a
    ClassifierSet, inherit from this class and implement each of the
    abstract methods and properties. An LCS algorithm is responsible for
    managing the population, distributing reward to the appropriate rules,
    and determining the action selection strategy that is used.

    Usage:
        This is an abstract base class. Use a subclass, such as
        XCSAlgorithm, to create an instance.

    Init Arguments: n/a (See appropriate subclass.)
    """

    def new_model(self, scenario):
        """Create and return a new classifier set initialized for handling
        the given scenario.

        Usage:
            scenario = MUXProblem()
            model = algorithm.new_model(scenario)
            model.run(scenario, learn=True)

        Arguments:
            scenario: A Scenario instance.
        Return:
            A new, untrained classifier set, suited for the given scenario.
        """
        assert isinstance(scenario, scenarios.Scenario)
        return ClassifierSet(self, scenario.get_possible_actions())

    def run(self, scenario):
        """Run the algorithm, utilizing a classifier set to choose the
        most appropriate action for each situation produced by the
        scenario. Improve the situation/action mapping on each reward
        cycle to maximize reward. Return the classifier set that was
        created.

        Usage:
            scenario = MUXProblem()
            model = algorithm.run(scenario)

        Arguments:
            scenario: A Scenario instance.
        Return:
            A new classifier set, trained on the given scenario.
        """
        assert isinstance(scenario, scenarios.Scenario)
        model = self.new_model(scenario)
        model.run(scenario, learn=True)
        return model

    @property
    @abstractmethod
    def action_selection_strategy(self):
        """The action selection strategy used to govern the trade-off
        between exploration (acquiring new experience) and exploitation
        (utilizing existing experience to maximize reward)."""
        raise NotImplementedError()

    @abstractmethod
    def get_future_expectation(self, match_set):
        """Return a numerical value representing the expected future payoff
        of the previously selected action, given only the current match
        set. The match_set argument is a MatchSet instance representing the
        current match set.

        Usage:
            match_set = model.match(situation)
            expectation = model.algorithm.get_future_expectation(match_set)
            payoff = previous_reward + discount_factor * expectation
            previous_match_set.payoff = payoff

        Arguments:
            match_set: A MatchSet instance.
        Return:
            A float, the estimate of the expected near-future payoff for
            the situation for which match_set was generated, based on the
            contents of match_set.
        """
        raise NotImplementedError()

    @abstractmethod
    def covering_is_required(self, match_set):
        """Return a Boolean indicating whether covering is required for the
        current match set. The match_set argument is a MatchSet instance
        representing the current match set before covering is applied.

        Usage:
            match_set = model.match(situation)
            if model.algorithm.covering_is_required(match_set):
                new_rule = model.algorithm.cover(match_set)
                assert new_rule.condition(situation)
                model.add(new_rule)
                match_set = model.match(situation)

        Arguments:
            match_set: A MatchSet instance.
        Return:
            A bool indicating whether match_set contains too few matching
            classifier rules and therefore needs to be augmented with a
            new one.
        """
        raise NotImplementedError()

    @abstractmethod
    def cover(self, match_set, action=None):
        """Return a new classifier rule that can be added to the match set,
        with a condition that matches the situation of the match set and an
        action selected to avoid duplication of the actions already
        contained therein. The match_set argument is a MatchSet instance
        representing the match set to which the returned rule may be added.

        Usage:
            match_set = model.match(situation)
            if model.algorithm.covering_is_required(match_set):
                new_rule = model.algorithm.cover(match_set)
                assert new_rule.condition(situation)
                model.add(new_rule)
                match_set = model.match(situation)

        Arguments:
            match_set: A MatchSet instance.
        Return:
            A new ClassifierRule instance, appropriate for the addition to
            match_set and to the classifier set from which match_set was
            drawn.
        """
        raise NotImplementedError()

    @abstractmethod
    def distribute_payoff(self, match_set):
        """Distribute the payoff received in response to the selected
        action of the given match set among the rules in the action set
        which deserve credit for recommending the action. The match_set
        argument is the MatchSet instance which suggested the selected
        action and earned the payoff.

        Usage:
            match_set = model.match(situation)
            match_set.select_action()
            match_set.payoff = reward
            model.algorithm.distribute_payoff(match_set)

        Arguments:
            match_set: A MatchSet instance for which the accumulated payoff
                needs to be distributed among its classifier rules.
        Return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, match_set):
        """Update the classifier set from which the match set was drawn,
        e.g. by applying a genetic algorithm. The match_set argument is the
        MatchSet instance whose classifier set should be updated. The
        classifier set which is to be updated can be accessed through the
        match set's model property.

        Usage:
            match_set = model.match(situation)
            match_set.select_action()
            match_set.payoff = reward
            model.algorithm.distribute_payoff(match_set)
            model.algorithm.update(match_set)

        Arguments:
            match_set: A MatchSet instance for which the classifier set
                from which it was drawn needs to be updated based on the
                match set's payoff distribution.
        Return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def prune(self, model):
        """Reduce the classifier set's population size, if necessary, by
        removing lower-quality *rules. Return a list containing any rules
        whose numerosities dropped to zero as a result of this call. (The
        list may be empty, if no rule's numerosity dropped to 0.) The
        model argument is a ClassifierSet instance which utilizes this
        algorithm.

        Usage:
            deleted_rules = model.algorithm.prune(model)

        Arguments:
            model: A ClassifierSet instance whose population may need to
                be reduced in size.
        Return:
            A possibly empty list of ClassifierRule instances which were
            removed entirely from the classifier set because their
            numerosities dropped to 0.
        """
        raise NotImplementedError()


class ActionSet:
    """A set of rules (classifiers) drawn from the same classifier set, all
    suggesting the same action and having conditions which matched the same
    situation, together with information as to the conditions under which
    the rules matched together.

    Usage:
        rules = {
            rule.condition: rule
            for rule in model
            if rule.action == action and rule.condition(situation)
        }
        action_set = ActionSet(model, situation, action, rules)

    Init Arguments:
        model: The ClassifierSet from which this action set was drawn.
        situation: The situation against which the classifier rules in this
            action set all matched.
        action: The action which the classifier rules in this action set
            collectively suggest.
        rules: A dictionary of the form {rule.condition: rule}, where each
            value is a ClassifierRule instance and its associated key is
            the condition of that rule.

    NOTE: For efficiency, the ActionSet instance uses the rules dictionary
          directly rather than making a copy. You should not modify this
          dictionary once it has been passed to the ActionSet.
    """

    def __init__(self, model, situation, action, rules):
        assert isinstance(model, ClassifierSet)
        assert isinstance(rules, dict)
        assert all(
            isinstance(rule, ClassifierRule) and
            rule.condition == condition and
            rule.condition(situation)
            for condition, rule in rules.items()
        )

        self._model = model
        self._situation = situation
        self._action = action
        self._rules = rules  # {condition: rule}

        self._prediction = None  # We'll calculate this later as needed
        self._prediction_weight = None

        # Capture the time stamp of the model at which the action set was
        # created, since this can be expected to change later.
        self._time_stamp = model.time_stamp

    @property
    def model(self):
        """The classifier set from which the classifier rules in the action
        set were drawn."""
        return self._model

    @property
    def situation(self):
        """The common situation against which all the classifier rules'
        conditions matched."""
        return self._situation

    @property
    def action(self):
        """The common action suggested by all the classifier rules in the
        action set."""
        return self._action

    @property
    def conditions(self):
        """An iterator over the conditions of the classifier rules in the
        action set."""
        return iter(self._rules)

    @property
    def time_stamp(self):
        """The time stamp of the classifier set at which this action set
        was generated."""
        return self._time_stamp

    def _compute_prediction(self):
        """Compute the combined prediction and prediction weight for this
        action set. The combined prediction is the weighted average of the
        individual predictions of the classifiers. The combined prediction
        weight is the sum of the individual prediction weights of the
        classifiers.

        Usage:
            Do not call this method directly. Use the prediction and/or
            prediction_weight properties instead.

        Arguments: None
        Return: None
        """
        total_weight = 0
        total_prediction = 0
        for rule in self._rules.values():
            total_weight += rule.prediction_weight
            total_prediction += (rule.prediction *
                                 rule.prediction_weight)
        self._prediction = total_prediction / (total_weight or 1)
        self._prediction_weight = total_weight

    @property
    def prediction(self):
        """The combined prediction of expected payoff for taking the
        suggested action given the situation. This is the weighted average
        of the individual predictions of the classifiers constituting this
        action set."""
        if self._prediction is None:
            self._compute_prediction()
        return self._prediction

    @property
    def prediction_weight(self):
        """The total weight of the combined prediction made by this action
        set. This is the sum of the weights of the individual predictions
        made by the classifiers constituting this action set."""
        if self._prediction_weight is None:
            self._compute_prediction()
        return self._prediction_weight

    def __contains__(self, rule):
        """Defining this determines the behavior of "item in instance"."""
        assert isinstance(rule, ClassifierRule)
        return (
            rule.action == self._action and
            rule.condition in self._rules
        )

    def __iter__(self):
        """Defining this determines the behavior of iter(instance)."""
        return iter(self._rules.values())

    def __len__(self):
        return len(self._rules)

    @property
    def total_numerosity(self):
        return sum(rule.numerosity for rule in self._rules.values())

    def __getitem__(self, rule):
        """Return the existing version of the classifier rule having the
        same condition and action and appearing in this action set. This
        is useful for looking up a rule to avoid duplication."""
        assert rule.action is self._action
        return self._rules[rule.condition]

    def remove(self, rule):
        """Remove this classifier rule from the action set. (Does not
        affect numerosity.) A KeyError is raised if the rule is not present
        in the action set when this method is called.

        Usage:
            if rule in action_set:
                action_set.remove(rule)

        Arguments:
            rule: The ClassifierRule instance to be removed.
        Return: None
        """
        del self._rules[rule.condition]


class MatchSet:
    """A collection of coincident action sets. This represents the set of
    all rules that matched within the same situation, organized into groups
    according to which action each rule recommends.

    Usage:
        from collections import defaultdict

        by_action = defaultdict(dict)
        for rule in model:
            if rule.condition(situation):
                by_action[action][rule.condition] = rule

        match_set = MatchSet(model, situation, by_action)

    Init Arguments:
        model: The ClassifierSet instance from which the classifier rules
            in this match set were drawn.
        situation: The situation against which the rules in this match set
            all matched.
        by_action: A 2-tiered dictionary of the form {action: {condition:
            rule}}, containing the classifier rules in this match set. The
            the values of the inner dictionary should be ClassifierRule
            instances, and for each of them,
                assert by_action[rule.action][rule.condition] is rule
            should succeed.

    NOTE: For efficiency, the MatchSet instance uses the inner dictionaries
          in by_action directly rather than making copies of them. You
          should not modify these dictionaries once they have been passed
          to the MatchSet.
    """

    def __init__(self, model, situation, by_action):
        assert isinstance(model, ClassifierSet)
        assert isinstance(by_action, dict)

        self._model = model
        self._situation = situation
        self._algorithm = model.algorithm
        self._time_stamp = model.time_stamp

        self._action_sets = {
            action: ActionSet(model, situation, action, rules)
            for action, rules in by_action.items()
        }

        self._best_actions = None
        self._best_prediction = None

        self._selected_action = None
        self._payoff = 0
        self._closed = False

    @property
    def model(self):
        """The classifier set from which this match set was drawn."""
        return self._model

    @property
    def situation(self):
        """The situation against which the rules in this match set all
        matched."""
        return self._situation

    @property
    def algorithm(self):
        """The algorithm managing the model that produced this match
        set."""
        return self._algorithm

    @property
    def time_stamp(self):
        """The time stamp of the model at which this match set was
        produced."""
        return self._time_stamp

    def __iter__(self):
        """Defining this determines the behavior of this class with respect
        to iteration, including the "iter(instance)" and "for item in
        instance:" constructs."""
        return iter(self._action_sets)

    def __len__(self):
        """Defining this determines the behavior of len(instance)."""
        return len(self._action_sets)

    def __getitem__(self, action):
        """Defining this determines the behavior of instance[key]."""
        return self._action_sets[action]

    def get(self, action, default=None):
        """Return the action set, if any, associated with this action. If
        no action set is associated with this action, return the default.
        If no default is provided, None is used.

        Usage:
            action_set = match_set.get(action)

        Arguments:
            action: The action suggested by the desired ActionSet.
            default: The value returned if no such ActionSet exists. If no
                value is provided, None is used.
        """
        return self._action_sets.get(action, default)

    @property
    def best_prediction(self):
        """The highest value from among the predictions made by the action
        sets in this match set."""
        if self._best_prediction is None and self._action_sets:
            self._best_prediction = max(
                action_set.prediction
                for action_set in self._action_sets.values()
            )
        return self._best_prediction

    @property
    def best_actions(self):
        """A tuple containing the actions whose action sets have the best
        prediction."""
        if self._best_actions is None:
            best_prediction = self.best_prediction
            self._best_actions = tuple(
                action
                for action, action_set in self._action_sets.items()
                if action_set.prediction == best_prediction
            )
        return self._best_actions

    def select_action(self):
        """Select an action according to the action selection strategy of
        the associated algorithm. If an action has already been selected,
        raise a ValueError instead.

        Usage:
            if match_set.selected_action is None:
                match_set.select_action()

        Arguments: None
        Return:
            The action that was selected by the action selection strategy.
        """
        if self._selected_action is not None:
            raise ValueError("The action has already been selected.")
        strategy = self._algorithm.action_selection_strategy
        self._selected_action = strategy(self)
        return self._selected_action

    def _get_selected_action(self):
        """Getter method for the selected_action property."""
        return self._selected_action

    def _set_selected_action(self, action):
        """Setter method for the selected_action property."""
        assert action in self._action_sets

        if self._selected_action is not None:
            raise ValueError("The action has already been selected.")
        self._selected_action = action

    selected_action = property(
        _get_selected_action,
        _set_selected_action,
        doc="""The action which was selected for execution and which
            deserves credit for whatever payoff is received. This will be
            None if no action has been selected. An action can be selected
            by calling match_set.select_action() or by assigning directly
            to this property. Note, however, that if an action has already
            been selected, attempting to assign to this property will cause
            a ValueError to be raised."""
    )

    @property
    def prediction(self):
        """The prediction associated with the selected action. If the
        action has not been selected yet, this will be None."""
        if self._selected_action is None:
            return None
        return self._action_sets[self._selected_action].prediction

    def _get_payoff(self):
        """Getter method for the payoff property."""
        return self._payoff

    def _set_payoff(self, payoff):
        """Setter method for the payoff property."""
        if self._selected_action is None:
            raise ValueError("The action has not been selected yet.")
        if self._closed:
            raise ValueError("The payoff for this match set has already"
                             "been applied.")
        self._payoff = float(payoff)

    payoff = property(
        _get_payoff,
        _set_payoff,
        doc="""The payoff received for the selected action. This starts out
            as 0 and should be assigned or incremented to reflect the total
            payoff (both immediate reward and discounted expected future
            reward) in response to the selected action. Attempting to
            modify this property before an action has been selected or
            after the payoff has been applied will result in a ValueError.
            """
    )

    def pay(self, predecessor):
        """If the predecessor is not None, gives the appropriate amount of
        payoff to the predecessor in payment for its contribution to this
        match set's expected future payoff. The predecessor argument should
        be either None or a MatchSet instance whose selected action led
        directly to this match set's situation.

        Usage:
            match_set = model.match(situation)
            match_set.pay(previous_match_set)

        Arguments:
            predecessor: The MatchSet instance which was produced by the
                same classifier set in response to the immediately
                preceding situation, or None if this is the first situation
                in the scenario.
        Return: None
        """
        assert predecessor is None or isinstance(predecessor, MatchSet)

        if predecessor is not None:
            expectation = self._algorithm.get_future_expectation(self)
            predecessor.payoff += expectation

    def apply_payoff(self):
        """Apply the payoff that has been accumulated from immediate
        reward and/or payments from successor match sets. Attempting to
        call this method before an action has been selected or after it
        has already been called for the same match set will result in a
        ValueError.

        Usage:
            match_set.select_action()
            match_set.payoff = reward
            match_set.apply_payoff()

        Arguments: None
        Return: None
        """
        if self._selected_action is None:
            raise ValueError("The action has not been selected yet.")
        if self._closed:
            raise ValueError("The payoff for this match set has already"
                             "been applied.")
        self._algorithm.distribute_payoff(self)
        self._payoff = 0
        self._algorithm.update(self)
        self._closed = True

    @property
    def closed(self):
        """A Boolean indicating whether the payoff for this match set has
        been applied. Once the payoff has been applied, attempting to
        modify or apply the payoff will result in a ValueError."""
        return self._closed


class ClassifierSet:
    """A set of classifier rules which work together to collectively
    classify inputs provided to the classifier set. The classifier set
    represents the accumulated experience of the LCS algorithm with respect
    to a particular scenario or type of scenario. Each rule in the
    classifier set consists of a condition which identifies which
    situations its suggestions apply to and an action which represents the
    suggested course of action by that rule. Each rule has its own
    associated metadata which the algorithm uses to determine how much
    weight should be given to that rule's suggestions, as well as how the
    population should evolve to improve future performance.

    Usage:
        algorithm = XCSAlgorithm()
        possible_actions = range(5)
        model = ClassifierSet(algorithm, possible_actions)

    Init Arguments:
        algorithm: The LCSAlgorithm instance which will manage this
            classifier set's population and behavior.
        possible_actions: A sequence containing the possible actions that
            may be suggested by classifier rules later appearing in this
            classifier set.
    """

    def __init__(self, algorithm, possible_actions):
        assert isinstance(algorithm, LCSAlgorithm)

        # The population is stored as a tiered dictionary structure of the
        # form {condition: {action: rule}}. Storing it in this form
        # allows the conditions to be iterated over and tested against each
        # situation exactly once, rather than repeatedly (once for each
        # unique occurrence in a classifier rule).
        self._population = {}

        self._algorithm = algorithm
        self._possible_actions = frozenset(possible_actions)
        self._time_stamp = 0

    @property
    def algorithm(self):
        """The algorithm in charge of managing this classifier set."""
        return self._algorithm

    @property
    def possible_actions(self):
        """The possible actions that can potentially be suggested at some
        point by a rule in this classifier set."""
        return self._possible_actions

    @property
    def time_stamp(self):
        """The number of training cycles completed since the classifier set
        was initialized."""
        return self._time_stamp

    def add_possible_action(self, action):
        """Add a new possible action."""
        self._possible_actions |= frozenset([action])

    def __iter__(self):
        """Defining this determines the behavior of instances of this class
        with respect to iteration constructs such as "iter(instance)" and
        "for item in instance:"."""
        for by_action in self._population.values():
            for rule in by_action.values():
                yield rule
                assert rule.numerosity > 0

    def __len__(self):
        """Defining this determines the behavior of len(instance)."""
        return sum(
            len(by_action)
            for by_action in self._population.values()
        )

    def __contains__(self, rule):
        """Defining this determines the behavior of "item in instance"."""
        assert isinstance(rule, ClassifierRule)
        assert rule.algorithm is self.algorithm
        return rule.action in self._population.get(rule.condition, ())

    def __getitem__(self, rule):
        """Defining this determines the behavior of
        "value = instance[item]" constructs."""
        if rule not in self:
            raise KeyError(rule)
        return self._population[rule.condition][rule.action]

    def __delitem__(self, rule):
        """Defining this determines the behavior of
        "del instance[item]"."""
        if rule not in self:
            raise KeyError(rule)
        self.discard(rule)

    def __str__(self):
        """Defining this determines the behavior of str(instance)."""
        return '\n'.join(str(rule) for rule in sorted(self))

    def match(self, situation):
        """Accept a situation (input) and return a MatchSet containing the
        classifier rules whose conditions match the situation. If
        appropriate per the algorithm managing this classifier set, create
        new rules to ensure sufficient coverage of the possible actions.

        Usage:
            match_set = model.match(situation)

        Arguments:
            situation: The situation for which a match set is desired.
        Return:
            A MatchSet instance for the given situation, drawn from the
            classifier rules in this classifier set.
        """

        # Find the conditions that match against the current situation, and
        # group them according to which action(s) they recommend.
        by_action = {}
        for condition, actions in self._population.items():
            if not condition(situation):
                continue

            for action, rule in actions.items():
                assert rule.numerosity > 0
                if action in by_action:
                    by_action[action][condition] = rule
                else:
                    by_action[action] = {condition: rule}

        # Construct the match set.
        match_set = MatchSet(self, situation, by_action)

        # If an insufficient number of actions are recommended, create some
        # new rules (condition/action pairs) until there are enough actions
        # being recommended.
        if self._algorithm.covering_is_required(match_set):
            # Ask the algorithm to provide a new classifier rule to add to
            # the population.
            rule = self._algorithm.cover(match_set)
            assert rule.numerosity > 0

            # Ensure that the condition provided by the algorithm does
            # indeed match the situation. If not, there is a bug in the
            # algorithm.
            assert rule.condition(situation)

            # Add the new classifier, getting back a list of the rule(s)
            # which had to be removed to make room for it.
            replaced = self.add(rule)

            # Remove the rules that were removed the population from the
            # action set, as well. Note that they may not appear in the
            # action set, in which case nothing is done.
            for replaced_rule in replaced:
                action = replaced_rule.action
                condition = replaced_rule.condition
                if action in by_action and condition in by_action[action]:
                    del by_action[action][condition]
                    if not by_action[action]:
                        del by_action[action]

            # Add the new classifier to the action set. This is done after
            # the replaced rules are removed, just in case the algorithm
            # provided us with a rule that was already present and was
            # displaced.
            if rule.action not in by_action:
                by_action[rule.action] = {}
            by_action[rule.action][rule.condition] = rule

            # Reconstruct the match set with the modifications we just
            # made.
            match_set = MatchSet(self, situation, by_action)

        # Return the newly created match set.
        return match_set

    def add(self, rule):
        """Add a new classifier rule to the classifier set. Return a list
        containing zero or more rules that were deleted from the classifier
        by the algorithm in order to make room for the new rule. The rule
        argument should be a ClassifierRule instance. The behavior of this
        method depends on whether the rule already exists in the
        classifier set. When a rule is already present, the rule's
        numerosity is added to that of the version of the rule already
        present in the population. Otherwise, the new rule is captured.
        Note that this means that for rules already present in the
        classifier set, the metadata of the existing rule is not
        overwritten by that of the one passed in as an argument.

        Usage:
            displaced_rules = model.add(rule)

        Arguments:
            rule: A ClassifierRule instance which is to be added to this
                classifier set.
        Return:
            A possibly empty list of ClassifierRule instances which were
            removed altogether from the classifier set (as opposed to
            simply having their numerosities decremented) in order to make
            room for the newly added rule.
        """

        assert isinstance(rule, ClassifierRule)
        assert rule.numerosity > 0

        condition = rule.condition
        action = rule.action

        # If the rule already exists in the population, then we virtually
        # add the rule by incrementing the existing rule's numerosity. This
        # prevents redundancy in the rule set. Otherwise we capture the
        # new rule.
        if condition not in self._population:
            self._population[condition] = {}

        if action in self._population[condition]:
            existing_rule = self._population[condition][action]
            existing_rule.numerosity += rule.numerosity
        else:
            self._population[condition][action] = rule

        # Any time we add a rule, we need to call this to keep the
        # population size under control.
        result = self._algorithm.prune(self)
        assert rule.numerosity > 0 or rule not in self
        return result

    def discard(self, rule, count=1):
        """Remove one or more instances of a rule from the classifier set.
        Return a Boolean indicating whether the rule's numerosity dropped
        to zero. (If the rule's numerosity was already zero, do nothing and
        return False.)

        Usage:
            if rule in model and model.discard(rule, count=3):
                print("Rule numerosity dropped to zero.")

        Arguments:
            rule: A ClassifierRule instance whose numerosity is to be
                decremented.
            count: An int, the size of the decrement to the rule's
                numerosity; default is 1.
        Return:
            A bool indicating whether the rule was removed altogether from
            the classifier set, as opposed to simply having its numerosity
            decremented.
        """
        assert isinstance(rule, ClassifierRule)
        assert isinstance(count, int) and count >= 0

        rule = self.get(rule)
        if rule is None:
            return False

        # Only actually remove the rule if its numerosity drops below 1.
        rule.numerosity -= count
        if rule.numerosity <= 0:
            # Ensure that if there is still a reference to this rule
            # elsewhere, its numerosity is still well-defined.
            rule.numerosity = 0

            del self._population[rule.condition][rule.action]
            if not self._population[rule.condition]:
                del self._population[rule.condition]
            return True

        return False

    def get(self, rule, default=None):
        """Return the existing version of the given rule. If the rule is
        not present in the classifier set, return the default. If no
        default was given, use None. This is useful for eliminating
        duplicate copies of rules.

        Usage:
            unique_rule = model.get(possible_duplicate, possible_duplicate)

        Arguments:
            rule: The ClassifierRule instance which may be a duplicate of
                another already contained in the classifier set.
            default: The value returned if the rule is not a duplicate of
                another already contained in the classifier set.
        Return:
            If the rule is a duplicate of another already contained in the
            classifier set, the existing one is returned. Otherwise, the
            value of default is returned.
        """
        assert isinstance(rule, ClassifierRule)

        if (rule.condition not in self._population or
                rule.action not in self._population[rule.condition]):
            return default
        return self._population[rule.condition][rule.action]

    def update_time_stamp(self):
        """Update the time stamp to indicate another completed training
        cycle.

        Usage:
            if training_cycle_complete:
                model.update_time_stamp()

        Arguments: None
        Return: None
        """
        self._time_stamp += 1

    def run(self, scenario, learn=True):
        """Run the algorithm, utilizing the classifier set to choose the
        most appropriate action for each situation produced by the
        scenario. If learn is True, improve the situation/action mapping to
        maximize reward. Otherwise, ignore any reward received.

        Usage:
            model.run(scenario, learn=True)

        Arguments:
            scenario: A Scenario instance which this classifier set is to
                interact with.
            learn: A bool indicating whether the classifier set should
                attempt to optimize its performance based on reward
                received for each action, as opposed to simply using what
                it has already learned from previous runs and ignoring
                reward received; default is True.
        Return: None
        """

        assert isinstance(scenario, scenarios.Scenario)

        previous_match_set = None

        # Repeat until the scenario has run its course.
        while scenario.more():
            # Gather information about the current state of the
            # environment.
            situation = scenario.sense()

            # Determine which rules match the current situation.
            match_set = self.match(situation)

            # Select the best action for the current situation (or a random
            # one, if we are on an exploration step).
            match_set.select_action()

            # Perform the selected action
            # and find out what the received reward was.
            reward = scenario.execute(match_set.selected_action)

            # If the scenario is dynamic, don't immediately apply the
            # reward; instead, wait until the next iteration and factor in
            # not only the reward that was received on the previous step,
            # but the (discounted) reward that is expected going forward
            # given the resulting situation observed after the action was
            # taken. This is a classic feature of temporal difference (TD)
            # algorithms, which acts to stitch together a general picture
            # of the future expected reward without actually waiting the
            # full duration to find out what it will be.
            if learn:
                # Ensure we are not trying to learn in a non-learning
                # scenario.
                assert reward is not None

                if scenario.is_dynamic:
                    if previous_match_set is not None:
                        match_set.pay(previous_match_set)
                        previous_match_set.apply_payoff()
                    match_set.payoff = reward

                    # Remember the current reward and match set for the
                    # next iteration.
                    previous_match_set = match_set
                else:
                    match_set.payoff = reward
                    match_set.apply_payoff()

        # This serves to tie off the final stitch. The last action taken
        # gets only the immediate reward; there is no future reward
        # expected.
        if learn and previous_match_set is not None:
            previous_match_set.apply_payoff()


class XCSClassifierRule(ClassifierRule):
    """This classifier rule subtype is used by the XCS algorithm. The
    metadata stored by the XCS algorithm for each classifier rule consists
    of a time stamp indicating the last time the rule participated in a GA
    population update, an average reward indicating the payoff expected for
    this rule's suggestions, an error value indicating how inaccurate the
    reward prediction is on average, a fitness computed through a complex
    set of equations specific to XCS which is used both as the prediction
    weight and by the GA to determine probabilities of reproduction and
    deletion, an experience value which represents the number of times the
    rule's suggestion has been taken and its parameters have been
    subsequently updated, and action set size which is the average number
    of other classifiers appearing in an action set with the rule and
    thereby competing for the same niche, and a numerosity value which
    represents the number of (virtual) occurrences of the rule in the
    classifier set.

    Usage:
        rule = XCSClassifierRule(
            condition=BitCondition('01##1'),
            action=random.choice(list(model.possible_actions)),
            algorithm=model.algorithm,  # An XCSAlgorithm instance
            time_stamp=model.time_stamp
        )

    Init Arguments:
        condition: The condition which this rule uses to determine whether
            it should be included in a MatchSet.
        action: The action which this rule always suggests.
        algorithm: The XCSAlgorithm managing the classifier set to which
            this rule belongs.
        time_stamp: The time stamp of the classifier set to which this
            rule belongs, as of the moment this rule is created.
    """

    def __init__(self, condition, action, algorithm, time_stamp):
        assert isinstance(algorithm, XCSAlgorithm)
        assert isinstance(time_stamp, int)

        self._algorithm = algorithm
        self._condition = condition
        self._action = action

        # The iteration of the algorithm at which this rule was last
        # updated
        self.time_stamp = time_stamp

        # The predicted (averaged) reward for this rule
        self.average_reward = algorithm.initial_prediction

        # The observed error in this rule's prediction
        self.error = algorithm.initial_error

        # The fitness of this rule within the GA
        self.fitness = algorithm.initial_fitness

        # The number of times this rule has been evaluated
        self.experience = 0

        # The average number of rules sharing the same niche as this rule
        self.action_set_size = 1

        # The number of instances of this rule in the classifier set, which
        # is used to eliminate redundancy
        self.numerosity = 1

    def __str__(self):
        """Defining this sets the behavior for str(instance)."""
        return (
            str(self.condition) + ' => ' + str(self.action) + '\n    ' +
            '\n    '.join(
                key.replace('_', ' ').title() + ': ' +
                str(getattr(self, key))
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
        )

    def __lt__(self, other):
        """Defining this sets the behavior for instance1 < instance2. This
        is here strictly for sorting purposes in calls to
        ClassifierSet.__str__."""
        if not isinstance(other, XCSClassifierRule):
            return NotImplemented
        attribute_order = (
            'numerosity',
            'fitness',
            'experience',
            'error',
            'average_reward',
            'action_set_size',
            'time_stamp'
        )
        for attribute in attribute_order:
            my_key = getattr(self, attribute)
            other_key = getattr(other, attribute)
            if my_key < other_key:
                return attribute not in ('error', 'action_set_size')
            if my_key > other_key:
                return attribute in ('error', 'action_set_size')
        return False

    @property
    def algorithm(self):
        """The algorithm associated with this classifier rule."""
        return self._algorithm

    @property
    def condition(self):
        """The match condition for this classifier rule."""
        return self._condition

    @property
    def action(self):
        """The action suggested by this classifier rule."""
        return self._action

    @property
    def prediction(self):
        """The prediction made by this rule. For XCS, this is the average
        reward received when the condition matches and the action is
        taken."""
        return self.average_reward

    @property
    def prediction_weight(self):
        """The weight of this rule's prediction as compared to others in
        the same action set. For XCS, this is the fitness of the rule."""
        return self.fitness

    def extend(self, count=1):
        """Extend the condition with the requested number of wildcards,
        which is 1 by default."""
        condition = self._condition + bitstrings.BitCondition('#' * count)
        result = type(self)(
            condition,
            self._action,
            self._algorithm,
            self.time_stamp
        )
        result.average_reward = self.average_reward
        result.error = self.error
        result.fitness = self.fitness
        result.experience = self.experience
        result.action_set_size = self.action_set_size
        result.numerosity = self.numerosity
        return result


class XCSAlgorithm(LCSAlgorithm):
    """The XCS algorithm. This class defines how a classifier set is
    managed by the XCS algorithm to optimize for expected reward and
    descriptive brevity. There are numerous parameters which can be
    modified to control the behavior of the algorithm:

        accuracy_coefficient (default: .1, range: (0, 1])
            Affects the size of the "cliff" between measured accuracies of
            inaccurate versus accurate classifiers. A smaller value results
            in a larger "cliff". The default value is good for a wide array
            of scenarios; only modify this if you know what you are doing.

        accuracy_power (default: 5, range: (0, +inf))
            Affects the rate at which measured accuracies of inaccurate
            classifiers taper off. A larger value results in more rapid
            decline in accuracy as prediction error rises. The default
            value is good for a wide array of scenarios; only modify this
            if you know what you are doing.

        crossover_probability (default: .75, range: [0, 1])
            The probability that crossover will be applied to the selected
            parents in a GA selection step. The default value is good for a
            wide array of scenarios.

        deletion_threshold (default: 20, range: [0, +inf))
            The minimum experience of a classifier before its fitness is
            considered in its probability of deletion in a GA deletion
            step. The higher the value, the longer new classifiers are
            given to optimize their reward predictions before being held
            accountable.

        discount_factor (default: .71, range: [0, 1))
            The rate at which future expected reward is discounted before
            being added to the current reward to produce the payoff value
            that is used to update classifier parameters such as reward
            prediction, error, and fitness. Larger values produce more far-
            sighted behavior; smaller values produce more hedonistic
            behavior. For scenarios in which the current action does not
            affect future rewards beyond the current step, set this to 0.
            For scenarios in which actions do affect rewards beyond the
            immediate iteration, set this to higher values. Do not set this
            to 1 as it will produce undefined behavior.

        do_action_set_subsumption (default: False, range: {True, False})
            Subsumption is the replacement of a classifier with another
            classifier, already existing in the classifier set, which is a
            generalization of it and is considered accurate. (See notes
            on the error_threshold parameter.) This parameter determines
            whether subsumption is applied to action sets after they are
            selected and receive a payoff.

        do_ga_subsumption (default: False, range: {True, False})
            Subsumption is the replacement of a classifier with another
            classifier, already existing in the classifier set, which is a
            generalization of it and is considered accurate. (See notes on
            the error_threshold parameter.) This parameter controls whether
            subsumption is applied to eliminate newly added classifiers in
            a GA selection step.

        error_threshold (default: .01, range: [0, maximum reward])
            Determines how much prediction error is tolerated before a
            classifier is classified as inaccurate. This parameter is
            typically set to approximately 1% of the expected range of
            possible rewards received for each action. A range of [0, 1]
            for reward is assumed, and the default value is calculated as
            1% of 1 - 0, or .01. If your scenario is going to have a
            significantly wider or narrower range of potential rewards, you
            should set this parameter appropriately for that range.

        exploration_probability (default: .5, range: [0, 1])
            The probability of choosing a random, potentially suboptimal
            action from the suggestions made by the match set rather than
            choosing an optimal one. It is advisable to set this above 0
            for all scenarios to ensure that the reward predictions
            converge. For scenarios in which on-line performance is
            important, set this value closer to 0 to focus on optimizing
            reward. For scenarios in which the final expression of the
            solution in terms of the evolved population of classifiers is
            of greater significance than optimizing on-line reward, set
            this to a larger value.

        exploration_strategy (default: None,
                              range: ActionSelectionStrategy instances)
            If this is set, exploration_probability will be ignored and the
            action selection strategy provided will be used. (This is not a
            canonical parameter of the XCS algorithm.)

        fitness_threshold (default: .1, range: [0, 1])
            The fraction of the mean fitness of the population below which
            a classifier's fitness is considered in its probability of
            deletion in a GA deletion step.

        ga_threshold (default: 35, range: [0, +inf))
            Determines how often the genetic algorithm is applied to an
            action set. The average time since each classifier in the
            action set last participated in a GA step is computed and
            compared against this threshold; if the value is higher than
            the threshold, the GA is applied. Set this value higher to
            apply the GA less often, or lower to apply it more often.

        idealization_factor (default: 0, range: [0, 1])
            When payoff is computed, the expected future reward is
            multiplied by discount_factor and added to the actual reward
            before being passed to the action set. This makes XCS a member
            of the TD (temporal differences) family of reinforcement
            learning algorithms. There are two major subfamilies of
            algorithms for TD-based reinforcement learning, called SARSA
            and Q-learning, which estimate expected future rewards
            differently. SARSA uses the prediction of the action selected
            for execution on the next step as its estimate, even on
            exploration steps, while Q-learning uses the highest prediction
            of any of the candidate actions in an attempt to eliminate the
            negative bias introduced by exploration steps in learning the
            optimal action. Each strategy has is pros and cons. The
            idealization_factor parameter allows for mixtures of these two
            approaches, with 0 representing a purely SARSA-like approach
            and 1 representing a purely Q-learning-like approach. (This is
            not a canonical parameter of the XCS algorithm.)

        initial_error (default: .00001, range: (0, +inf))
            The value used to initialize the error parameter in the
            metadata for new classifier rules. It is recommended that this
            value be a positive number close to zero. The default value is
            good for a wide variety of scenarios.

        initial_fitness (default: .00001, range: (0, +inf))
            The value used to initialize the fitness parameter in the
            metadata for new classifier rules. It is recommended that this
            value be a positive number close to zero. The default value is
            good for a wide variety of scenarios.

        initial_prediction (default: .00001, range: [0, +inf))
            The value used to initialize the reward prediction in the
            metadata for new classifier rules. It is recommended that this
            value be a number slightly above the minimum possible reward
            for the scenario. It is assumed that the minimum reward is 0;
            if your scenario's minimum reward is significantly different,
            this value should be set appropriately.

        learning_rate (default: .15, range: (0, 1))
            The minimum update rate for time-averaged classifier rule
            parameters. A small non-zero value is recommended.

        max_population_size (default: 200, range: [1, +inf))
            The maximum number of classifiers permitted in the classifier
            set's population. A larger population may converge to a better
            solution and reach a higher level of performance faster, but
            will take longer to converge to an optimal classifier set.

        minimum_actions (default: None, range: [1, +inf))
            The minimum number of actions required in a match set, below
            which covering occurs. (Covering is the random generation of a
            classifier that matches the current situation.) When this is
            set to None, the number of possible actions dictated by the
            scenario is used. If this is set to a value greater than that,
            covering will occur on every step no matter
            what.

        mutation_probability (default: .03, range: [0, 1])
            XCS operates on bit-strings as its input values, and uses bit-
            conditions which act as templates to match against those bit-
            strings. This parameter represents the probability of a
            mutation at any location along the condition, converting a
            wildcard to a non- wildcard or vice versa, in the condition of
            a new classifier generated by the genetic algorithm. Each
            position in the bit condition is evaluated for mutation
            independently of the others.

        subsumption_threshold (default: 20, range: [0, +inf))
            The minimum experience of a classifier before it can subsume
            another classifier. This value should be high enough that the
            accuracy measure has had time to converge.

        wildcard_probability (default: .33, range: [0, 1])
            The probability of any given bit being a wildcard in the
            conditions of the randomly generated classifiers produced
            during covering. If this value is too low, it can cause
            perpetual churn in the population as classifiers are displaced
            by new ones with a low probability of matching future
            situations. If it is too high, it can result in the sustained
            presence of overly general classifiers in the classifier set.

    Usage:
        scenario = MUXProblem()
        algorithm = XCSAlgorithm()
        algorithm.exploration_probability = .1
        model = algorithm.run(scenario)

    Init Arguments: None
    """

    # For a detailed explanation of each parameter, please see the original
    # paper, "An Algorithmic Description of XCS", by Martin Butz and
    # Stewart Wilson, and/or the documentation above.
    max_population_size = 200          # N
    learning_rate = .15                # beta
    accuracy_coefficient = .1          # alpha
    error_threshold = .01              # epsilon_0
    accuracy_power = 5                 # nu
    discount_factor = .71              # gamma
    ga_threshold = 35                  # theta_GA
    crossover_probability = .75        # chi
    mutation_probability = .03         # mu
    deletion_threshold = 20            # theta_del
    fitness_threshold = .1             # delta
    subsumption_threshold = 20         # theta_sub
    wildcard_probability = .33         # P_#
    initial_prediction = .00001        # p_I
    initial_error = .00001             # epsilon_I
    initial_fitness = .00001           # F_I
    exploration_probability = .5       # p_exp
    minimum_actions = None             # theta_mna
    do_ga_subsumption = False          # doGASubsumption
    do_action_set_subsumption = False  # doActionSetSubsumption

    # If this is None, epsilon-greedy selection with epsilon ==
    # exploration_probability is used. Otherwise, exploration_probability
    # is ignored. For canonical XCS, this is not an available parameter and
    # should be set to None.
    exploration_strategy = None

    # This is the ratio that determines how much of the discounted future
    # reward comes from the best prediction versus the actual prediction
    # for the next match set. For canonical XCS, this is not an available
    # parameter and should be set to 0 in that case.
    idealization_factor = 0

    averaged_error = True  # True for standard XCS
    relative_error = False  # False for standard XCS
    rank_based_accuracy = False  # False for standard XCS
    min_specificity = 0  # 0 for standard XCS
    max_specificity = None  # None for standard XCS
    even_coverage = False  # False for standard XCS

    @property
    def action_selection_strategy(self):
        """The action selection strategy used to govern the trade-off
        between exploration (acquiring new experience) and exploitation
        (utilizing existing experience to maximize reward)."""
        return (
            self.exploration_strategy or
            EpsilonGreedySelectionStrategy(self.exploration_probability)
        )

    def get_future_expectation(self, match_set):
        """Return a numerical value representing the expected future payoff
        of the previously selected action, given only the current match
        set. The match_set argument is a MatchSet instance representing the
        current match set.

        Usage:
            match_set = model.match(situation)
            expectation = model.algorithm.get_future_expectation(match_set)
            payoff = previous_reward + discount_factor * expectation
            previous_match_set.payoff = payoff

        Arguments:
            match_set: A MatchSet instance.
        Return:
            A float, the estimate of the expected near-future payoff for
            the situation for which match_set was generated, based on the
            contents of match_set.
        """
        assert isinstance(match_set, MatchSet)
        assert match_set.algorithm is self

        return self.discount_factor * (
            self.idealization_factor * match_set.best_prediction +
            (1 - self.idealization_factor) * match_set.prediction
        )

    def covering_is_required(self, match_set):
        """Return a Boolean indicating whether covering is required for the
        current match set. The match_set argument is a MatchSet instance
        representing the current match set before covering is applied.

        Usage:
            match_set = model.match(situation)
            if model.algorithm.covering_is_required(match_set):
                new_rule = model.algorithm.cover(match_set)
                assert new_rule.condition(situation)
                model.add(new_rule)
                match_set = model.match(situation)

        Arguments:
            match_set: A MatchSet instance.
        Return:
            A bool indicating whether match_set contains too few matching
            classifier rules and therefore needs to be augmented with a
            new one.
        """
        assert isinstance(match_set, MatchSet)
        assert match_set.algorithm is self

        if self.minimum_actions is None:
            return len(match_set) < len(match_set.model.possible_actions)
        else:
            return len(match_set) < self.minimum_actions

    def cover(self, match_set, action=None):
        """Return a new classifier rule that can be added to the match set,
        with a condition that matches the situation of the match set and an
        action selected to avoid duplication of the actions already
        contained therein. The match_set argument is a MatchSet instance
        representing the match set to which the returned rule may be added.

        Usage:
            match_set = model.match(situation)
            if model.algorithm.covering_is_required(match_set):
                new_rule = model.algorithm.cover(match_set)
                assert new_rule.condition(situation)
                model.add(new_rule)
                match_set = model.match(situation)

        Arguments:
            match_set: A MatchSet instance.
        Return:
            A new ClassifierRule instance, appropriate for the addition to
            match_set and to the classifier set from which match_set was
            drawn.
        """

        assert isinstance(match_set, MatchSet)
        assert match_set.model.algorithm is self
        assert action is None or action in match_set.model.possible_actions

        # Create a new condition that matches the situation.
        condition = bitstrings.BitCondition.cover(
            match_set.situation,
            self.wildcard_probability
        )

        if self.min_specificity or self.max_specificity is not None:
            condition = condition.limit_specificity(
                match_set.situation,
                self.min_specificity,
                self.max_specificity
            )

        if action is None:
            # Pick a random action that (preferably) isn't already
            # suggested by some other rule for this situation.
            action_candidates = (
                frozenset(match_set.model.possible_actions) -
                frozenset(match_set)
            )
            if not action_candidates:
                action_candidates = match_set.model.possible_actions
            action = random.choice(list(action_candidates))

        # Create the new rule.
        return XCSClassifierRule(
            condition,
            action,
            self,
            match_set.time_stamp
        )

    def distribute_payoff(self, match_set):
        """Distribute the payoff received in response to the selected
        action of the given match set among the rules in the action set
        which deserve credit for recommending the action. The match_set
        argument is the MatchSet instance which suggested the selected
        action and earned the payoff.

        Usage:
            match_set = model.match(situation)
            match_set.select_action()
            match_set.payoff = reward
            model.algorithm.distribute_payoff(match_set)

        Arguments:
            match_set: A MatchSet instance for which the accumulated payoff
                needs to be distributed among its classifier rules.
        Return: None
        """

        assert isinstance(match_set, MatchSet)
        assert match_set.algorithm is self
        assert match_set.selected_action is not None

        payoff = float(match_set.payoff)

        action_set = match_set[match_set.selected_action]
        action_set_size = sum(rule.numerosity for rule in action_set)

        if self.rank_based_accuracy:
            by_rank = sorted(
                action_set,
                key=lambda rule: abs(payoff - rule.average_reward)
            )
            length = len(by_rank)
            ranks = {rule: index / length
                     for index, rule in enumerate(by_rank, 1)}

        # Update the average reward, error, and action set size of each
        # rule participating in the action set.
        for rule in action_set:
            rule.experience += 1

            update_rate = max(self.learning_rate, 1 / rule.experience)

            rule.average_reward += (
                (payoff - rule.average_reward) *
                update_rate
            )

            if self.rank_based_accuracy:
                rule.error += (ranks[rule] - rule.error) * update_rate
            else:
                rule.error += (
                    (abs(payoff - rule.average_reward) - rule.error) *
                    update_rate

                )

            rule.action_set_size += (
                (action_set_size - rule.action_set_size) *
                update_rate
            )

        # Update the fitness of the rules.
        self._update_fitness(action_set, payoff)

        # If the parameters so indicate, perform action set subsumption.
        if self.do_action_set_subsumption:
            self._action_set_subsumption(action_set)

    def update(self, match_set):
        """Update the classifier set from which the match set was drawn,
        e.g. by applying a genetic algorithm. The match_set argument is the
        MatchSet instance whose classifier set should be updated.

        Usage:
            match_set = model.match(situation)
            match_set.select_action()
            match_set.payoff = reward
            model.algorithm.distribute_payoff(match_set)
            model.algorithm.update(match_set)

        Arguments:
            match_set: A MatchSet instance for which the classifier set
                from which it was drawn needs to be updated based on the
                match set's payoff distribution.
        Return: None
        """

        assert isinstance(match_set, MatchSet)
        assert match_set.model.algorithm is self
        assert match_set.selected_action is not None

        # Increment the iteration counter.
        match_set.model.update_time_stamp()

        action_set = match_set[match_set.selected_action]

        # If the average number of iterations since the last update for
        # each rule in the action set is too small, return early instead of
        # applying the GA.
        average_time_passed = (
            match_set.model.time_stamp -
            self._get_average_time_stamp(action_set)
        )
        if average_time_passed <= self.ga_threshold:
            return

        # Update the time step for each rule to indicate that they were
        # updated by the GA.
        self._set_timestamps(action_set)

        # Select two parents from the action set, with probability
        # proportionate to their fitness.
        parent1 = self._select_parent(action_set)
        parent2 = self._select_parent(action_set)

        # With the probability specified in the parameters, apply the
        # crossover operator to the parents. Otherwise, just take the
        # parents unchanged.
        if random.random() < self.crossover_probability:
            condition1, condition2 = parent1.condition.crossover_with(
                parent2.condition
            )
        else:
            condition1, condition2 = parent1.condition, parent2.condition

        # Apply the mutation operator to each child, randomly flipping
        # their mask bits with a small probability.
        condition1 = self._mutate(condition1, action_set.situation)
        condition2 = self._mutate(condition2, action_set.situation)

        if self.min_specificity or self.max_specificity is not None:
            condition1 = condition1.limit_specificity(
                match_set.situation,
                self.min_specificity,
                self.max_specificity
            )
            condition2 = condition2.limit_specificity(
                match_set.situation,
                self.min_specificity,
                self.max_specificity
            )

        # If the newly generated children are already present in the
        # population (or if they should be subsumed due to GA subsumption)
        # then simply increment the numerosities of the existing rules in
        # the population.
        new_children = []
        for condition in condition1, condition2:
            # If the parameters specify that GA subsumption should be
            # performed, look for an accurate parent that can subsume the
            # new child.
            if self.do_ga_subsumption:
                subsumed = False
                for parent in parent1, parent2:
                    should_subsume = (
                        (parent.experience >
                         self.subsumption_threshold) and
                        parent.error < self.error_threshold and
                        parent.condition(condition)
                    )
                    if should_subsume:
                        if parent in action_set.model:
                            parent.numerosity += 1
                            self.prune(action_set.model)
                        else:
                            # Sometimes the parent is removed from a
                            # previous subsumption
                            parent.numerosity = 1
                            action_set.model.add(parent)
                        subsumed = True
                        break
                if subsumed:
                    continue

            # Provided the child has not already been subsumed and it is
            # present in the population, just increment its numerosity.
            # Otherwise, if the child has neither been subsumed nor does it
            # already exist, remember it so we can add it to the classifier
            # set in just a moment.
            child = XCSClassifierRule(
                condition,
                action_set.action,
                self,
                action_set.model.time_stamp
            )
            if child in action_set.model:
                action_set.model.add(child)
            else:
                new_children.append(child)

        # If there were any children which weren't subsumed and weren't
        # already present in the classifier set, add them.
        if new_children:
            average_reward = .5 * (
                parent1.average_reward +
                parent2.average_reward
            )

            error = .5 * (parent1.error + parent2.error)

            # .1 * (average fitness of parents)
            fitness = .05 * (
                parent1.fitness +
                parent2.fitness
            )

            for child in new_children:
                child.average_reward = average_reward
                child.error = error
                child.fitness = fitness
                action_set.model.add(child)

    def prune(self, model):
        """Reduce the classifier set's population size, if necessary, by
        removing lower-quality *rules. Return a list containing any rules
        whose numerosities dropped to zero as a result of this call. (The
        list may be empty, if no rule's numerosity dropped to 0.) The
        model argument is a ClassifierSet instance which utilizes this
        algorithm.

        Usage:
            deleted_rules = model.algorithm.prune(model)

        Arguments:
            model: A ClassifierSet instance whose population may need to
                be reduced in size.
        Return:
            A possibly empty list of ClassifierRule instances which were
            removed entirely from the classifier set because their
            numerosities dropped to 0.
        """

        assert isinstance(model, ClassifierSet)
        assert model.algorithm is self

        # Determine the (virtual) population size.
        total_numerosity = sum(rule.numerosity for rule in model)

        # If the population size is already small enough, just return early
        if total_numerosity <= self.max_population_size:
            return []  # No rule's numerosity dropped to zero.

        # Determine the average fitness of the rules in the population.
        total_fitness = sum(rule.fitness for rule in model)
        average_fitness = total_fitness / total_numerosity

        # Determine the probability of deletion, as a function of both
        # accuracy and niche sparsity.
        total_votes = 0
        deletion_votes = {}
        for rule in model:
            vote = rule.action_set_size * rule.numerosity

            sufficient_experience = (
                rule.experience > self.deletion_threshold
            )
            low_fitness = (
                rule.fitness / rule.numerosity <
                self.fitness_threshold * average_fitness
            )
            if sufficient_experience and low_fitness:
                vote *= average_fitness / (rule.fitness /
                                           rule.numerosity)

            deletion_votes[rule] = vote
            total_votes += vote

        # Choose a rule to delete based on the probabilities just computed.
        selector = random.uniform(0, total_votes)
        for rule, vote in deletion_votes.items():
            selector -= vote
            if selector <= 0:
                assert rule in model
                if model.discard(rule):
                    return [rule]
                else:
                    return []

        assert False  # We should never reach this point.

    def _update_fitness(self, action_set, payoff):
        """Update the fitness values of the rules belonging to this action
        set."""

        if self.averaged_error:
            errors = {rule: rule.error for rule in action_set}
        elif self.rank_based_accuracy:
            by_rank = sorted(
                action_set,
                key=lambda rule: abs(rule.prediction - payoff)
            )
            length = len(by_rank)
            errors = {rule: index / length
                      for index, rule in enumerate(by_rank, 1)}
        else:
            errors = {rule: abs(rule.prediction - payoff)
                      for rule in action_set}

        # Compute the accuracy of each rule.
        if self.relative_error:
            min_error = min(errors.values())
            max_error = max(errors.values())
            if min_error >= max_error:
                #max_error += 1
                min_error -= 1
            ratio = 1 / (max_error - min_error)
            accuracies = {rule: (max_error - error) * ratio
                          for rule, error in errors.items()}
            total_accuracy = sum(
                accuracy * rule.numerosity
                for rule, accuracy in accuracies.items()
            )
        else:
            # Accuracy is inversely proportional to error. Below a certain
            # error threshold, accuracy becomes constant. Accuracy values
            # range over (0, 1].
            accuracies = {}
            total_accuracy = 0
            for rule in action_set:
                error = errors[rule]
                if error < self.error_threshold:
                    accuracy = 1
                else:
                    accuracy = (
                        self.accuracy_coefficient *
                        (error / self.error_threshold) **
                        -self.accuracy_power
                    )
                accuracies[rule] = accuracy
                total_accuracy += accuracy * rule.numerosity

        # Prevent divide by 0
        total_accuracy = total_accuracy or 1

        # Use the relative accuracies of the rules to update their fitness
        for rule in action_set:
            accuracy = accuracies[rule]
            fitness_target = \
                accuracy * rule.numerosity / total_accuracy
            rule.fitness += \
                self.learning_rate * (fitness_target - rule.fitness)

    def _action_set_subsumption(self, action_set):
        """Perform action set subsumption."""
        # Select a condition with maximum bit count among those having
        # sufficient experience and sufficiently low error.
        selected_rule = None
        selected_bit_count = None
        for rule in action_set:
            if not (rule.experience > self.subsumption_threshold and
                    rule.error < self.error_threshold):
                continue
            bit_count = rule.condition.count()
            if (selected_rule is None or
                    bit_count > selected_bit_count or
                    (bit_count == selected_bit_count and
                     random.randrange(2))):
                selected_rule = rule
                selected_bit_count = bit_count

        # If no rule was found satisfying the requirements, return
        # early.
        if selected_rule is None:
            return

        # Subsume each rule which the selected rule generalizes. When a
        # rule is subsumed, all instances of the subsumed rule are replaced
        # with instances of the more general one in the population.
        to_remove = []
        for rule in action_set:
            if (selected_rule is not rule and
                    selected_rule.condition(rule.condition)):
                selected_rule.numerosity += rule.numerosity
                action_set.model.discard(rule, rule.numerosity)
                to_remove.append(rule)
        for rule in to_remove:
            action_set.remove(rule)

    @staticmethod
    def _get_average_time_stamp(action_set):
        """Return the average time stamp for the rules in this action
        set."""
        # This is the average value of the iteration counter upon the most
        # recent update of each rule in this action set.
        total_time_stamps = sum(rule.time_stamp * rule.numerosity
                                for rule in action_set)
        total_numerosity = sum(rule.numerosity for rule in action_set)
        return total_time_stamps / (total_numerosity or 1)

    @staticmethod
    def _set_timestamps(action_set):
        """Set the time stamp of each rule in this action set to the given
        value."""
        # Indicate that each rule has been updated at the given iteration.
        for rule in action_set:
            rule.time_stamp = action_set.model.time_stamp

    def _select_parent(self, action_set):
        """Select a rule from this action set, with probability
        proportionate to its fitness, to act as a parent for a new rule in
        the classifier set. Return the selected rule."""
        if self.even_coverage:
            situation_length = len(action_set.situation)
            counts = ([0] * situation_length, [0] * situation_length)
            for rule in action_set.model:
                for index, bit in enumerate(rule.condition):
                    if bit is not None:
                        counts[bit][index] += 1

            least_balanced_bits = [
                counts[1][index] < counts[0][index]
                for index in range(situation_length)
            ]

            balance_weights = (
                abs(counts[1][index] - counts[0][index]) /
                (1 + counts[0][index] + counts[1][index])
                for index in range(situation_length)
            )

            selection_weights = {
                rule: rule.fitness +
                      (.5 + sum(
                          (weight * (condition_bit == least_balanced_bit))
                          for weight, condition_bit, least_balanced_bit in
                          zip(balance_weights,
                              rule.condition,
                              least_balanced_bits)
                      )) /
                      (1 + sum(
                          weight
                          for weight, condition_bit in zip(
                              balance_weights,
                              rule.condition
                          )
                          if condition_bit is not None
                      ))
                for rule in action_set
            }

            total_weight = sum(selection_weights.values())
            selector = random.uniform(0, total_weight)
            for rule, weight in selection_weights.items():
                selector -= weight
                if selector <= 0:
                    return rule

        total_fitness = sum(rule.fitness for rule in action_set)
        selector = random.uniform(0, total_fitness)
        for rule in action_set:
            selector -= rule.fitness
            if selector <= 0:
                return rule

        # If for some reason a case slips through the above loop, perhaps
        # due to floating point error, we fall back on uniform selection.
        return random.choice(list(action_set))

    def _mutate(self, condition, situation):
        """Create a new condition from the given one by probabilistically
        applying point-wise mutations. Bits that were originally wildcarded
        in the parent condition acquire their values from the provided
        situation, to ensure the child condition continues to match it."""

        # Go through each position in the condition, randomly flipping
        # whether the position is a value (0 or 1) or a wildcard (#). We do
        # this in a new list because the original condition's mask is
        # immutable.
        mutation_points = bitstrings.BitString.random(
            len(condition.mask),
            self.mutation_probability
        )
        mask = condition.mask ^ mutation_points

        # The bits that aren't wildcards always have the same value as the
        # situation, which ensures that the mutated condition still matches
        # the situation.
        if isinstance(situation, bitstrings.BitCondition):
            mask &= situation.mask
            return bitstrings.BitCondition(situation.bits, mask)
        return bitstrings.BitCondition(situation, mask)


def test(algorithm=None, scenario=None):
    """Run the algorithm on the scenario, creating a new classifier set in
    the process. Log the performance as the scenario unfolds. Return a
    tuple, (total_steps, total_reward, total_seconds, model), indicating
    the performance of the algorithm in the scenario and the resulting
    classifier set that was produced. By default, the algorithm used is a
    new XCSAlgorithm instance with exploration probability .1 and GA and
    action set subsumption turned on, and the scenario is a MUXProblem
    instance with 10,000 reward cycles.

    Usage:
        algorithm = XCSAlgorithm()
        scenario = HaystackProblem()
        steps, reward, seconds, model = test(algorithm, scenario)

    Arguments:
        algorithm: The LCSAlgorithm instance which should be run; default
            is a new XCSAlgorithm instance with exploration probability
            set to .1 and GA and action set subsumption turned on.
        scenario: The Scenario instance which the algorithm should be run
            on; default is a MUXProblem instance with 10,000 training
            cycles.
    Return:
        A tuple, (total_steps, total_reward, total_time, model), where
        total_steps is the number of training cycles executed, total_reward
        is the total reward received summed over all executed training
        cycles, total_time is the time in seconds from start to end of the
        call to model.run(), and model is the ClassifierSet instance that
        was created and trained.
    """

    assert algorithm is None or isinstance(algorithm, LCSAlgorithm)
    assert scenario is None or isinstance(scenario, scenarios.Scenario)

    import logging
    import time

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if scenario is None:
        # Define the scenario.
        scenario = scenarios.MUXProblem(10000)

    if not isinstance(scenario, scenarios.ScenarioObserver):
        # Put the scenario into a wrapper that will report things back to
        # us for visibility.
        scenario = scenarios.ScenarioObserver(scenario)

    if algorithm is None:
        # Define the algorithm.
        algorithm = XCSAlgorithm()
        algorithm.exploration_probability = .1
        algorithm.do_ga_subsumption = True
        algorithm.do_action_set_subsumption = True

    assert isinstance(algorithm, LCSAlgorithm)
    assert isinstance(scenario, scenarios.ScenarioObserver)

    # Create the classifier system from the algorithm.
    model = ClassifierSet(algorithm, scenario.get_possible_actions())

    # Run the algorithm on the scenario. This does two things
    # simultaneously:
    #   1. Learns a model of the problem space from experience.
    #   2. Attempts to maximize the reward received.
    # Since initially the algorithm's model has no experience incorporated
    # into it, performance will be poor, but it will improve over time as
    # the algorithm continues to be exposed to the scenario.
    start_time = time.time()
    model.run(scenario, learn=True)
    end_time = time.time()

    logger.info('Classifiers:\n\n%s\n', model)
    logger.info("Total time: %.5f seconds", end_time - start_time)

    return (
        scenario.steps,
        scenario.total_reward,
        end_time - start_time,
        model
    )
