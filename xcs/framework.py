import random
from abc import ABCMeta, abstractmethod


from . import scenarios


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
    def cover(self, match_set):
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

    def __iter__(self):
        """Defining this determines the behavior of instances of this class
        with respect to iteration constructs such as "iter(instance)" and
        "for item in instance:"."""
        for by_action in self._population.values():
            for rule in by_action.values():
                yield rule

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
        return self._algorithm.prune(self)

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
