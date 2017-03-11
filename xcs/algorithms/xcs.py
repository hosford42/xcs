import random


from .. import bitstrings
from ..framework import ClassifierRule, LCSAlgorithm, EpsilonGreedySelectionStrategy, MatchSet, ClassifierSet


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

        assert isinstance(match_set, MatchSet)
        assert match_set.model.algorithm is self

        # Create a new condition that matches the situation.
        condition = bitstrings.BitCondition.cover(
            match_set.situation,
            self.wildcard_probability
        )

        # Pick a random action that (preferably) isn't already suggested by
        # some other rule for this situation.
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

        # Update the average reward, error, and action set size of each
        # rule participating in the action set.
        for rule in action_set:
            rule.experience += 1

            update_rate = max(self.learning_rate, 1 / rule.experience)

            rule.average_reward += (
                (payoff - rule.average_reward) *
                update_rate
            )

            rule.error += (
                (abs(payoff - rule.average_reward) - rule.error) *
                update_rate

            )

            rule.action_set_size += (
                (action_set_size - rule.action_set_size) *
                update_rate
            )

        # Update the fitness of the rules.
        self._update_fitness(action_set)

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

    def _update_fitness(self, action_set):
        """Update the fitness values of the rules belonging to this action
        set."""
        # Compute the accuracy of each rule. Accuracy is inversely
        # proportional to error. Below a certain error threshold, accuracy
        # becomes constant. Accuracy values range over (0, 1].
        total_accuracy = 0
        accuracies = {}
        for rule in action_set:
            if rule.error < self.error_threshold:
                accuracy = 1
            else:
                accuracy = (
                    self.accuracy_coefficient *
                    (rule.error / self.error_threshold) **
                    -self.accuracy_power
                )
            accuracies[rule] = accuracy
            total_accuracy += accuracy * rule.numerosity

        # On rare occasions we have zero total accuracy. This avoids a div
        # by zero
        total_accuracy = total_accuracy or 1

        # Use the relative accuracies of the rules to update their fitness
        for rule in action_set:
            accuracy = accuracies[rule]
            rule.fitness += (
                self.learning_rate *
                (accuracy * rule.numerosity / total_accuracy -
                 rule.fitness)
            )

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

    @staticmethod
    def _select_parent(action_set):
        """Select a rule from this action set, with probability
        proportionate to its fitness, to act as a parent for a new rule in
        the classifier set. Return the selected rule."""
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
