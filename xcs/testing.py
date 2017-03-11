from .framework import LCSAlgorithm, ClassifierSet
from .algorithms.xcs import XCSAlgorithm

from . import scenarios


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
