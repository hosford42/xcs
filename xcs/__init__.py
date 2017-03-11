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

This package implements the XCS (Accuracy-based Classifier System)
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
from .framework import ActionSet, ClassifierRule, ClassifierSet, LCSAlgorithm, MatchSet
from .algorithms.xcs import XCSClassifierRule, XCSAlgorithm
from .testing import test


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
