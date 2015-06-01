# XCS

*Accuracy-based Learning Classifier Systems for Python 3*

## Links
* [Project Home](http://hosford42.github.io/xcs/)
* [Tutorial](https://pythonhosted.org/xcs/)
* [Source](https://github.com/hosford42/xcs)
* [Distribution](https://pypi.python.org/pypi/xcs)

The package is available for download under the permissive [Revised BSD License](https://github.com/hosford42/xcs/blob/master/LICENSE).

## Description
XCS is a Python 3 implementation of the XCS algorithm as described in the 2001 paper, [An Algorithmic Description of XCS](http://link.springer.com/chapter/10.1007/3-540-44640-0_15), by 
[Martin Butz](http://www.uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/cognitive-modeling/staff/staff/martin-v-butz.html) and [Stewart Wilson](http://prediction-dynamics.com/). XCS is a type of [Learning Classifier System (LCS)](http://en.wikipedia.org/wiki/Learning_classifier_system), a [machine learning](http://en.wikipedia.org/wiki/Machine_learning) algorithm that utilizes a [genetic algorithm](http://en.wikipedia.org/wiki/Genetic_algorithm) acting on a rule-based system, to solve a [reinforcement learning](http://en.wikipedia.org/wiki/Reinforcement_learning) problem.

In its canonical form, XCS accepts a fixed-width string of bits as its input, and attempts to select the best action from a predetermined list of choices using an evolving set of rules that match inputs and offer appropriate suggestions. It then receives a reward signal indicating the quality of its decision, which it uses to adjust the rule set that was used to make the decision. This process is subsequently repeated, allowing the algorithm to evaluate the changes it has already made and further refine the rule set.

A key feature of XCS is that, unlike many other machine learning algorithms, it not only learns the optimal input/output mapping, but also produces a minimal set of rules for describing that mapping. This is a big advantage over other learning algorithms such as [neural networks](http://en.wikipedia.org/wiki/Artificial_neural_network) whose models are largely opaque to human analysis, making XCS an important tool in any data scientist's tool belt.

The XCS library provides not only an implementation of the standard XCS algorithm, but a set of interfaces which together constitute a framework for implementing and experimenting with other LCS variants. Future plans for the XCS library include continued expansion of the tool set with additional algorithms, and refinement of the interface to support reinforcement learning algorithms in general.

## Related Projects
* Pier Luca Lanzi's [XCS Library (xcslib)](http://xcslib.sourceforge.net/) (C++)
* Ryan J. Urbanowicz's [LCS Implementations for SNP Environment](http://gbml.org/2010/03/24/python-lcs-implementations-xcs-ucs-mcs-for-snp-environment/) and [ExSTraCS](http://www.sourceforge.net/projects/exstracs/) (Python)
* Martin Butz's [JavaXCSF](http://www.cm.inf.uni-tuebingen.de/Code) (Java)
