To Do
=====

* Finish migrating this list over to Issues for proper tracking.
* Add more algorithms and variants:
  * ACS (Anticipatory Classifier System)
  * ZCS (Zeroth Level Classifier System)
  * MCS (Minimal Classifier System)
  * MCS (Multiple Classifier System, yes same acronym)
  * UCS (Supervised Classifier System, as best as I can tell)
  * etc. (Add them to this list as you think of them.)
* Add proper logging, instead of the silly print statements currently in place
* Everything could use a code review and more extensive testing
* Add methods for saving and loading the population. Otherwise, what good is any of this?
* Add more benchmarking problems.
* Create an abstract interface for the algorithms. Ideally, not only should XCS variants conform to this interface, but so 
  should any reinforcement learning algorithm.
* Speed up code as much as possible by taking full advantage of numpy. Consider generating a compiled binary (.pyd/.so) for
  speeding things up, but keep the pure Python implementations around for anyone who would like to experiment by modifying
  an algorithm. Maybe Cython would be a good middle ground?
* Add a wrapper class that can extend any bit-based reinforcement learning algorithm to one that can accept arbitrary
  inputs and utilize arbitrary input features. This should encompass both simple extensions such as setting/clearing a
  bit in the input bitstring if an input float value is in/out of a particular range, and more complex extensions such as
  utilizing an evolving set of s-expressions produced by genetic programming.
* In support of the abovementioned wrapper class for extending the algorithms to new input types with potentially changing
  bit meanings, add methods for slicing and merging a population along input bit indices. For example, suppose we want to
  delete input bit 4 from all rule conditions because it corresponds to a feature that is being dropped. If the appropriate
  meta-operators are defined for the population, they can propagate their changes down to affect every single condition in
  the same way... something along the lines of: pop.conditions = pop.conditions[:4] + pop.conditions[5:], where the
  conditions property returns a placeholder object that represents the combined set of all conditions in the population
  taken as a unit to permit operations on them.
* Flesh out the [FAQ](https://github.com/hosford42/xcs/wiki/FAQ) and the remainder of the [Wiki](https://github.com/hosford42/xcs/wiki), and then add links to those pages to the documentation in the packaged distribution.
