# from xcs.bitstrings import BitStringBase
# from xcs.bitstrings import *  # BitString
from xcs._python_bitstrings import BitString
from xcs._python_bitstrings import BitString
from xcs._numpy_bitstrings import BitString as bs1

class RealCarryingBitString(bs1):

    def crossover_with(self, other, points=2):
        """Perform 2-point crossover on this bit condition and another of
        the same length, returning the two resulting children.

        Usage:
            offspring1, offspring2 = condition1.crossover_with(condition2)

        Arguments:
            other: A second BitCondition of the same length as this one.
            points: An int, the number of crossover points of the
                crossover operation.
        Return:
            A tuple (condition1, condition2) of BitConditions, where the
            value at each position of this BitCondition and the other is
            preserved in one or the other of the two resulting conditions.
        """

        assert isinstance(other, type(self))
        assert len(self) == len(other)

        template = BitString.crossover_template(len(self), points)
        inv_template = ~template

        bits1 = (self._bits & template) | (other._bits & inv_template)
        mask1 = (self._mask & template) | (other._mask & inv_template)

        bits2 = (self._bits & inv_template) | (other._bits & template)
        mask2 = (self._mask & inv_template) | (other._mask & template)

        # Convert the modified sequences back into BitConditions
        return type(self)(bits1, mask1), type(self)(bits2, mask2)
