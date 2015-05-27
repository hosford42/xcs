# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:     xcs.bitstrings
# Purpose:  Implements bit-string and bit-condition data types used by the XCS
#           algorithm.
#
# Author:       Aaron Hosford
#
# Created:      5/5/2015
# Copyright:    (c) Aaron Hosford 2015, all rights reserved
# Licence:      Revised (3 Clause) BSD License
# -------------------------------------------------------------------------------

"""
xcs/bitstrings.py
(c) Aaron Hosford 2015, all rights reserved
Revised BSD License

Provides bit-string and bit-condition data types used by the XCS algorithm.

This file is part of the public API of the xcs package.
"""

__author__ = 'Aaron Hosford'
__all__ = [
    'BitString',
    'BitCondition',
]

import random


# We have two different implementations of BitString, one in _fast_bitstrings and one in _slow_bitstrings. The fast
# version is dependent on numpy being installed, whereas the slow one is written in pure Python with no external
# dependencies. This attempts to load the fast version and, failing that, falls back on the slower one.
try:
    # noinspection PyProtectedMember
    from xcs._fast_bitstrings import BitString
except ImportError:
    # noinspection PyUnresolvedReferences,PyProtectedMember
    from xcs._slow_bitstrings import BitString


class BitCondition:
    """A pair of bit strings, one indicating the bit values, and the other indicating the bit mask, which together act
    as a matching template for bit strings. Like bit strings, bit conditions are hashable and immutable. Think of
    BitConditions as patterns which can match against BitStrings of the same length. At each index, we can have a 1,
    a 0, or a #. If the value is 1 or 0, the BitString must have the same value at that index. If the value is #, the
    BitString can have any value at that index.

    BitConditions are matched against BitStrings in one of two ways:
        Method 1:
            result = condition // bitstring
            # result now contains a new BitString which contains a 1 for each position that violated the pattern, and
            # a 0 for each position that did not. This tells us exactly where the condition and the bitstring disagree
        Method 2:
            result = condition(bitstring)
            # result now contains a single Boolean value which is True if the bitstring fully satisfies the pattern
            # specified by the condition, or False if the bitstring disagrees with the condition at at least one index

    BitConditions can also match against other BitConditions in the same way that they are matched against BitStrings,
    with the sole exception that if the condition being used as the pattern specifies a 1 or 0 at a particular index,
    and the condition being used as the substrate contains an # at that point, the match fails.
    """

    @classmethod
    def cover(cls, bits, wildcard_probability):
        """Create a new bit condition that matches the provided bit string, with the indicated per-index wildcard
         probability."""

        if not isinstance(bits, BitString):
            bits = BitString(bits)

        mask = BitString([random.random() < wildcard_probability for _ in range(len(bits))])
        return cls(bits, mask)

    def __init__(self, bits, mask):
        # Convert bits to a bit string if it isn't one already
        if not isinstance(bits, BitString):
            bits = BitString(bits)

        # Convert mask to a bit string of the same length as bits
        if isinstance(mask, int):
            mask = BitString.from_int(mask, len(bits))
        elif not isinstance(mask, BitString):
            mask = BitString(mask)

        # Verify the bits and mask bit strings have the same length
        if len(bits) != len(mask):
            raise ValueError("Length mismatch between bits and mask")

        self._bits = bits & mask
        self._mask = mask
        self._hash = None  # We will calculate this later if it is needed.

    @property
    def bits(self):
        """The bit string indicating the bit values of this bit condition. Indices that are wildcarded will have a
        value of False."""
        return self._bits

    @property
    def mask(self):
        """The bit string indicating the bit mask. A value of True for a bit indicates it must match the value bit
        string. A value of False indicates it is masked/wildcarded."""
        return self._mask

    def count(self):
        return self._mask.count()

    def __str__(self):
        # Overloads str(condition)
        return ''.join('1' if bit else ('#' if bit is None else '0') for bit in self)

    def __repr__(self):
        # Overloads repr(condition)
        return type(self).__name__ + repr((self._bits, self._mask))

    def __len__(self):
        # Overloads len(condition)
        return len(self._bits)

    def __iter__(self):
        # Overloads iter(condition), and also, for bit in condition
        # The values yielded by the iterator are True (1), False (0), or None (#)
        for bit, mask in zip(self._bits, self._mask):
            yield bit if mask else None

    def __getitem__(self, index):
        # Overloads condition[index]
        # The values yielded by the index operator are True (1), False (0), or None (#)
        return self._bits[index] if self._mask[index] else None

    def __hash__(self):
        # Overloads hash(condition)
        # If we haven't already calculated the hash value, do so now.
        if self._hash is None:
            self._hash = hash(tuple(self))
        return self._hash

    def __eq__(self, other):
        # Overloads ==
        # noinspection PyProtectedMember
        if not isinstance(other, BitCondition) or len(self._bits) != len(other._bits):
            return False
        return self._bits == other._bits and self._mask == other._mask

    def __ne__(self, other):
        # Overloads !=
        return not self == other

    def __and__(self, other):
        # Overloads &
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)((self._bits | ~self._mask) & (other._bits | ~other._mask), self._mask | other._mask)

    def __or__(self, other):
        # Overloads |
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)(self._bits | other._bits, self._mask & other._mask)

    def __xor__(self, other):
        # Overloads ^
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)(self._bits ^ other._bits, self._mask & other._mask)

    def __invert__(self):
        # Overloads unary ~
        return type(self)(~self._bits, self._mask)

    def __add__(self, other):
        # Overloads +
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)(self._bits + other._bits, self._mask + other._mask)

    def __floordiv__(self, other):
        # Overloads the // operator, which we use to find the indices in the other value that do/can disagree
        # with this condition.
        if isinstance(other, BitCondition):
            return ((self._bits ^ other._bits) | ~other._mask) & self._mask

        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)

        return (self._bits ^ other) & self._mask

    def __call__(self, other):
        # Overloads condition(bitstring)
        # Returns a Boolean value that indicates whether the other value satisfies this condition.
        mismatches = self // other
        return not mismatches.any()

    def crossover_with(self, other):
        """Perform 2-point crossover on this bit condition and another of the same length, returning the two resulting
        children."""

        if not isinstance(other, BitCondition):
            raise TypeError(other)
        if len(self) != len(other):
            raise ValueError(other)

        # TODO: Revamp this to take advantage of numpy array speeds

        # Select two crossover points with point1 <= point2
        point1 = random.randrange(len(self._bits))
        point2 = random.randrange(len(self._bits))
        if point1 > point2:
            point1, point2 = point2, point1

        # Convert the two conditions into list form so we can modify them; remember BitConditions are immutable.
        bits1 = list(self._bits)
        mask1 = list(self._mask)
        bits2 = list(other._bits)
        mask2 = list(other._mask)

        # Perform the crossover, swapping the values of the two conditions for each index such that
        # point1 <= index < point2
        for index in range(point1, point2):
            bits1[index], bits2[index] = bits2[index], bits1[index]
            mask1[index], mask2[index] = mask2[index], mask1[index]

        # Convert the modified sequences back into BitConditions
        return type(self)(bits1, mask1), type(self)(bits2, mask2)
