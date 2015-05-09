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

try:
    # noinspection PyProtectedMember
    from xcs._fast_bitstrings import BitString
except ImportError:
    # noinspection PyUnresolvedReferences,PyProtectedMember
    from xcs._slow_bitstrings import BitString


class BitCondition:
    """A pair of bit strings, one indicating the bit values, and the other indicating the bit mask, which together act
    as a matching template for bit strings. Like bit strings, bit conditions are hashable and immutable."""

    @classmethod
    def cover(cls, bits, wildcard_probability):
        """Create a new bit condition that matches the provided bit string, with the indicated per-index wildcard
         probability."""

        if not isinstance(bits, BitString):
            bits = BitString(bits)

        mask = BitString([random.random() < wildcard_probability for _ in range(len(bits))])
        return cls(bits, mask)

    def __init__(self, bits, mask):
        if not isinstance(bits, BitString):
            bits = BitString(bits)

        if isinstance(mask, int):
            mask = BitString.from_int(mask, len(bits))
        elif not isinstance(mask, BitString):
            mask = BitString(mask)

        if len(bits) != len(mask):
            raise ValueError("Length mismatch between bits and mask")

        self._bits = bits & mask
        self._mask = mask
        self._hash = None

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

    def __str__(self):
        return ''.join('1' if bit else ('#' if bit is None else '0') for bit in self)

    def __repr__(self):
        return type(self).__name__ + repr((self._bits, self._mask))

    def __len__(self):
        return len(self._bits)

    def __iter__(self):
        for bit, mask in zip(self._bits, self._mask):
            yield bit if mask else None

    def __getitem__(self, index):
        return self._bits[index] if self._mask[index] else None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(tuple(self))
        return self._hash

    def __eq__(self, other):
        # noinspection PyProtectedMember
        if not isinstance(other, BitCondition) or len(self._bits) != len(other._bits):
            return False
        return self._bits == other._bits and self._mask == other._mask

    def __ne__(self, other):
        return not self == other

    def __and__(self, other):
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)((self._bits | ~self._mask) & (other._bits | ~other._mask), self._mask | other._mask)

    def __or__(self, other):
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)(self._bits | other._bits, self._mask & other._mask)

    def __xor__(self, other):
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)(self._bits ^ other._bits, self._mask & other._mask)

    def __invert__(self):
        return type(self)(~self._bits, self._mask)

    def __add__(self, other):
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)(self._bits + other._bits, self._mask + other._mask)

    def __floordiv__(self, other):
        if isinstance(other, BitCondition):
            return ((self._bits ^ other._bits) | ~other._mask) & self._mask

        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)

        return (self._bits ^ other) & self._mask

    def __call__(self, other):
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

        point1 = random.randrange(len(self._bits))
        point2 = random.randrange(len(self._bits))
        if point1 > point2:
            point1, point2 = point2, point1

        bits1 = list(self._bits)
        bits2 = list(other._bits)

        mask1 = list(self._mask)
        mask2 = list(other._mask)

        for index in range(point1, point2):
            bits1[index], bits2[index] = bits2[index], bits1[index]
            mask1[index], mask2[index] = mask2[index], mask1[index]

        return type(self)(bits1, mask1), type(self)(bits2, mask2)
