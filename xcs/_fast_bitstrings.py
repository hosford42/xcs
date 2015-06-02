# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:     xcs._fast_bitstrings
# Purpose:  Fast bit-string, implemented on top of numpy bool arrays.
#
# Author:       Aaron Hosford
#
# Created:      5/9/2015
# Copyright:    (c) Aaron Hosford 2015, all rights reserved
# Licence:      Revised (3 Clause) BSD License
# -------------------------------------------------------------------------------

"""
xcs/_fast_bitstrings.py
(c) Aaron Hosford 2015, all rights reserved
Revised BSD License

Fast bit-string and bit-condition classes, implemented on top of numpy bool arrays.

This file is part of the xcs package.
"""

__author__ = 'Aaron Hosford'
__all__ = [
    'BitString',
]

import random

# If numpy isn't installed, this will raise an ImportError, causing bitstrings.py to fall back on the pure Python
# implementation defined in _slow_bitstrings.py
import numpy

# Sometimes numpy isn't uninstalled properly and what's left is an empty folder. It's unusable, but still imports
# without error. This ensures we don't take it for granted that the module is usable.
try:
    numpy.ndarray
except AttributeError:
    raise ImportError('The numpy module failed to uninstall properly.')


from xcs.bitstrings import _BitStringBase


class BitString(_BitStringBase):
    """A hashable, immutable sequence of bits (Boolean values).

    In addition to operations for indexing and iteration, implements standard bitwise operations, including & (bitwise
    and), | (bitwise or), ^ (bitwise xor), and ~ (bitwise not). Also implements the + operator, which acts like string
    concatenation.

    A bit string can also be cast as an integer or an ordinary string.
    """

    @classmethod
    def random(cls, length, bit_prob=.5):
        """Create a bit string of the given length, with the probability of each bit being set equal to bit_prob, which
         defaults to .5."""

        assert isinstance(length, int) and length >= 0
        assert isinstance(bit_prob, (int, float)) and 0 <= bit_prob <= 1

        bits = numpy.random.choice([False, True], size=(length,), p=[1-bit_prob, bit_prob])
        bits.flags.writeable = False
        return cls(bits)

    @classmethod
    def crossover_template(cls, length, points=2):
        """Create a crossover template with the given number of points. The crossover template can be used as a mask
        to crossover two bitstrings of the same length:

            assert len(parent1) == len(parent2)
            template = BitString.crossover_template(len(parent1))
            inv_template = ~template
            child1 = (parent1 & template) | (parent2 & inv_template)
            child2 = (parent1 & inv_template) | (parent2 & template)
        """

        assert isinstance(length, int) and length >= 0
        assert isinstance(points, int) and points >= 0

        points = random.sample(range(length + 1), points)
        points.sort()
        points.append(length)
        previous = 0
        include_range = bool(random.randrange(2))
        pieces = []
        for point in points:
            if point > previous:
                fill = (numpy.ones if include_range else numpy.zeros)
                pieces.append(fill(point - previous, dtype=bool))
            include_range = not include_range
            previous = point
        bits = numpy.concatenate(pieces)
        bits.flags.writeable = False
        return cls(bits)

    def __init__(self, bits):
        if isinstance(bits, numpy.ndarray) and bits.dtype == numpy.bool:
            # noinspection PyUnresolvedReferences
            if bits.flags.writeable:
                # noinspection PyNoneFunctionAssignment
                bits = bits.copy()  # If it's writable, we need to make a copy
                bits.writeable = False  # Make sure our copy isn't writable
            hash_value = None
        elif isinstance(bits, int):
            bits = numpy.zeros(bits, bool)  # If we're just given a number, treat it as a length and fill with 0s
            bits.flags.writeable = False  # Make sure the bit array isn't writable
            hash_value = None
        elif isinstance(bits, BitString):
            # No need to make a copy because we use immutable bit arrays
            # We can just grab a reference to the same bit array the other bitstring is using
            bits, hash_value = bits._bits, bits._hash
        elif isinstance(bits, str):
            bit_list = []
            for char in bits:
                if char == '1':
                    bit_list.append(True)
                elif char == '0':
                    bit_list.append(False)
                elif char == '#':
                    raise ValueError("BitStrings cannot contain wildcards. Did you mean to create a BitCondition?")
                else:
                    raise ValueError("Invalid character: " + repr(char))
            bits = numpy.array(bit_list, bool)
            bits.flags.writeable = False
            hash_value = None
        else:
            bits = numpy.array(bits, bool)  # Make a new bit array from the given values
            bits.flags.writeable = False  # Make sure the bit array isn't writable
            hash_value = None

        super().__init__(bits, hash_value)

    def any(self):
        """Returns True iff at least one bit is set."""
        return self._bits.any()

    def count(self):
        """Returns the number of bits set to True in the bit string."""
        return int(numpy.count_nonzero(self._bits))

    def __getitem__(self, index):
        # Overloads bitstring[index]
        result = self._bits[index]
        if isinstance(index, slice):
            result.flags.writeable = False
            return BitString(result)
        return result

    def __hash__(self):
        # Overloads hash(bitstring)
        # If the hash value hasn't already been calculated, do so now.
        if self._hash is None:
            self._hash = len(self._bits) + (hash(int(self)) << 13)
        return self._hash

    def __eq__(self, other):
        # Overloads ==
        # noinspection PyProtectedMember
        return isinstance(other, BitString) and numpy.array_equal(self._bits, other._bits)

    def __and__(self, other):
        # Overloads &
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = numpy.bitwise_and(self._bits, other._bits)
        # noinspection PyUnresolvedReferences
        bits.flags.writeable = False  # Make sure the bit array isn't writable so it can be used by the constructor
        return type(self)(bits)

    def __or__(self, other):
        # Overloads |
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = numpy.bitwise_or(self._bits, other._bits)
        # noinspection PyUnresolvedReferences
        bits.flags.writeable = False  # Make sure the bit array isn't writable so it can be used by the constructor
        return type(self)(bits)

    def __xor__(self, other):
        # Overloads ^
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = numpy.bitwise_xor(self._bits, other._bits)
        # noinspection PyUnresolvedReferences
        bits.flags.writeable = False  # Make sure the bit array isn't writable so it can be used by the constructor
        return type(self)(bits)

    def __invert__(self):
        # Overloads unary ~
        bits = numpy.bitwise_not(self._bits)
        # noinspection PyUnresolvedReferences
        bits.flags.writeable = False  # Make sure the bit array isn't writable so it can be used by the constructor
        return type(self)(bits)

    def __add__(self, other):
        # Overloads +
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = numpy.concatenate((self._bits, other._bits))
        bits.flags.writeable = False  # Make sure the bit array isn't writable so it can be used by the constructor
        return type(self)(bits)
