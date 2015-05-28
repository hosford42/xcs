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


class BitString:
    """A hashable, immutable sequence of bits (Boolean values).

    In addition to operations for indexing and iteration, implements standard bitwise operations, including & (bitwise
    and), | (bitwise or), ^ (bitwise xor), and ~ (bitwise not). Also implements the + operator, which acts like string
    concatenation.

    A bit string can also be cast as an integer or an ordinary string.
    """

    @classmethod
    def from_int(cls, value, length=None):
        """Create a bit string from an integer value. If the length parameter is provided, it determines the number of
        bits in the bit string. Otherwise, the minimum length required to represent the value is used."""

        if not isinstance(value, int):
            raise TypeError(value)

        # Progressively chop off low-end bits from the int, adding them to the bits list,
        # until we have reached the given length (if provided) or no more non-zero bits
        # remain (if length was not specified).
        bits = []
        while value:
            if length is not None and len(bits) >= length:
                break
            bits.append(value % 2)
            value >>= 1

        # Ensure that if length was provided, we have the correct number of bits in our list.
        if length:
            if len(bits) < length:
                bits.extend([0] * (length - len(bits)))
            elif len(bits) > length:
                bits = bits[:length]

        # Reverse the order of the bits, so the high-order bits go on the left and the low-
        # order bits go on the right, just as a person would expect when looking at the
        # number written out in binary.
        bits.reverse()

        return cls(bits)

    @classmethod
    def random(cls, length, bit_prob=.5):
        """Create a bit string of the given length, with the probability of each bit being set equal to bit_prob, which
         defaults to .5."""
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
                self._bits = bits.copy()  # If it's writable, we need to make a copy
                self._bits.writeable = False  # Make sure our copy isn't writable
            else:
                self._bits = bits  # If it isn't writable, it's safe to just keep a reference
        elif isinstance(bits, int):
            self._bits = numpy.zeros(bits, bool)  # If we're just given a number, treat it as a length and fill with 0s
            self._bits.flags.writeable = False  # Make sure the bit array isn't writable
        elif isinstance(bits, BitString):
            # No need to make a copy because we use immutable bit arrays
            self._bits = bits._bits  # We can just grab a reference to the same bit array the other bitstring is using
        else:
            self._bits = numpy.array(bits, bool)  # Make a new bit array from the given values
            self._bits.flags.writeable = False  # Make sure the bit array isn't writable

        self._hash = None  # We'll calculate this later if we need it.

    def any(self):
        """Returns True iff at least one bit is set."""
        return self._bits.any()

    def count(self):
        """Returns the number of bits set to True in the bit string."""
        return int(numpy.count_nonzero(self._bits))

    def __str__(self):
        # Overloads str(bitstring)
        return ''.join('1' if bit else '0' for bit in self._bits)

    def __repr__(self):
        # Overloads repr(bitstring)
        return type(self).__name__ + '(' + repr([int(bit) for bit in self._bits]) + ')'

    def __int__(self):
        # Overloads int(bitstring)
        value = 0
        for bit in self._bits:
            value <<= 1
            value += int(bit)
        return value

    def __len__(self):
        # Overloads len(bitstring)
        return len(self._bits)

    def __iter__(self):
        # Overloads iter(bitstring), and also, for bit in bitstring
        return iter(self._bits)

    def __getitem__(self, index):
        # Overloads bitstring[index]
        return self._bits[index]

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

    def __ne__(self, other):
        # Overloads !=
        return not self == other

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
