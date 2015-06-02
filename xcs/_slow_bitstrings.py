# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:     xcs._slow_bitstrings
# Purpose:  Slow bit-string and bit-condition classes, implemented using standard
#           Python data types.
#
# Author:       Aaron Hosford
#
# Created:      5/9/2015
# Copyright:    (c) Aaron Hosford 2015, all rights reserved
# Licence:      Revised (3 Clause) BSD License
# -------------------------------------------------------------------------------

"""
xcs/_slow_bitstrings.py
(c) Aaron Hosford 2015, all rights reserved
Revised BSD License

Slow bit-string and bit-condition classes, implemented using standard Python
data types.

This file is part of the xcs package.
"""

__author__ = 'Aaron Hosford'
__all__ = [
    'BitString',
]

import random


from xcs.bitstrings import _BitStringBase


class BitString(_BitStringBase):
    """A hashable, immutable sequence of bits (Boolean values). This is the slower, Python-only implementation that
    doesn't depend on numpy.

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

        bits = tuple(random.random() < bit_prob for _ in range(length))
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
        bits = []
        for point in points:
            if point > previous:
                bits.extend(include_range for _ in range(point - previous))
            include_range = not include_range
            previous = point
        return cls(bits)

    def __init__(self, bits):
        if isinstance(bits, int):
            bits = (False,) * bits
            hash_value = None
        elif isinstance(bits, BitString):
            # No need to make a copy because we use immutable bit arrays
            bits = bits._bits
            hash_value = bits._hash
        elif isinstance(bits, tuple) and all(isinstance(value, bool) for value in bits):
            self._bits = bits
            hash_value = None
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
            bits = tuple(bit_list)
            hash_value = None
        else:
            bits = tuple(bool(value) for value in bits)
            hash_value = None

        super().__init__(bits, hash_value)

    def any(self):
        """Returns True iff at least one bit is set."""
        return any(self._bits)

    def count(self):
        """Returns the number of bits set to True in the bit string."""
        return sum(self._bits)

    def __getitem__(self, index):
        # Overloads bitstring[index]
        result = self._bits[index]
        if isinstance(index, slice):
            return BitString(result)
        return result

    def __hash__(self):
        # Overloads hash(bitstring)
        if self._hash is None:
            self._hash = hash(self._bits)
        return self._hash

    def __eq__(self, other):
        # Overloads ==
        # noinspection PyProtectedMember
        return isinstance(other, BitString) and self._bits == other._bits

    def __and__(self, other):
        # Overloads &
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = tuple(my_bit and other_bit for my_bit, other_bit in zip(self._bits, other._bits))
        return type(self)(bits)

    def __or__(self, other):
        # Overloads |
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = tuple(my_bit or other_bit for my_bit, other_bit in zip(self._bits, other._bits))
        return type(self)(bits)

    def __xor__(self, other):
        # Overloads ^
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = tuple(my_bit != other_bit for my_bit, other_bit in zip(self._bits, other._bits))
        return type(self)(bits)

    def __invert__(self):
        # Overloads unary ~
        bits = tuple(not bit for bit in self._bits)
        return type(self)(bits)

    def __add__(self, other):
        # Overloads +
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = self._bits + other._bits
        return type(self)(bits)
