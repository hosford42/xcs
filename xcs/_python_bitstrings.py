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
xcs/_python_bitstrings.py
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


from .bitstrings import _BitStringBase


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

        bits = 0
        for _ in range(length):
            bits <<= 1
            bits += (random.random() < bit_prob)

        return cls(bits, length)

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
        bits = 0
        for point in points:
            if point > previous:
                bits <<= point - previous
                if include_range:
                    bits += (1 << (point - previous)) - 1
            include_range = not include_range
            previous = point
        return cls(bits, length)

    def __init__(self, bits, length=None):
        if isinstance(bits, int):
            if length is None:
                length = bits.bit_length()
            else:
                assert length >= bits.bit_length()
            if bits < 0:
                bits &= (1 << length) - 1
            hash_value = None
        elif isinstance(bits, BitString):
            # No need to make a copy because we use immutable bit arrays
            bits, length, hash_value = bits._bits, bits._length, bits._hash
        elif isinstance(bits, str):
            bit_str = bits
            bits = 0
            for char in bit_str:
                bits <<= 1
                if char == '1':
                    bits += 1
                else:
                    assert char == '0'
            if length is None:
                length = len(bit_str)
            else:
                assert length >= len(bit_str)
            hash_value = None
        else:
            bit_sequence = bits
            bits = 0
            count = 0
            for bit in bit_sequence:
                count += 1
                bits <<= 1
                if bit:
                    bits += 1
            if length is None:
                length = count
            else:
                assert length >= count
            hash_value = None

        super().__init__(bits, hash_value)
        self._length = length

    def any(self):
        """Returns True iff at least one bit is set."""
        return bool(self._bits)

    def count(self):
        """Returns the number of bits set to True in the bit string."""
        result = 0
        bits = self._bits
        while bits:
            result += bits % 2
            bits >>= 1
        return result

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # Overloads bitstring[index]
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            if start < 0:
                return BitString(0, 0)
            if step == -1:
                step = 1
                start, stop = stop, start
            if step == 1:
                length = stop - start
                bits = (self._bits >> (self._length - stop)) % (1 << length)
                return BitString(bits, length)
            else:
                return BitString([self[point] for point in range(start, stop, step)])

        assert isinstance(index, int)

        if index >= 0:
            assert index < self._length
            return (self._bits >> (self._length - index - 1)) % 2
        else:
            assert -index <= self._length
            return (self._bits >> (-index - 1)) % 2

    def __hash__(self):
        # Overloads hash(bitstring)
        if self._hash is None:
            self._hash = hash(self._bits) ^ hash(self._length)
        return self._hash

    def __eq__(self, other):
        # Overloads ==
        return isinstance(other, BitString) and self._bits == other._bits and self._length == other._length

    def __and__(self, other):
        # Overloads &
        if isinstance(other, BitString):
            assert self._length == other._length
        else:
            other = BitString(other, self._length)
        return BitString(self._bits & other._bits, self._length)

    def __or__(self, other):
        # Overloads |
        if isinstance(other, BitString):
            assert self._length == other._length
        else:
            other = BitString(other, self._length)
        return BitString(self._bits | other._bits, self._length)

    def __xor__(self, other):
        # Overloads ^
        if isinstance(other, BitString):
            assert self._length == other._length
        else:
            other = BitString(other, self._length)
        return BitString(self._bits ^ other._bits, self._length)

    def __invert__(self):
        # Overloads unary ~
        return BitString(~self._bits % (1 << self._length), self._length)

    def __add__(self, other):
        # Overloads +
        if not isinstance(other, BitString):
            other = BitString(other)
        return BitString((self._bits << other._length) + other._bits, self._length + other._length)

    @classmethod
    def from_int(cls, value, length=None):
        """Create a bit string from an integer value. If the length parameter is provided, it determines the number of
        bits in the bit string. Otherwise, the minimum length required to represent the value is used."""

        return cls(value, length)

    def __int__(self):
        # Overloads int(bitstring)
        return self._bits

    def __iter__(self):
        # Overloads iter(bitstring), and also, for bit in bitstring
        for index in range(self._length - 1, -1, -1):
            yield (self._bits >> index) % 2
