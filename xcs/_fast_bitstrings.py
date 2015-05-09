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

import numpy


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

        bits = []
        while value:
            if length is not None and len(bits) >= length:
                break
            bits.append(value % 2)
            value >>= 1

        if length:
            if len(bits) < length:
                bits.extend([0] * (length - len(bits)))
            elif len(bits) > length:
                bits = bits[:length]

        bits.reverse()

        return cls(bits)

    def __init__(self, bits):
        if isinstance(bits, numpy.ndarray) and bits.dtype == numpy.bool:
            if bits.flags.writeable:
                self._bits = bits.copy()
                self._bits.writeable = False
            else:
                self._bits = bits
        elif isinstance(bits, int):
            self._bits = numpy.zeros(bits, bool)
            self._bits.flags.writeable = False
        elif isinstance(bits, BitString):
            # No need to make a copy because we use immutable bit arrays
            self._bits = bits._bits
        else:
            self._bits = numpy.array(bits, bool)
            self._bits.flags.writeable = False

        self._hash = None

#    @property
#    def bits(self):
#        """The numpy array containing the actual bits of the bit string. Note that the array is immutable."""
#        # Safe because we use immutable bit arrays
#        return self._bits

    def any(self):
        '''Returns True iff at least one bit is set.'''
        return self._bits.any()

    def __str__(self):
        return ''.join('1' if bit else '0' for bit in self._bits)

    def __repr__(self):
        return type(self).__name__ + '(' + repr([int(bit) for bit in self._bits]) + ')'

    def __int__(self):
        value = 0
        for bit in self._bits:
            value <<= 1
            value += int(bit)
        return value

    def __len__(self):
        return len(self._bits)

    def __iter__(self):
        return iter(self._bits)

    def __getitem__(self, index):
        return self._bits[index]

    def __hash__(self):
        if self._hash is None:
            self._hash = len(self._bits) + (hash(int(self)) << 13)
        return self._hash

    def __eq__(self, other):
        # noinspection PyProtectedMember
        return isinstance(other, BitString) and numpy.array_equal(self._bits, other._bits)

    def __ne__(self, other):
        return not self == other

    def __and__(self, other):
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = numpy.bitwise_and(self._bits, other._bits)
        # noinspection PyUnresolvedReferences
        bits.flags.writeable = False
        return type(self)(bits)

    def __or__(self, other):
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = numpy.bitwise_or(self._bits, other._bits)
        # noinspection PyUnresolvedReferences
        bits.flags.writeable = False
        return type(self)(bits)

    def __xor__(self, other):
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = numpy.bitwise_xor(self._bits, other._bits)
        # noinspection PyUnresolvedReferences
        bits.flags.writeable = False
        return type(self)(bits)

    def __invert__(self):
        bits = numpy.bitwise_not(self._bits)
        # noinspection PyUnresolvedReferences
        bits.flags.writeable = False
        return type(self)(bits)

    def __add__(self, other):
        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)
        bits = numpy.concatenate((self._bits, other._bits))
        bits.flags.writeable = False
        return type(self)(bits)


