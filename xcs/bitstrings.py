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

from abc import ABCMeta, abstractmethod
import random


import xcs


def numpy_is_available():
    """Return a Boolean indicating whether numpy can be imported."""
    return xcs.numpy is not None


class BitStringBase(metaclass=ABCMeta):
    """Abstract base class for hashable, immutable sequences of bits (Boolean values).

    In addition to operations for indexing and iteration, provides standard bitwise operations, including & (bitwise
    and), | (bitwise or), ^ (bitwise xor), and ~ (bitwise not). Also implements the + operator, which acts like string
    concatenation.

    A bit string can also be cast as an integer or an ordinary string.
    """

    @classmethod
    @abstractmethod
    def random(cls, length, bit_prob=.5):
        """Create a bit string of the given length, with the probability of each bit being set equal to bit_prob, which
         defaults to .5."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def crossover_template(cls, length, points=2):
        """Create a crossover template with the given number of points. The crossover template can be used as a mask
        to crossover two bitstrings of the same length:

            assert len(parent1) == len(parent2)
            template = BitString.crossover_template(len(parent1))
            inv_template = ~template
            child1 = (parent1 & template) | (parent2 & inv_template)
            child2 = (parent1 & inv_template) | (parent2 & template)
        """
        raise NotImplementedError()

    def __init__(self, bits, hash_value):
        assert hash_value is None or isinstance(hash_value, int)

        self._bits = bits
        self._hash = hash_value

    @abstractmethod
    def any(self):
        """Returns True iff at least one bit is set."""
        raise NotImplementedError()

    @abstractmethod
    def count(self):
        """Returns the number of bits set to True in the bit string."""
        raise NotImplementedError()

    def __str__(self):
        # Overloads str(bitstring)
        return ''.join('1' if bit else '0' for bit in self)

    def __repr__(self):
        # Overloads repr(bitstring)
        return type(self).__name__ + '(' + str(self) + ')'

    @abstractmethod
    def __int__(self):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        # Overloads !=
        return not self == other

    @abstractmethod
    def __and__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __or__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __xor__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __invert__(self):
        raise NotImplementedError()

    @abstractmethod
    def __add__(self, other):
        raise NotImplementedError()


# There are two different implementations of BitString, one in _numpy_bitstrings and one in _python_bitstrings. The
# numpy version is dependent on numpy being installed, whereas the python one is written in pure Python with no external
# dependencies. By default, the python implementation is used. The user can override this behavior, if appropriate,
# by calling use_numpy().
from ._python_bitstrings import BitString
_using_numpy = False


def using_numpy():
    """Return a Boolean indicating whether the numpy implementation is currently in use."""
    return _using_numpy


def use_numpy():
    """Force the package to use the numpy-based BitString implementation. If numpy is not available, this will result
    in an ImportError. IMPORTANT: Bitstrings of different implementations cannot be mixed. Attempting to do so will
    result in undefined behavior."""
    global BitString, _using_numpy
    from ._numpy_bitstrings import BitString
    _using_numpy = True


def use_pure_python():
    """Force the package to use the pure Python BitString implementation. IMPORTANT: Bitstrings of different
    implementations cannot be mixed. Attempting to do so will result in undefined behavior."""
    global BitString, _using_numpy
    from ._python_bitstrings import BitString
    _using_numpy = False


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

        mask = BitString([random.random() > wildcard_probability for _ in range(len(bits))])
        return cls(bits, mask)

    def __init__(self, bits, mask=None):
        if mask is None:
            if isinstance(bits, str):
                bit_list = []
                mask = []
                for char in bits:
                    if char == '1':
                        bit_list.append(True)
                        mask.append(True)
                    elif char == '0':
                        bit_list.append(False)
                        mask.append(True)
                    elif char == '#':
                        bit_list.append(False)
                        mask.append(False)
                    else:
                        raise ValueError("Invalid character: " + repr(char))
                bits = BitString(bit_list)
                mask = BitString(mask)
                hash_value = None
            elif isinstance(bits, BitCondition):
                bits, mask, hash_value = bits._bits, bits._mask, bits._hash
            else:
                if not isinstance(bits, BitString):
                    bits = BitString(bits)
                mask = BitString.from_int(~0, len(bits))
                hash_value = None
        else:
            if not isinstance(bits, BitString):
                bits = BitString(bits)
            if not isinstance(mask, BitString):
                mask = BitString(mask)
            hash_value = None

        assert len(bits) == len(mask)

        self._bits = bits & mask
        self._mask = mask
        self._hash = hash_value

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
        if isinstance(index, slice):
            return BitCondition(self._bits[index], self._mask[index])
        return self._bits[index] if self._mask[index] else None

    def __hash__(self):
        # Overloads hash(condition)
        # If we haven't already calculated the hash value, do so now.
        if self._hash is None:
            self._hash = hash(tuple(self))
        return self._hash

    def __eq__(self, other):
        # Overloads ==
        if not isinstance(other, BitCondition):
            return False
        return len(self._bits) == len(other._bits) and self._bits == other._bits and self._mask == other._mask

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

        assert isinstance(other, (BitString, BitCondition))

        mismatches = self // other
        return not mismatches.any()

    def crossover_with(self, other, points=2):
        """Perform 2-point crossover on this bit condition and another of the same length, returning the two resulting
        children."""

        assert isinstance(other, BitCondition)
        assert len(self) == len(other)

        template = BitString.crossover_template(len(self), points)
        inv_template = ~template

        bits1 = (self._bits & template) | (other._bits & inv_template)
        mask1 = (self._mask & template) | (other._mask & inv_template)

        bits2 = (self._bits & inv_template) | (other._bits & template)
        mask2 = (self._mask & inv_template) | (other._mask & template)

        # Convert the modified sequences back into BitConditions
        return type(self)(bits1, mask1), type(self)(bits2, mask2)
