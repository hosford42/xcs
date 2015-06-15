# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# xcs
# ---
# Accuracy-based Classifier Systems for Python 3
#
# http://hosford42.github.io/xcs/
#
# (c) Aaron Hosford 2015, all rights reserved
# Revised (3 Clause) BSD License
#
# Implements the XCS (Accuracy-based Classifier System) algorithm,
# as described in the 2001 paper, "An Algorithmic Description of XCS,"
# by Martin Butz and Stewart Wilson.
#
# -------------------------------------------------------------------------

"""
Accuracy-based Classifier Systems for Python 3

This xcs submodule provides bit-string and bit-condition data types used by
the XCS algorithm.

Pure Python versus numpy:
    This submodule has two alternate implementations for the BitString
    class. One is based on Python ints, has no external dependencies, and
    is used by default. The other is based on numpy arrays, requires numpy
    to be installed, and can be activated by calling use_numpy(). If you
    change your mind, you can always switch back to the pure Python
    implementation by calling use_pure_python(). If you're not sure which
    one you're using, you can tell by calling using_numpy(). Before you
    call use_numpy(), it is recommended that you verify that numpy is
    available by calling numpy_is_available() to avoid an import error.
    While it is safe to switch back and forth between implementations as
    many times as you like, you must not mix BitString or BitCondition
    instances from one implementation with those of the other; to do so may
    lead to undefined behavior.

    It is worth noting that the Python int-based and numpy array-based
    implementations have (somewhat surprisingly) roughly comparable speeds.
    In fact, on some systems, the Python-based implementation is visibly
    faster. If you are concerned with speed, it is best to actually test
    the two implementations on your system to see which is faster. If not,
    the pure Python implementation, enabled by default, is recommended.




Copyright (c) 2015, Aaron Hosford
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of xcs nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

__author__ = 'Aaron Hosford'

__all__ = [
    # Classes
    'BitCondition',
    'BitString',

    # Functions
    'numpy_is_available',
    'use_numpy',
    'use_pure_python',
    'using_numpy',
]

from abc import ABCMeta, abstractmethod
import random

import xcs


def numpy_is_available():
    """Return a Boolean indicating whether numpy can be imported.

    Usage:
        if numpy_is_available():
            import numpy

    Arguments: None
    Return:
        A bool indicating whether numpy can be imported.
    """
    return xcs.numpy is not None


# IMPORTANT:
#   This class must appear *before* the BitString class is imported. It is
#   a rather ugly solution, but _numpy_bitstrings.py and
#   _python_bitstrings.py import this module, and then this module imports
#   one of them. This lets us switch back and forth as needed.
class BitStringBase(metaclass=ABCMeta):
    """Abstract base class for hashable, immutable sequences of bits
    (Boolean values). There are two separate implementations of the
    BitString class, each of which inherits from this base class. One is
    implemented in pure Python (using Python ints), and the other is
    implemented using numpy arrays. Inheriting from this abstract base
    class serves to ensure that both implementations provide the same
    interface.

    Usage:
        This is an abstract base class. Use the BitString subclass to
        create an instance.

    Init Arguments:
        bits: The object the implementation uses to represent the bits of
            the BitString.
        hash_value: None, indicating the hash value will be computed later,
            or an int representing the hash value of the BitString.
    """

    @classmethod
    @abstractmethod
    def random(cls, length, bit_prob=.5):
        """Create a bit string of the given length, with the probability of
        each bit being set equal to bit_prob, which defaults to .5.

        Usage:
            # Create a random BitString of length 10 with mostly zeros.
            bits = BitString.random(10, bit_prob=.1)

        Arguments:
            length: An int, indicating the desired length of the result.
            bit_prob: A float in the range [0, 1]. This is the probability
                of any given bit in the result having a value of 1; default
                is .5, giving 0 and 1 equal probabilities of appearance for
                each bit's value.
        Return:
            A randomly generated BitString instance of the requested
            length.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def crossover_template(cls, length, points=2):
        """Create a crossover template with the given number of points. The
        crossover template can be used as a mask to crossover two
        bitstrings of the same length.

        Usage:
            assert len(parent1) == len(parent2)
            template = BitString.crossover_template(len(parent1))
            inv_template = ~template
            child1 = (parent1 & template) | (parent2 & inv_template)
            child2 = (parent1 & inv_template) | (parent2 & template)

        Arguments:
            length: An int, indicating the desired length of the result.
            points: An int, the number of crossover points.
        Return:
            A BitString instance of the requested length which can be used
            as a crossover template.
        """
        raise NotImplementedError()

    def __init__(self, bits, hash_value):
        assert hash_value is None or isinstance(hash_value, int)

        self._bits = bits
        self._hash = hash_value

    @abstractmethod
    def any(self):
        """Returns True iff at least one bit is set.

        Usage:
            assert not BitString('0000').any()
            assert BitString('0010').any()

        Arguments: None
        Return:
            A bool indicating whether at least one bit has value 1.
        """
        raise NotImplementedError()

    @abstractmethod
    def count(self):
        """Returns the number of bits set to True in the bit string.

        Usage:
            assert BitString('00110').count() == 2

        Arguments: None
        Return:
            An int, the number of bits with value 1.
        """
        raise NotImplementedError()

    def __str__(self):
        """Overloads str(bitstring)"""
        return ''.join('1' if bit else '0' for bit in self)

    def __repr__(self):
        """Overloads repr(bitstring)"""
        return type(self).__name__ + '(' + repr(str(self)) + ')'

    @abstractmethod
    def __int__(self):
        """Overloads int(instance)"""
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        """Overloads len(instance)"""
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        """Overloads iter(instance)"""
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self):
        """Overloads hash(instance)"""
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other):
        """Overloads instance1 == instance2"""
        raise NotImplementedError()

    def __ne__(self, other):
        """Overloads !="""
        return not self == other

    @abstractmethod
    def __and__(self, other):
        """Overloads instance1 & instance2"""
        raise NotImplementedError()

    @abstractmethod
    def __or__(self, other):
        """Overloads instance1 | instance2"""
        raise NotImplementedError()

    @abstractmethod
    def __xor__(self, other):
        """Overloads instance1 ^ instance2"""
        raise NotImplementedError()

    @abstractmethod
    def __invert__(self):
        """Overloads ~instance"""
        raise NotImplementedError()

    @abstractmethod
    def __add__(self, other):
        """Overloads instance1 + instance2"""
        raise NotImplementedError()


# There are two different implementations of BitString, one in
# _numpy_bitstrings and one in _python_bitstrings. The numpy version is
# dependent on numpy being installed, whereas the python one is written in
# pure Python with no external dependencies. By default, the python
# implementation is used, since it is of comparable speed and has no
# external dependencies. The user can override this behavior, if desired,
# by calling use_numpy().
from ._python_bitstrings import BitString
_using_numpy = False


def using_numpy():
    """Return a Boolean indicating whether the numpy implementation is
    currently in use.

    Usage:
        if using_numpy():
            use_pure_python()

    Arguments: None
    Return:
        A bool indicating whether the numpy implementation is currently in
        use, as opposed to the pure Python implementation.
    """
    return _using_numpy


def use_numpy():
    """Force the package to use the numpy-based BitString implementation.
    If numpy is not available, this will result in an ImportError.
    IMPORTANT: Bitstrings of different implementations cannot be mixed.
    Attempting to do so will result in undefined behavior.

    Usage:
        use_numpy()
        assert using_numpy()

    Arguments: None
    Return: None
    """
    global BitString, _using_numpy
    from ._numpy_bitstrings import BitString
    _using_numpy = True


def use_pure_python():
    """Force the package to use the pure Python BitString implementation.
    IMPORTANT: Bitstrings of different implementations cannot be mixed.
    Attempting to do so will result in undefined behavior.

    Usage:
        use_pure_python()
        assert not using_numpy()

    Arguments: None
    Return: None
    """
    global BitString, _using_numpy
    from ._python_bitstrings import BitString
    _using_numpy = False


class BitCondition:
    """A pair of bit strings, one indicating the bit values, and the other
    indicating the bit mask, which together act as a matching template for
    bit strings. Like bit strings, bit conditions are hashable and
    immutable. Think of BitConditions as patterns which can match against
    BitStrings of the same length. At each index, we can have a 1, a 0, or
    a # (wildcard). If the value is 1 or 0, the BitString must have the
    same value at that index. If the value is #, the BitString can have any
    value at that index.

    BitConditions are matched against BitStrings in one of two ways:
        Method 1:
            result = condition // bitstring
            # result now contains a new BitString which contains a 1 for
            # each position that violated the pattern, and a 0 for each
            # position that did not. This tells us exactly where the
            # condition and the bitstring disagree
        Method 2:
            result = condition(bitstring)
            # result now contains a single Boolean value which is True if
            # the bitstring fully satisfies the pattern specified by the
            # condition, or False if the bitstring disagrees with the
            # condition at at least one index

    BitConditions can also match against other BitConditions in the same
    way that they are matched against BitStrings, with the sole exception
    that if the condition being used as the pattern specifies a 1 or 0 at a
    particular index, and the condition being used as the substrate
    contains an # at that point, the match fails. This means that if
    you have two conditions, condition1 and condition2, where condition1
    matches a bitstring and condition2 matches condition1, then condition2
    is guaranteed to match the bitstring, as well.

    Usage:
        # A few ways to create a BitCondition instance
        condition1 = BitCondition('001###01#1')
        condition2 = BitCondition(BitString('0010010111'),
                                  BitString('1110001101'))
        assert condition1 == condition2
        condition3 = BitCondition.cover('0010010111', .25)
        assert condition3(BitString('0010010111'))  # It matches

        # They print up nicely
        assert str(condition1) == '001###01#1'
        print(condition1)  # Prints: 001###01#1
        print(repr(condition1))  # Prints: BitCondition('001###01#1')

        # Indexing is from left to right, like an ordinary string.
        # (Wildcards are represented as the value None at the given index.)
        assert condition1[0] == 0
        assert condition1[-1] == 1
        assert condition1[4] is None

        # They are immutable
        condition1[3] = 0  # This will raise a TypeError

        # Slicing works
        assert condition1[3:-3] == BitCondition('###0')

        # You can iterate over them
        for bit in condition1:
            if bit is None:
                print("Found a wildcard!)

        # Unlike bitstrings, they cannot be cast as ints
        as_int = int(condition1)  # This will raise a TypeError

        # They can be used in hash-based containers
        s = {condition1, condition3}
        d = {condition1: "a", condition3: "b"}

        # Unlike bitstrings, they do not support the any() method
        condition1.any()  # This will raise an AttributeError

        # Unlike bitstrings, BitCondition.count() returns the number of
        # bits that are not wildcards, rather than the number of bits that
        # have a value of 1.
        assert condition1.count() == condition1.mask.count() == 6

        # The bitwise operators for BitConditions work differently from
        # those of BitStrings; provided the bits of each condition are
        # compatible, i.e. there is no point where their bits disagree
        # and neither of them is a wildcard, then &, |, and ~ actually
        # represent set operations over the BitStrings that the conditions
        # will match.
        assert condition1 & condition1 == condition1
        assert condition1 | condition1 == condition1
        assert (condition1 | ~condition1)(BitString.random(10))
        assert condition1(condition1 & condition3)  # They are compatible
        assert condition3(condition1 & condition3)  # They are compatible
        assert (condition1 | condition3)(condition1)  # They are compatible
        assert (condition1 | condition3)(condition3)  # They are compatible

        # BitConditions can also be concatenated together like strings
        concatenation = condition1 + condition3
        assert len(concatenation) == 10 * 2

        # They support the Genetic Algorithm's crossover operator directly
        child1, child2 = condition1.crossover_with(condition3)

    Init Arguments:
        bits: If mask is provided, a sequence from which the bits of the
            condition can be determined. If mask is omitted, a sequence
            from which the bits and mask of the condition can be
            determined.
        mask: None, or a sequence from which the mask can be determined,
            having the same length as the sequence provided for bits.
    """

    @classmethod
    def cover(cls, bits, wildcard_probability):
        """Create a new bit condition that matches the provided bit string,
        with the indicated per-index wildcard probability.

        Usage:
            condition = BitCondition.cover(bitstring, .33)
            assert condition(bitstring)

        Arguments:
            bits: A BitString which the resulting condition must match.
            wildcard_probability: A float in the range [0, 1] which
            indicates the likelihood of any given bit position containing
            a wildcard.
        Return:
            A randomly generated BitCondition which matches the given bits.
        """

        if not isinstance(bits, BitString):
            bits = BitString(bits)

        mask = BitString([
            random.random() > wildcard_probability
            for _ in range(len(bits))
        ])

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
                        raise ValueError("Invalid character: " +
                                         repr(char))
                bits = BitString(bit_list)
                mask = BitString(mask)
                hash_value = None
            elif isinstance(bits, BitCondition):
                bits, mask, hash_value = bits._bits, bits._mask, bits._hash
            else:
                if not isinstance(bits, BitString):
                    bits = BitString(bits)
                mask = BitString(~0, len(bits))
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
        """The bit string indicating the bit values of this bit condition.
        Indices that are wildcarded will have a value of False."""
        return self._bits

    @property
    def mask(self):
        """The bit string indicating the bit mask. A value of 1 for a
        bit indicates it must match the value bit string. A value of 0
        indicates it is masked/wildcarded."""
        return self._mask

    def count(self):
        """Return the number of bits that are not wildcards.

        Usage:
            non_wildcard_count = condition.count()

        Arguments: None
        Return:
            An int, the number of positions in the BitCondition which are
            not wildcards.
        """
        return self._mask.count()

    def __str__(self):
        """Overloads str(condition)"""
        return ''.join(
            '1' if bit else ('#' if bit is None else '0')
            for bit in self
        )

    def __repr__(self):
        """Overloads repr(condition)"""
        return type(self).__name__ + '(' + repr(str(self)) + ')'

    def __len__(self):
        """Overloads len(condition)"""
        return len(self._bits)

    def __iter__(self):
        """Overloads iter(condition), and also, for bit in condition. The
        values yielded by the iterator are True (1), False (0), or
        None (#)."""
        for bit, mask in zip(self._bits, self._mask):
            yield bit if mask else None

    def __getitem__(self, index):
        """Overloads condition[index]. The values yielded by the index
        operator are True (1), False (0), or None (#)."""
        if isinstance(index, slice):
            return BitCondition(self._bits[index], self._mask[index])
        return self._bits[index] if self._mask[index] else None

    def __hash__(self):
        """Overloads hash(condition)."""
        # If we haven't already calculated the hash value, do so now.
        if self._hash is None:
            self._hash = hash(tuple(self))
        return self._hash

    def __eq__(self, other):
        """Overloads =="""
        if not isinstance(other, BitCondition):
            return False
        return (
            len(self._bits) == len(other._bits) and
            self._bits == other._bits and
            self._mask == other._mask
        )

    def __ne__(self, other):
        """Overloads !="""
        return not self == other

    def __and__(self, other):
        """Overloads &"""
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)(
            (self._bits | ~self._mask) & (other._bits | ~other._mask),
            self._mask | other._mask
        )

    def __or__(self, other):
        """Overloads |"""
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)(
            self._bits | other._bits,
            self._mask & other._mask & ~(self._bits ^ other._bits)
        )

    def __invert__(self):
        """Overloads unary ~"""
        return type(self)(~self._bits, self._mask)

    def __add__(self, other):
        """Overloads +"""
        if not isinstance(other, BitCondition):
            return NotImplemented
        return type(self)(
            self._bits + other._bits,
            self._mask + other._mask
        )

    def __floordiv__(self, other):
        """Overloads the // operator, which we use to find the indices in
        the other value that do/can disagree with this condition."""
        if isinstance(other, BitCondition):
            return ((self._bits ^ other._bits) | ~other._mask) & self._mask

        if isinstance(other, int):
            other = BitString.from_int(other, len(self._bits))
        elif not isinstance(other, BitString):
            other = BitString(other)

        return (self._bits ^ other) & self._mask

    def __call__(self, other):
        """Overloads condition(bitstring). Returns a Boolean value that
        indicates whether the other value satisfies this condition."""

        assert isinstance(other, (BitString, BitCondition))

        mismatches = self // other
        return not mismatches.any()

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
