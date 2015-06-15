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

This xcs submodule provides a version of the BitString class, implemented
using ordinary Python ints. Speed is comparable to numpy arrays.

Do not access the contents of this module directly. See the documentation
for xcs.bitstrings for a detailed explanation of how to properly access
the appropriate BitString implementation.




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
    'BitString',
]

import random


from .bitstrings import BitStringBase


class BitString(BitStringBase):
    """A hashable, immutable sequence of bits (Boolean values). Implemented
    in pure Python, with no external dependencies. Speed is comparable to
    numpy-based implementation.

    Usage:
        # A few ways to create a BitString instance
        bitstring1 = BitString('0010010111')
        bitstring2 = BitString(134, 10)
        bitstring3 = BitString([0] * 10)
        bitstring4 = BitString.random(10)

        # They print up nicely
        assert str(bitstring1) == '0010010111'
        print(bitstring1)  # Prints: 0010010111
        print(repr(bitstring1))  # Prints: BitString('0010010111')

        # Indexing is from left to right, like an ordinary string
        assert bitstring1[0] == 0
        assert bitstring1[-1] == 1

        # They are immutable
        bitstring1[3] = 0  # This will raise a TypeError

        # Slicing works
        assert bitstring1[3:-3] == BitString('0010')

        # You can iterate over them
        for bit in bitstring1:
            if bit == 1:
                print("Found one!)

        # They can be cast as ints
        assert int(bitstring2) == 134

        # They can be used in hash-based containers
        s = {bitstring1, bitstring2, bitstring3}
        d = {bitstring1: "a", bitstring2: "b", bitstring3: "c"}

        # BitString.any() is True whenever there is at least 1 non-zero bit
        assert bitstring1.any()
        assert not bitstring3.any()

        # BitString.count() returns the number of non-zero bits
        assert bitstring1.count() == 5
        assert bitstring3.count() == 0

        # BitStrings can be treated like integer masks
        intersection = bitstring1 & bitstring2
        union = bitstring2 | bitstring3
        complement = ~bitstring1

        # And they can also be concatenated together like strings
        concatenation = bitstring3 + bitstring4
        assert len(concatenation) == 10 * 2

        # BitString.crossover_template() is a special class method for
        # creating BitString instances that can be used for N-point
        # crossover operators, e.g. like those used in Genetic Algorithms.
        template = BitString.crossover_template(10)
        child = (bitstring1 & template) | (bitstring3 & ~template)


    Init Arguments:
        bits: An int or a sequence of bools which is used to determine the
            values of the bits in the BitString.
        length: An int indicating the expected length of the BitString, or
            None. Default is None, which causes the length of bits to be
            used if it is a sequence, or bits.bit_length() if bits is an
            int.

    NOTE: If the bits argument is an int, length must be None or an int
          of length >= bits.bit_length(). If bits is a sequence of bools,
          then the length of the sequence must exactly equal length. If
          these length constraints are not met, a ValueError is raised.
    """

    @classmethod
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

        assert isinstance(length, int) and length >= 0
        assert isinstance(bit_prob, (int, float)) and 0 <= bit_prob <= 1

        bits = 0
        for _ in range(length):
            bits <<= 1
            bits += (random.random() < bit_prob)

        return cls(bits, length)

    @classmethod
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

        assert isinstance(length, int) and length >= 0
        assert isinstance(points, int) and points >= 0

        # Select the crossover points.
        points = random.sample(range(length + 1), points)

        # Prep the points for the loop.
        points.sort()
        points.append(length)

        # Fill the bits in with alternating ranges of 0 and 1 according to
        # the selected crossover points.
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
        """Returns True iff at least one bit is set.

        Usage:
            assert not BitString('0000').any()
            assert BitString('0010').any()

        Arguments: None
        Return:
            A bool indicating whether at least one bit has value 1.
        """
        return bool(self._bits)

    def count(self):
        """Returns the number of bits set to True in the bit string.

        Usage:
            assert BitString('00110').count() == 2

        Arguments: None
        Return:
            An int, the number of bits with value 1.
        """
        result = 0
        bits = self._bits
        while bits:
            result += bits % 2
            bits >>= 1
        return result

    def __int__(self):
        """Overloads int(bitstring)"""
        return self._bits

    def __len__(self):
        """Overloads len(instance)"""
        return self._length

    def __iter__(self):
        """Overloads iter(bitstring), and also, for bit in bitstring"""
        for index in range(self._length - 1, -1, -1):
            yield (self._bits >> index) % 2

    def __getitem__(self, index):
        """Overloads bitstring[index]"""
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            if start < 0:
                return BitString(0, 0)
            if step == -1:
                step = 1
                start, stop = stop, start
            if step == 1:
                length = stop - start
                bits = ((self._bits >> (self._length - stop)) %
                        (1 << length))
                return BitString(bits, length)
            else:
                return BitString([
                    self[point]
                    for point in range(start, stop, step)
                ])

        assert isinstance(index, int)

        if index >= 0:
            assert index < self._length
            return (self._bits >> (self._length - index - 1)) % 2
        else:
            assert -index <= self._length
            return (self._bits >> (-index - 1)) % 2

    def __hash__(self):
        """Overloads hash(bitstring)"""
        if self._hash is None:
            self._hash = hash(self._bits) ^ hash(self._length)
        return self._hash

    def __eq__(self, other):
        """Overloads =="""
        if not isinstance(other, BitString):
            return NotImplemented
        return self._bits == other._bits and self._length == other._length

    def __and__(self, other):
        """Overloads &"""
        if isinstance(other, BitString):
            assert self._length == other._length
        else:
            other = BitString(other, self._length)
        return BitString(self._bits & other._bits, self._length)

    def __or__(self, other):
        """Overloads |"""
        if isinstance(other, BitString):
            assert self._length == other._length
        else:
            other = BitString(other, self._length)
        return BitString(self._bits | other._bits, self._length)

    def __xor__(self, other):
        """Overloads ^"""
        if isinstance(other, BitString):
            assert self._length == other._length
        else:
            other = BitString(other, self._length)
        return BitString(self._bits ^ other._bits, self._length)

    def __invert__(self):
        """Overloads unary ~"""
        return BitString(~self._bits % (1 << self._length), self._length)

    def __add__(self, other):
        """Overloads +"""
        if not isinstance(other, BitString):
            other = BitString(other)
        return BitString(
            (self._bits << other._length) + other._bits,
            self._length + other._length
        )
