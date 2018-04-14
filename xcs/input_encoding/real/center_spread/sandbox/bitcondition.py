
import random
import abc

from xcs.input_encoding.real.center_spread.sandbox.bitstrings import BitString


class BitCondition(metaclass=abc.ABCMeta):
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

    def crossover_with(self, other, block_size, points):
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

        template = BitString.crossover_template(len(self), block_size, points)
        inv_template = ~template

        bits1 = (self._bits & template) | (other._bits & inv_template)
        mask1 = (self._mask & template) | (other._mask & inv_template)

        bits2 = (self._bits & inv_template) | (other._bits & template)
        mask2 = (self._mask & inv_template) | (other._mask & template)

        # Convert the modified sequences back into BitConditions
        return type(self)(bits1, mask1), type(self)(bits2, mask2)
