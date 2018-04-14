
from typing import List, Tuple
from random import sample

from xcs.bitstrings import BitConditionBase, BitString as CoreBitString
from xcs.input_encoding.real.center_spread.util import EncoderDecoder


class BitConditionRealEncoding(BitConditionBase):
    """See Documentation of base class."""

    # @classmethod
    # def cover(cls, situation, wildcard_probability):
    #     """Create a new bit condition that matches the provided bit string,
    #     with the indicated per-index wildcard probability.
    #
    #     Usage:
    #         condition = BitCondition.cover(bitstring, .33)
    #         assert condition(bitstring)
    #
    #     Arguments:
    #         bits: A BitString which the resulting condition must match.
    #         wildcard_probability: A float in the range [0, 1] which
    #         indicates the likelihood of any given bit position containing
    #         a wildcard.
    #     Return:
    #         A randomly generated BitCondition which matches the given bits.
    #     """
    #
    #     if not isinstance(bits, BitString):
    #         bits = BitString(bits)
    #
    #     mask = BitString([
    #         random.random() > wildcard_probability
    #         for _ in range(len(bits))
    #     ])
    #
    #     return cls(bits, mask)

    def _encode(self, center_spreads: List[Tuple[float, float]]) -> CoreBitString:
        result = CoreBitString('')
        for (center, spread) in center_spreads:
            result += self.real_translator.encode(center)
            result += self.real_translator.encode(spread)
        return result

    def __init__(self, encoder: EncoderDecoder, center_spreads: List[Tuple[float, float]], mutation_strength: float):
        assert len(center_spreads) > 0
        assert (mutation_strength > 0) and (mutation_strength < 1)
        self.real_translator = encoder
        self.mutation_strength = mutation_strength
        BitConditionBase.__init__(self, bits=self._encode(center_spreads), mask=None)
        self.center_spreads = center_spreads

    def __str__(self):
        """Overloads str(condition)"""
        return ','.join(["(%d, %d)" % (center, spread) for (center, spread) in self])

    def __len__(self):
        """Overloads len(condition)"""
        return len(self.center_spreads)

    def __getitem__(self, index):
        """Overloads condition[index]. The values yielded by the index
        operator are True (1), False (0), or None (#)."""
        if isinstance(index, slice):
            return BitConditionRealEncoding(self.real_translator, self.center_spreads[index], self.mutation_strength)
            # return BitCondition(self._bits[index], self._mask[index])
        # return self._bits[index] if self._mask[index] else None
        return self.center_spreads[index]

    def __call__(self, situation):
        """Overloads condition(situation). Returns a Boolean value that
        indicates whether the other value satisfies this condition."""

        assert isinstance(situation, BitString)
        assert len(self) == len(situation)

        center_spreads = [(center, spread) for (center, spread) in self]
        values = [value for value in situation]
        return all(
            [((value >= center - spread) and (value <= center + spread))
             for ((center, spread), value) in zip(center_spreads, values)])

    def __iter__(self):
        """Overloads iter(bitstring), and also, for bit in bitstring"""
        for (center, spread) in self.center_spreads:
            yield (center, spread)
        # for interval in range(1, int(len(self.bits) / (2 * self.real_translator.encoding_bits)) + 1):
        #     center_start = (interval - 1) * (2 * self.real_translator.encoding_bits)
        #     spread_start = center_start + self.real_translator.encoding_bits
        #     center = self.real_translator.decode(self.bits[center_start: spread_start])
        #     spread = self.real_translator.decode(self.bits[spread_start: spread_start + self.real_translator.encoding_bits])
        #     yield (center, spread)

    def mutate(self, situation):
        center_spreads = [(center, spread) for (center, spread) in self]  # TODO: actually mutate values in a way that is still matches 'situation'
        print("TODO: actually mutate values in a way that is still matches 'situation'")
        return BitConditionRealEncoding(
            encoder=self.real_translator,
            center_spreads=center_spreads,
            mutation_strength=self.mutation_strength)

    def crossover_with(self, other, block_size, points):
        # TODO: get rid of parameter 'block_size'
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
        assert isinstance(other, BitConditionRealEncoding)
        assert len(self) == len(other)
        assert points < len(self)

        if self == other:
            # nothing to do
            print(" =====> ARE THE SAME????????????????????????")  # TODO: take this out.
            return self, other
        else:
            pts = [-1] + sample(range(len(self) - 2), points) + [len(self) - 1]
            pts.sort()
            genome_1 = self
            genome_2 = other
            # active = self
            result = ([], [])
            for begin, end in zip(map(lambda x: x + 1, pts[:-1]), map(lambda x: x + 1, pts[1:])):
                result = (result[0] + genome_1.center_spreads[begin: end], result[1] + genome_2.center_spreads[begin: end])
                # result += [(genome_1[begin: end], genome_2[begin: end])]
                genome_1, genome_2 = (self, other) if genome_1 == other else (other, self)
                # active = self if active == other else other
            return (BitConditionRealEncoding(self.real_translator, result[0], self.mutation_strength), BitConditionRealEncoding(self.real_translator, result[1], self.mutation_strength))
            raise NotImplementedError()




        template = BitString.crossover_template(len(self), block_size, points)
        inv_template = ~template

        bits1 = (self._bits & template) | (other._bits & inv_template)
        mask1 = (self._mask & template) | (other._mask & inv_template)

        bits2 = (self._bits & inv_template) | (other._bits & template)
        mask2 = (self._mask & inv_template) | (other._mask & template)

        # Convert the modified sequences back into BitConditions
        return type(self)(bits1, mask1), type(self)(bits2, mask2)




class BitString(CoreBitString):

    def __init__(self, encoder: EncoderDecoder, reals: List[float]):
        assert len(reals) > 0
        self.as_reals = reals
        as_bitstring = CoreBitString('')
        for a_real in reals:
            as_bitstring += encoder.encode(a_real)
        CoreBitString.__init__(self, bits=as_bitstring)
        self.real_translator = encoder

    def __len__(self):
        """Overloads len(instance)"""
        return len(self.as_reals)

    def __iter__(self):
        """Overloads iter(bitstring), and also, for bit in bitstring"""
        for real in self.as_reals:
            yield real
        # for interval in range(1, int(len(self) / self.real_translator.encoding_bits) + 1):
        #     value_start = (interval - 1) * self.real_translator.encoding_bits
        #     value = self.real_translator.decode(self[value_start : value_start + self.real_translator.encoding_bits])
        #     yield value

    def __str__(self):
        """Overloads str(bitstring)"""
        return ','.join([str(value) for value in self])

    def cover(self):
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
        from random import random
        center_spread_list = []
        for value in self:
            center = random() * (self.real_translator.extremes[1] - self.real_translator.extremes[0]) + self.real_translator.extremes[0]
            center = self.real_translator.encode_as_int(center)
            spread = abs(center - value)  # min value that makes 'value' match
            center_spread_list.append((center, spread))
        result = BitConditionRealEncoding(encoder=self.real_translator, center_spreads=center_spread_list, mutation_strength=0.1)  # TODO: value of mutation strenght!!!!!
        assert result(self)
        return result
        # from xcs.bitstrings import BitCondition  # TODO: take this out of here!
        #
        # bits = self._bits
        # if not isinstance(bits, BitString):
        #     bits = BitString(bits, len(self))
        #
        # mask = BitString([
        #     random.random() > self.cover_wildcard_probability
        #     for _ in range(len(bits))
        # ])
        #
        # return BitCondition(bits, mask)

