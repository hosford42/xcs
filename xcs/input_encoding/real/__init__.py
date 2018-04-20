import abc

from xcs.bitstrings import BitString


class EncoderDecoder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def encode_as_int(self, d: float) -> int:
        """Does encoding of a float into one of 'm' ints, where m = 2^k (k=number of bits used for representation)"""
        raise NotImplementedError()

    @abc.abstractmethod
    def encode(self, d: float) -> BitString:
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, b: BitString) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def mutate_float(self, d: float, factor: float) -> float:
        """Mutates a bitstring encoded with this specific encoder, by a certain factor (in [0,1])"""
        raise NotImplementedError()

    @abc.abstractmethod
    def mutate(self, b: BitString, factor: float) -> BitString:
        """Mutates a bitstring encoded with this specific encoder, by a certain factor (in [0,1])"""
        raise NotImplementedError()
