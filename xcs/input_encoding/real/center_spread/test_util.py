import unittest
import math

from random import random, sample, choice
from xcs.input_encoding.real.center_spread.util import EncoderDecoder


class UnitTestUtil(unittest.TestCase):
    def setUp(self):
        pass

    def test_encoding_decoding(self):
        for encoding_bits in [2, 4, 5, 8, 10]:
            print("******************* [encoding bits = %d]" % (encoding_bits))
            m = -10
            M = 20
            enc_dec = EncoderDecoder(min_value=m, max_value=M, encoding_bits=encoding_bits)
            # all_ds = [m + random() * (M - m) for _ in range(10)]
            all_ds = list(set([m, M] + [m + random() * (M - m) for _ in range(10)]))  # sample(list(range(m, M)), min(10, M - m))))
            all_ds.sort()
            for d in all_ds:
                r = enc_dec.encode_as_int(d=d)
                print("%.2f -> %d" % (d, r))
                self.assertTrue((r >= 0) and (r <= (math.pow(2, encoding_bits) - 1)))
                self.assertLessEqual(r.bit_length(), encoding_bits)
                a_bitstring = enc_dec.encode(d)
                self.assertEqual(enc_dec.decode(a_bitstring), r)

    def test_encoding_decoding_mutation(self):
        encoding_bits = 4
        m = -10
        M = 20
        enc_dec = EncoderDecoder(min_value=m, max_value=M, encoding_bits=encoding_bits)
        for _ in range(10):
            factor = random()
            # print("With factor = %.2f" % (factor))
            max_spread = (M - m) * factor
            for _ in range(10):
                d = random() * (M - m) + m
                r = enc_dec.mutate_float(d, factor=factor)
                # print("%.2f -> %d" % (d, r))
                self.assertTrue((r >= m) and (r <= M))
                current_spread = abs(r-d)
                self.assertTrue(
                    current_spread <= max_spread,
                    "Spread ~ %.2f, but max spread = (%.2f - %.2f)*%.2f ~ %.2f" % (current_spread, M, m, factor, max_spread))
