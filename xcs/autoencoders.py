__author__ = 'Aaron Hosford'

import math
import random

from xcs import ClassifierSet
from xcs import bitstrings


def logistic(x):
    """Computes the logistic function, e^x / (e^x + 1)."""
    try:
        return 1 / (1 + math.exp(-x))
    except (ValueError, ZeroDivisionError, OverflowError):
        return int(x > 0)


class BitStringAutoEncoder:
    """Learns a sparse encoding for the probability distribution on the
    input space, where inputs are arbitrary bit strings of equal length.

    This class can optionally automatically identify an appropriate
    encoding size, or use a predetermined encoding size. It tends to
    generate somewhat sparse encodings. Typically, if there are N bits of
    information in the input space (i.e. a minimum of N bits is required to
    encode the input space for the inputs to be reconstructed without
    error) then this algorithm will use approximately 2 * N bits in the
    encoding.

    Additionally, in the process of learning the encoding, this algorithm
    also formulates "explanations", which are sequences of weights which
    can be combined additively to produce the common bit sequences
    appearing in the inputs. Each explanation corresponds to a single bit
    in the encoding, having its weights counted in the totals whenever the
    associated encoding bit is turned on.

    Note: This class implements a novel algorithm by the author of the
    module, Aaron Hosford. Unlike the XCS algorithm upon which it is built,
    this algorithm has not yet been formally published in a peer reviewed
    journal as yet.
    """

    def __init__(self, encoder_algorithm, input_size, encoding_size=None):
        self._input_size = input_size
        self._encoding_size = encoding_size
        if encoding_size is None:
            self._encoder = ClassifierSet(encoder_algorithm, [0])
            self._explanations = {0: [.5] * input_size}
        else:
            self._encoder = ClassifierSet(
                encoder_algorithm,
                range(encoding_size)
            )
            self._explanations = {
                obj: [random.random() for _ in range(input_size)]
                for obj in range(encoding_size)
            }

        self._ages = {obj: 0 for obj in self._explanations}
        self._cycles = 0

    def train(self, bits):
        assert isinstance(bits, (bitstrings.BitString, bitstrings.BitCondition))
        assert len(bits) == self._input_size

        self._cycles += 1

        encoding_match = self._encoder.match(bits)

        remaining_objects = set(self._explanations)

        unexplained = [0 if bit is None else 1 for bit in bits]

        contributions = {}
        encoded = []
        utilities = []

        totals = [0] * len(bits)
        error = .5 * sum(bit is not None for bit in bits)

        while remaining_objects and max(unexplained) >= .1:
            consonance = {}
            for obj in remaining_objects:
                explanation = self._explanations[obj]

                degree = sum(
                    (1 - abs(bit - exp)) * unexp
                    for bit, exp, unexp in zip(bits,
                                               explanation,
                                               unexplained)
                    if bit is not None
                )

                new_totals = [
                    total + (exp - .5)
                    for total, exp in zip(totals, explanation)
                ]

                new_error = sum(
                    abs(bit - (1 if total > 0 else 0 if total < 0 else .5)) + abs(bit - (.5 + total))
                    for bit, total in zip(bits, new_totals)
                    if bit is not None
                )

                consonance[obj] = degree / len(bits) + (error - new_error) / len(bits)

            next_obj = max(
                remaining_objects,
                key=lambda obj: (
                    consonance[obj] * (1 + 1 / (1 + self._ages[obj]))
                )
            )

            remaining_objects.discard(next_obj)
            explanation = self._explanations[next_obj]

            new_totals = [
                total + (exp - .5)
                for total, exp in zip(totals, explanation)
            ]

            new_error = sum(
                abs(bit - (1 if total > 0 else 0 if total < 0 else .5)) + abs(bit - (.5 + total))
                for bit, total in zip(bits, new_totals)
                if bit is not None
            )

            totals = new_totals
            error = new_error

            encoded.append(next_obj)

            utilities.append(consonance[next_obj])

            contributions[next_obj] = [
                (1 - abs(exp - bit)) if bit is not None else 0
                for exp, bit in zip(explanation, bits)
            ]

            unexplained = [
                max(0, unexp - contr)
                for unexp, contr in zip(
                    unexplained,
                    contributions[next_obj]
                )
            ]

        decoded = self.decode(bitstrings.BitString([
            index in encoded
            for index in range(len(self._explanations))
        ]))

        bad = decoded.count() < len(decoded) or decoded.bits != bits or len(encoded) == len(self._explanations)
        if bad:
            bad_count = sum(bits[index] != decoded[index] for index in range(len(bits)))
        else:
            bad_count = 0

        for obj_index, obj in enumerate(encoded):
            self._ages[obj] += 1
            explanation = self._explanations[obj]
            for index, bit in enumerate(bits):
                if bit is None:
                    continue
                if len(self._explanations) == 1:
                    dist_from_others = 0
                else:
                    dist_from_others = sum(
                        abs(explanation[index] - other[index]) ** 2
                        for other in self._explanations.values()
                        if other is not explanation
                    ) / (len(self._explanations) - 1)

                nearest = (
                    abs(self._explanations[obj][index] - bit) ==
                    min(abs(self._explanations[other][index] - bit) for other in self._explanations)
                )

                weakest = (
                    abs(self._explanations[obj][index] - .5) ==
                    min(abs(self._explanations[other][index] - .5) for other in self._explanations)
                )

                strongest = (
                    abs(self._explanations[obj][index] - .5) ==
                    max(abs(self._explanations[other][index] - .5) for other in self._explanations)
                )

                delta = bit - explanation[index]

                # This gives terrific convergence rates, but reduces the quality of explanations
                if decoded[index] == bit and random.random() >= 1 / self._ages[obj]:
                    continue

                if dist_from_others < 1 / len(self._explanations):
                    explanation[index] += delta * random.uniform(.5, 1)
                else:
                    explanation[index] += (
                        delta /
                        self._ages[obj] *
                        (.5 + .5 * nearest) *
                        (1 - dist_from_others ** .5)
                    )

        if self._encoding_size is None and (
                not encoded or
                len(self._ages) <= 2 or
                (bad and
                 (random.random() ** ((bad_count) / (len(bits))) <
                  math.log(len(bits), 2) / (len(self._ages))))):
            new_obj = max(self._encoder.possible_actions) + 1
            new_explanation = [
                #bit if bit is not None else random.uniform(.45, .55)
                .5 + (unexp if bit else -unexp) / 2
                #max(min(.5 + (unexp if bit else -unexp), 1), 0)
                #bit if bit is not None and unexp >= .1 else .5#random.uniform(.45, .55)
                #.5 + .5 * (bit - .5) if bit is not None and unexp >= .1 else .5
                for bit, unexp in zip(bits, unexplained)
            ]
            self._explanations[new_obj] = new_explanation
            self._ages[new_obj] = 0

            encoded.append(new_obj)

            contributions[new_obj] = [
                (1 - abs(exp - bit)) if bit is not None else 0
                for exp, bit in zip(new_explanation, bits)
            ]

            self._encoder.add_possible_action(new_obj)
            self._encoder.algorithm.max_population_size *= (new_obj + 2) / (new_obj + 1)
            new_rule = self._encoder.algorithm.cover(encoding_match, new_obj)
            self._encoder.add(new_rule)

        if encoded:
            for best_obj in encoded[:-1]:
                if random.randrange(2):
                    break
            else:
                best_obj = encoded[-1]

            if best_obj in encoding_match:
                payoff = (
                    1 -
                    bad -
                    sum(
                        (bits[index] != decoded[index]) *
                        ((1 - contributions[best_obj][index]) *
                         contributions[encoded[-1]][index]) ** .5
                        for index in range(len(bits))
                    ) / len(bits)
                )

                encoding_match.selected_action = best_obj
                encoding_match.payoff = payoff
                encoding_match.apply_payoff()
            else:
                new_rule = self._encoder.algorithm.cover(encoding_match, best_obj)
                self._encoder.add(new_rule)

    def encode(self, bits):
        assert isinstance(bits, (bitstrings.BitString, bitstrings.BitCondition))
        assert len(bits) == self._input_size

        encoding_match = self._encoder.match(bits)

        remaining_objects = set(encoding_match)
        unexplained = [0 if bit is None else 1 for bit in bits]
        contributions = {}

        encoded = []
        utilities = []
        totals = [0] * len(bits)
        error = .5 * sum(bit is not None for bit in bits)

        decoded = self.decode(bitstrings.BitString(
            (index in encoded)
            for index in range(len(self._explanations))
        ))
        while remaining_objects and (decoded.count() < len(decoded) or decoded.bits != bits):
            consonance = {}
            for obj in remaining_objects:
                explanation = self._explanations[obj]

                new_totals = [
                    total + (exp - .5)
                    for total, exp in zip(totals, explanation)
                ]

                new_error = sum(
                    abs(bit - (1 if total > 0 else 0 if total < 0 else .5)) + abs(bit - (.5 + total))
                    for bit, total in zip(bits, new_totals)
                    if bit is not None
                )

                consonance[obj] = (
                    (error - new_error) +
                    encoding_match[obj].prediction
                ) / len(bits)

            next_obj = max(
                remaining_objects,
                key=consonance.get
            )

            remaining_objects.discard(next_obj)
            explanation = self._explanations[next_obj]

            new_totals = [
                total + (exp - .5)
                for total, exp in zip(totals, explanation)
            ]

            new_error = sum(
                abs(bit - logistic(total))
                for bit, total in zip(bits, new_totals)
                if bit is not None
            )

            if new_error >= error:
                continue

            totals = new_totals
            error = new_error

            encoded.append(next_obj)
            decoded = self.decode(bitstrings.BitString(
                (index in encoded)
                for index in range(len(self._explanations))
            ))

            utilities.append(consonance[next_obj])

            contributions[next_obj] = [
                (1 - abs(exp - bit)) if bit is not None else 0
                for exp, bit in zip(explanation, bits)
            ]

            unexplained = [
                max(0, unexp - contr)
                for unexp, contr in zip(
                    unexplained,
                    contributions[next_obj]
                )
            ]

        return bitstrings.BitString([
            index in encoded
            for index in range(len(self._explanations))
        ])

    def decode(self, bits):
        assert isinstance(bits, bitstrings.BitString)
        encoded = [index for index, bit in enumerate(bits) if bit]

        weights = [0] * self._input_size

        # TODO: Consider trying min/max here instead of sums. May interact
        #       usefully with mod on line 192 to produce good reasons very
        #       rapidly...
        for obj in encoded:
            explanation = self._explanations[obj]
            weights = [
                weight + (exp - .5)
                for weight, exp in zip(weights, explanation)
            ]

        return bitstrings.BitCondition(
            [weight > 0 for weight in weights],
            [weight != 0 for weight in weights]
        )

    def test(self, bits):
        assert isinstance(bits, (bitstrings.BitString, bitstrings.BitCondition))
        assert len(bits) == self._input_size

        reconstructed = self.decode(self.encode(bits))
        correct = bits ^ ~reconstructed
        if isinstance(bits, bitstrings.BitString):
            return correct.count() / len(correct)
        else:
            #return sum(bit or 0 for bit in correct) / (correct.count() or 1)
            result = sum(bit or 0 for bit in correct)
            if isinstance(correct, bitstrings.BitCondition):
                # Count bits that were wildcarded in the input as 50%.
                # If the algorithm can reconstruct these bits, the score
                # will then increase accordingly, but if it incorrectly
                # reconstructs them, the score will drop. Either way,
                # performance differences are being measured instead of
                # ignored.
                result += .5 * (len(reconstructed) - reconstructed.count())
            result /= len(correct)
            return result


if __name__ == "__main__":
    from xcs import XCSAlgorithm

    input_size = 12
    encoded_size = None#8
    training_cycles = 50000

    if False:
        input_distribution = "uniformly random"

        def input_factory():
            return bitstrings.BitString.random(input_size)
    elif False:
        assert input_size % 4 == 0
        input_distribution = "bitwise operators (1)"

        def input_factory():
            bits = bitstrings.BitString.random(input_size // 4)
            bits2 = bitstrings.BitString(reversed(bits), input_size // 4)
            return bits + bits2 + (bits ^ bits2) + (bits & bits2)
    elif False:
        assert input_size % 5 == 0
        input_distribution = "bitwise operators (2)"
        piece_length = input_size // 5

        def input_factory():
            bits = bitstrings.BitString.random(piece_length * 2)
            left = bits[:piece_length]
            right = bits[piece_length:]
            return bits + (left | right) + (left & right) + (left ^ right)
    elif False:
        input_distribution = "factorization-based probability"
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        assert 3 <= input_size <= primes[-1]
        limit = max(index for index, prime in enumerate(primes)
                    if prime <= input_size)
        weight_sets = []
        for prime in primes[:limit]:
            weights = [.9 if not (1 + index) % prime else .1
                       for index in range(input_size)]
            weight_sets.append(weights)

        def input_factory():
            selection = random.sample(
                weight_sets,
                random.randrange(1, limit)
            )
            avg_weights = [(.5 + sum(weights)) / (1 + len(selection))
                           for weights in zip(*selection)]
            bit_vals = [random.random() < weight
                        for weight in avg_weights]
            return bitstrings.BitString(bit_vals)
    else:
        input_distribution = "single bit on"

        def input_factory():
            return bitstrings.BitString(1 << random.randrange(input_size), input_size)

    print("Input distribution name:", input_distribution)

    exp_algorithm = XCSAlgorithm()
    exp_algorithm.max_population_size = (encoded_size or 1) * 20
    exp_algorithm.even_coverage = True
    exp_algorithm.learning_rate = .1
    exp_algorithm.mutation_probability = .25 / input_size
    exp_algorithm.wildcard_probability = 1 - 1 / input_size / 3

    # TODO: Try stacking autoencoders.

    # TODO: Determine whether bitmap autoencoders produced by this algorithm
    #       can be used to efficiently initialize neural networks for rapid
    #       training.

    # TODO: Can this algorithm be used for association rule discovery? Do
    #       the explanations represent meaningful groupings of bits? What
    #       about the classifier set rules' conditions?

    # TODO: Review the class and determine possible ways to reduce the
    #       discrepancy between the train() and encode() methods without
    #       reducing performance. This may in fact highlight ways to
    #       improve performance.

    # TODO: Rename this class; the name is way too long.
    autoencoder = BitStringAutoEncoder(
        exp_algorithm,
        input_size,
        encoded_size
    )

    average = 0
    recent = 0
    last_error = -1
    try:
        for cycle in range(training_cycles):
            bits = input_factory()
            score = autoencoder.test(bits)
            if score < 1:
                last_error = cycle
            autoencoder.train(bits)
            average += (score - average) / (cycle + 1)
            recent += (score - recent) / min(cycle + 1, 1000)
            if cycle % 100 == 99:
                print(cycle + 1, last_error + 1, average, recent)
                encoded = autoencoder.encode(bits)
                decoded = autoencoder.decode(encoded)
                wrong = decoded ^ bits
                wrong_count = sum(bit or 0 for bit in wrong)
                print(bits, '=>', encoded, '=>', decoded, '(' + str(wrong) + ',', str(wrong_count) + ')')
                print(len(autoencoder._encoder))
                if cycle % 1000 == 999 and isinstance(autoencoder, BitStringAutoEncoder):
                    for action in sorted(autoencoder._encoder.possible_actions):
                        explanation = bitstrings.BitCondition(
                            [exp > .5 for exp in autoencoder._explanations[action]],
                            [exp != .5 for exp in autoencoder._explanations[action]]
                        )
                        print(action, explanation)
                print()
    finally:
        print(autoencoder._encoder)
        print("Explanations:")
        for obj, explanation in sorted(autoencoder._explanations.items(), key=lambda obj_explanation: (sum(obj_explanation[1]), obj_explanation[0])):
            condition = bitstrings.BitCondition([
                1 if exp >= 2/3 else (0 if exp <= 1/3 else None)
                for exp in explanation
            ])
            strength = sum(2 * abs(exp - .5) for exp in explanation)
            print('    ' + str(obj).ljust(5) + ''.join(str(round(exp, 3)).ljust(8) for exp in explanation) + '    ' + str(condition) + '    ' + str(round(strength, 3)))
        print("Max population size:", exp_algorithm.max_population_size)
        print("Input distribution name:", input_distribution)
        print("Input size:", input_size)
        print("Encoding size:", len(autoencoder._explanations), '(' + ('auto' if encoded_size is None else 'manual') + ')')
        print(training_cycles, last_error + 1, average, recent)
