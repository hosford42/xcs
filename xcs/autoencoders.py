__author__ = 'Aaron Hosford'

import math
import random

from xcs import ClassifierSet
from xcs import bitstrings


def logistic(x):
    try:
        return 1 / (1 + math.exp(-x))
    except (ValueError, ZeroDivisionError, OverflowError):
        return int(x > 0)


# This class can optionally automatically identify an appropriate encoding
# size and tends to generate somewhat sparse encodings. Typically, if there
# are N bits of entropy in the input space (i.e. a minimum of N bits is
# required to encode the input space without error) then this algorithm
# will use approximately 2 * N bits in the encoding.
class ExplanatoryClassifierSetAutoEncoder:

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
            1 if index in encoded else 0
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

                delta = bit - explanation[index]

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
                (bad and random.random() ** ((bad_count) / (len(bits))) <
                 math.log(len(bits), 2) / (len(self._ages)))):
            new_obj = max(self._encoder.possible_actions) + 1
            new_explanation = [
                .5 + (unexp if bit else -unexp) / 2
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
        while remaining_objects and max(unexplained) >= .1:
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
                key= consonance.get
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
            1 if index in encoded else 0
            for index in range(len(self._explanations))
        ])

    def decode(self, bits):
        assert isinstance(bits, bitstrings.BitString)
        encoded = [index for index, bit in enumerate(bits) if bit]

        weights = [0] * self._input_size

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
            return sum(bit or 0 for bit in correct) / (correct.count() or 1)


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
        input_distribution = "bitwise operators"

        def input_factory():
            bits = bitstrings.BitString.random(input_size // 4)
            bits2 = bitstrings.BitString(reversed(bits), input_size // 4)
            return bits + bits2 + (bits ^ bits2) + (bits & bits2)
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
    autoencoder = ExplanatoryClassifierSetAutoEncoder(
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
                if cycle % 1000 == 999 and isinstance(autoencoder, ExplanatoryClassifierSetAutoEncoder):
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
        for obj, explanation in sorted(autoencoder._explanations.items()):
            condition = bitstrings.BitCondition([
                1 if exp >= 2/3 else 0 if exp <= 2/3 else None
                for exp in explanation
            ])
            print('    ' + ''.join(str(round(exp, 3)).ljust(8) for exp in explanation) + '    ' + str(condition))
        print("Max population size:", exp_algorithm.max_population_size)
        print("Input distribution name:", input_distribution)
        print("Input size:", input_size)
        print("Encoding size:", len(autoencoder._explanations), '(' + ('auto' if encoded_size is None else 'manual') + ')')
        print(training_cycles, last_error + 1, average, recent)
