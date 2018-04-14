
from xcs.framework import ClassifierSet

class ClassifierSetRealConditions(ClassifierSet):

    def __init__(self, algorithm, possible_actions, bits_for_encoding: int):
        ClassifierSet.__init__(self, algorithm, possible_actions)
        self.bits_for_encoding = bits_for_encoding

    def match(self, situation):
        raise NotImplementedError("do it, Luis")


