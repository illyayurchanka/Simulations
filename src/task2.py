import numpy as np
from task1 import Automata
import itertools
from collections import Counter

class Enthropy_Gibbs_Automata(Automata):
    def __init__(self, width: int = 12, height: int = 100, rule_num: int = 30, reversable: bool = False, central: bool = True, initial_state=None):

        self.central = central
        self.rev = reversable

        assert width > 0 and height > 0, "Width and Height should be greater than 0"
        self.width = width
        self.height = height

        if initial_state is not None:
            assert isinstance(initial_state, list) and len(initial_state) == 8, "Initial state should be list with 8 elements"
            self.init_cond = initial_state
        else:
            self.init_cond = self._initial_condition()

        assert rule_num < 256, "There only 256 rules"
        assert rule_num > 0, "Number should be greater than 0"
        self.rule_num = rule_num

        self.rule = self._generate_rule()

        self.states = [[1,1,1], [1,1,0], [1,0,1], [1,0,0], [0,1,1], [0,1,0], [0,0,1], [0,0,0]]

