import numpy as np
from src.task1 import Automata
import itertools
from collections import Counter
import matplotlib.pyplot as plt


class Enthropy_Gibbs_Automata(Automata):
    def __init__(
        self,
        width: int = 12,
        height: int = 50,
        reversable: bool = False,
        central: bool = True,
        rule_num: int = 110,
        initial_state=None,
    ):

        self.central = central
        self.rev = reversable

        assert width > 0 and height > 0, "Width and Height should be greater than 0"
        self.width = width
        self.height = height

        if initial_state is not None:
            assert isinstance(initial_state, list) and len(initial_state) == 12, (
                "Initial state should be list with 12 elements"
            )
            self.init_cond = initial_state
        else:
            self.init_cond = self._initial_condition()

        self.rule_num = rule_num

        self.rule = self._generate_rule()

        self.states = [
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
        self.all_possible_iterations = list(itertools.product([0, 1], repeat=12))

    def _calculate_entropy(self, arr):
        return Counter(tuple(row) for row in arr)

    @staticmethod
    def _shannon_entropy(counter):
        """Compute Shannon entropy S = -sum(p * log2(p)) from a Counter of states."""
        total = sum(counter.values())
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _run_single(self, init_state):
        """Run automaton from init_state, return list of rows (one per time step)."""
        if self.rev:
            # reversible needs two initial rows; use a fixed second row of zeros
            row0 = [0] * self.width
            row1 = list(init_state)
            rows = [row1]
            prev, curr = row0, row1
            for _ in range(self.height - 1):
                nxt = self._generate_row_reversable(curr, prev)
                rows.append(nxt)
                prev, curr = curr, nxt
        else:
            rows = [list(init_state)]
            for _ in range(self.height - 1):
                rows.append(self._generate_row(rows[-1]))
        return rows

    def compute_entropy_over_all_ics(self):
        all_ics = self.all_possible_iterations  # 4096 initial conditions

        # trajectories[ic_idx][t] = row at time t for that IC
        trajectories = [self._run_single(list(ic)) for ic in all_ics]

        entropy_t = []
        for t in range(self.height):
            # one row per IC at time t
            states_at_t = [trajectories[ic][t] for ic in range(len(all_ics))]
            counter = self._calculate_entropy(states_at_t)
            entropy_t.append(self._shannon_entropy(counter))

        return entropy_t


def plot_entropy_comparison(height=50):
    print("Computing S(t) for Rule 110...")
    aut110 = Enthropy_Gibbs_Automata(
        width=12, height=height, rule_num=110, reversable=False
    )
    s110 = aut110.compute_entropy_over_all_ics()

    print("Computing S(t) for Rule 122R...")
    aut122r = Enthropy_Gibbs_Automata(
        width=12, height=height, rule_num=122, reversable=True
    )
    s122r = aut122r.compute_entropy_over_all_ics()

    t = list(range(height))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].plot(t, s110, color="steelblue")
    axes[0].set_title("Rule 110 — S(t)")
    axes[0].set_xlabel("Time step t")
    axes[0].set_ylabel("Shannon entropy S(t) [bits]")
    axes[0].set_ylim(0, 12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, s122r, color="tomato")
    axes[1].set_title("Rule 122R — S(t)")
    axes[1].set_xlabel("Time step t")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        "Gibbs/Shannon Entropy over all 2¹² initial conditions (12 cells)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig("img/entropy_comparison.png", dpi=150)
    plt.show()


    return s110, s122r
