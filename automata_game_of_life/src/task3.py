import numpy as np
from math import comb, log
import matplotlib.pyplot as plt
from src.task1 import Automata

def make_initial_condition(width: int = 500, n_black: int = 41) -> list:
    """
    Return a row of `width` cells: all white (0) except `n_black` black (1)
    cells centred in the middle.
    """
    row = [0] * width
    start = (width - n_black) // 2
    for i in range(n_black):
        row[start + i] = 1
    return row

def coarse_grained_entropy(trajectory: list, m: int = 5, T: int = 300) -> np.ndarray:
    height = len(trajectory)
    width = len(trajectory[0])
    n_groups = width // m  # number of complete groups

    log_omega = np.array(
        [log(comb(m, k), 2) if comb(m, k) > 0 else 0.0 for k in range(m + 1)]
    )

    row_entropy = np.zeros(height)
    for t, row in enumerate(trajectory):
        s = 0.0
        for j in range(n_groups):
            k_j = sum(row[j * m : (j + 1) * m])  # black cells in group j
            s += log_omega[k_j]
        row_entropy[t] = s

    S = np.convolve(row_entropy, np.ones(T) / T, mode="valid")
    return S

class CoarseGrainedAutomata(Automata):
    def __init__(
        self,
        width: int = 500,
        height: int = 15000,
        rule_num: int = 110,
        n_black: int = 41,
        reversable: bool = False,
    ):
        self.width = width
        self.height = height
        self.rule_num = rule_num
        self.rev = reversable
        self.central = False

        assert 0 < rule_num < 256, "Rule must be in 1..255"

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
        self.rule = self._generate_rule()

        row = make_initial_condition(width, n_black)
        self.init_cond = row  # used by parent's generate_automata
        self.init_cond2 = list(row)  # second row (for reversible)

    def generate_trajectory(self) -> list:
        if self.rev:
            row0 = self.init_cond
            row1 = self.init_cond2  # identical second row
            traj = [row1]
            prev, curr = row0, row1
            for _ in range(self.height - 1):
                nxt = self._generate_row_reversable(curr, prev)
                traj.append(nxt)
                prev, curr = curr, nxt
        else:
            traj = [list(self.init_cond)]
            for _ in range(self.height - 1):
                traj.append(self._generate_row(traj[-1]))
        return traj

    def run_and_plot(self, m: int = 5, T: int = 300, save: bool = True):
        """
        Generate trajectory, compute S(t), and plot.
        """
        traj = self.generate_trajectory()

        S = coarse_grained_entropy(traj, m=m, T=T)
        t = np.arange(len(S))

        label = f"Rule {self.rule_num}{'R' if self.rev else ''}"
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, S, lw=0.8, color="steelblue", label=label)
        ax.set_xlabel("Time step t")
        ax.set_ylabel(f"S(t)  [bits,  m={m}, T={T}]")
        ax.set_title(
            f"Coarse-grained entropy — {label}\n"
            f"N={self.width} cells, n_black={self.width // 2 - (self.width - 41) // 2 * 0 + 41}, "
            f"height={self.height}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            fname = (
                f"img/coarse_entropy_rule{self.rule_num}{'R' if self.rev else ''}.png"
            )
            plt.savefig(fname, dpi=150)
        plt.show()
        return traj, S



def compare_coarse_entropy(
    width: int = 500,
    height: int = 15000,
    n_black: int = 41,
    m: int = 5,
    T: int = 300,
    save: bool = True,
):
    results = {}
    configs = [
        dict(rule_num=110, reversable=False),
        dict(rule_num=122, reversable=True),
    ]

    for cfg in configs:
        aut = CoarseGrainedAutomata(width=width, height=height, n_black=n_black, **cfg)
        print(f"\nRule {cfg['rule_num']}{'R' if cfg['reversable'] else ''}")
        traj = aut.generate_trajectory()
        S = coarse_grained_entropy(traj, m=m, T=T)
        key = f"{cfg['rule_num']}{'R' if cfg['reversable'] else ''}"
        results[key] = S

    t110 = np.arange(len(results["110"]))
    t122r = np.arange(len(results["122R"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axes[0].plot(t110, results["110"], lw=0.8, color="steelblue")
    axes[0].set_title("Rule 110 (irreversible)")
    axes[0].set_xlabel("Time step t")
    axes[0].set_ylabel(f"S(t)  [bits,  m={m}, T={T}]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t122r, results["122R"], lw=0.8, color="tomato")
    axes[1].set_title("Rule 122R (reversible)")
    axes[1].set_xlabel("Time step t")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Coarse-grained entropy  |  N={width}, n_black={n_black}, m={m}, T={T}",
        fontsize=13,
    )
    plt.tight_layout()

    if save:
        fname = "img/coarse_entropy_comparison.png"
        plt.savefig(fname, dpi=150)
    plt.show()
    return results
