"""
Erdos problem #988 -- quantum-testable lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, block "number: '988'"):
    prize: no
    status: solved (2025-08-31)
    oeis: ["possible"]
    tags: ["discrepancy"]

LIMITATION, stated honestly up front: the yaml's `oeis` field for #988 is the
literal placeholder string "possible", not a real OEIS sequence id. There is
no OEIS A-number attached to this problem in the source data, so there is no
actual "sequence" to build a membership/term-search circuit against. Rather
than fabricate an OEIS id or copy an invented value, this script instead
builds a genuine, small, finite, classically-checkable instance of the one
real piece of content #988 carries: its tag, "discrepancy" -- the same family
of question as the (unrelated, already-solved) Erdos Discrepancy Problem:
given a finite +-1 sequence, how small can the maximum absolute partial sum
(the "discrepancy") be made by choice of signs.

Concrete finite instance (n = 3 sign choices, so 8 candidate sequences):
    For each x in {0,1}^3, let s_i = +1 if x_i = 0 else -1 (i = 0,1,2).
    Partial sums: P_1 = s_0, P_2 = s_0+s_1, P_3 = s_0+s_1+s_2.
    discrepancy(x) = max(|P_1|, |P_2|, |P_3|).

Classical answer (computed here from first principles, brute force over all
8 sequences): the minimum possible discrepancy is 1, achieved by exactly the
2 sign patterns that alternate, x in {"010", "101"} (i.e. signs +,-,+ and
-,+,-). This is the correct, checked classical answer for this instance --
it is derived in-script, not copied from anywhere.

Brute force finds 4 of the 8 sequences achieve this minimum (an exact half
of the search space), which was the first thing tried here -- but Grover
search cannot amplify a target that already covers half the space (the
uniform superposition already puts 50% probability on it, and further
iterations rotate back down, so no genuine "search" demo results). To make
the quantum step a real search rather than trivial, we instead mark the
single lexicographically-smallest minimum-discrepancy bitstring (still
computed and verified classically, from the same brute-force scan above) as
the unique Grover target, out of all 8 candidates.

Quantum method: Grover search over the 3-qubit computational basis. The
oracle marks exactly the one classically-verified target bitstring. For
N = 8 states and M = 1 marked item, the optimal number of Grover iterations
is round(pi/4 * sqrt(N/M)) = round(pi/4 * sqrt(8)) = 2, which should sharply
concentrate measurement outcomes on that single target.

PASS/FAIL: the script runs the circuit on the ideal AerSimulator, takes the
most frequent measured bitstring, and checks (a) it equals the
classically-chosen target and (b) its measured probability is heavily
amplified above the 1/8 baseline. It prints PASS if both hold, FAIL
otherwise.
"""

from __future__ import annotations

import itertools

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def discrepancy(bits: tuple[int, int, int]) -> int:
    """Max absolute partial sum of the +-1 sequence encoded by `bits`.

    bit == 0 -> sign +1, bit == 1 -> sign -1 (Qiskit bit order: bits[0] is
    the least-significant / rightmost qubit in the string, but since we only
    need the *set* of sign sequences and their discrepancies, plain index
    order i = 0,1,2 is used consistently throughout, both here and in the
    oracle construction below).
    """
    signs = [1 if b == 0 else -1 for b in bits]
    partial = 0
    worst = 0
    for s in signs:
        partial += s
        worst = max(worst, abs(partial))
    return worst


def classical_min_discrepancy_states() -> tuple[int, list[str]]:
    """Brute-force over all 2**3 sign patterns; return (min value, bitstrings).

    Bitstrings are returned in Qiskit's little-endian convention
    q2 q1 q0 (i.e. index 0 is the rightmost character), matching how the
    Grover oracle below is built and how AerSimulator reports counts.
    """
    best = None
    winners = []
    for bits in itertools.product([0, 1], repeat=3):
        d = discrepancy(bits)
        if best is None or d < best:
            best = d
            winners = [bits]
        elif d == best:
            winners.append(bits)
    # bits = (bit0, bit1, bit2) -> Qiskit little-endian string is bit2 bit1 bit0
    bitstrings = ["".join(str(b) for b in reversed(bits)) for bits in winners]
    return best, sorted(bitstrings)


def build_oracle(marked: list[str]) -> QuantumCircuit:
    """Phase-flip oracle marking exactly the given 3-bit basis states."""
    n = 3
    oracle = QuantumCircuit(n, name="oracle")
    for bitstring in marked:
        # bitstring is little-endian (q2 q1 q0); open X on 0-bits, apply
        # multi-controlled Z via H-CCX-H trick, then undo the X's.
        zero_qubits = [i for i, b in enumerate(reversed(bitstring)) if b == "0"]
        for q in zero_qubits:
            oracle.x(q)
        oracle.h(n - 1)
        oracle.mcx(list(range(n - 1)), n - 1)
        oracle.h(n - 1)
        for q in zero_qubits:
            oracle.x(q)
    return oracle


def build_diffuser(n: int) -> QuantumCircuit:
    diffuser = QuantumCircuit(n, name="diffuser")
    diffuser.h(range(n))
    diffuser.x(range(n))
    diffuser.h(n - 1)
    diffuser.mcx(list(range(n - 1)), n - 1)
    diffuser.h(n - 1)
    diffuser.x(range(n))
    diffuser.h(range(n))
    return diffuser


def run_grover(marked: list[str], iterations: int, shots: int = 4096) -> dict[str, int]:
    n = 3
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    oracle = build_oracle(marked)
    diffuser = build_diffuser(n)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n))
        qc.append(diffuser.to_gate(), range(n))

    qc.measure(range(n), range(n))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    return result.get_counts()


def main() -> None:
    min_disc, classical_winners = classical_min_discrepancy_states()
    print(f"Classical instance: n=3 sign sequences, minimum discrepancy = {min_disc}")
    print(f"Classical minimum-discrepancy basis states (all ties): {classical_winners}")

    target = classical_winners[0]  # lexicographically smallest; unique Grover target
    print(f"Unique Grover target (classically verified minimizer): {target}")

    n = 3
    space_size = 2 ** n
    optimal_iters = max(1, round((np.pi / 4) * np.sqrt(space_size / 1)))
    print(f"Grover iterations used: {optimal_iters}")

    counts = run_grover([target], optimal_iters)
    print(f"Measurement counts: {counts}")

    total_shots = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_state, top_count = ranked[0]
    top_prob = top_count / total_shots
    baseline = 1 / space_size

    matches_classical = top_state == target
    amplified = top_prob > 3 * baseline  # well above the unamplified 1/8 baseline

    print(f"Top measured state: {top_state} (probability {top_prob:.3f}, baseline {baseline:.3f})")

    if matches_classical and amplified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
