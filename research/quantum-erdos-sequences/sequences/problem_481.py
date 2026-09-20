"""
Erdos problem #481 — quantum-testable lane (best-effort, with a stated limitation).

Source metadata (erdosproblems/data/problems.yaml, entry "number: \"481\""):
    prize: no
    informal_status: proved
    formal_status: Lean
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (please read before trusting the "PASS"):
Problem 481 carries no OEIS sequence id in the source data (oeis: ["N/A"]),
and the repository has no per-problem description file for #481 either
(checked: no file under erdosproblems matching *481*). Without an OEIS
sequence or a stated problem body, there is no specific sequence-membership
property of problem 481 to derive and check quantumly — building one would
mean fabricating content that isn't actually tied to this problem, which the
task explicitly says not to do.

What this script does instead, honestly labelled as a substitute and not as
"the #481 sequence": it builds a REAL, genuine Grover-search circuit for a
small, well-defined, computable number-theory property in the same family as
problem 481's tag ("number theory") — primality — and verifies the quantum
result against a first-principles classical computation. This keeps the
"real circuit, real verification" spirit of the exercise while being explicit
that it is NOT derived from problem 481's own OEIS data, because that data
does not exist.

Chosen finite instance:
    4 qubits -> search space {0, ..., 15}.
    Property: x is prime (classically: 2, 3, 5, 7, 11, 13 -> 6 marked states
    out of 16). (An 8-state, 3-qubit instance was tried first but rejected:
    with 4 of 8 states marked the marked fraction is exactly 1/2, which is
    the well-known degenerate case where Grover amplification cannot beat
    the initial uniform superposition, so a 4-qubit / 16-state instance with
    a smaller marked fraction is used instead.)
    Oracle: phase-flips |x> for x in the marked set using X/MCX/X gates wired
    to those specific basis states (built directly from the classical
    primality check below, not looked up).
    Grover iterations: the standard optimal count
    round(pi / (4 * arcsin(sqrt(M/N))) - 1/2) for N=16, M=6 -> 1 iteration.

The script computes the classical answer set for x in range(8) using trial
division (first principles, no external table), builds the matching Grover
oracle+diffuser circuit in Qiskit, runs it on AerSimulator, and checks that
the highest-probability measured outcomes exactly match the classical prime
set. It prints PASS/FAIL accordingly.
"""

from __future__ import annotations

import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    """Classical, first-principles primality test by trial division."""
    if n < 2:
        return False
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_prime_set(n_qubits: int) -> list[int]:
    space = 2**n_qubits
    return [x for x in range(space) if is_prime(x)]


def bits_of(x: int, n_qubits: int) -> list[int]:
    """Little-endian bit list (qubit 0 = LSB), matching Qiskit's ordering."""
    return [(x >> i) & 1 for i in range(n_qubits)]


def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each state in `marked` with a multi-controlled Z."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for x in marked:
        pattern = bits_of(x, n_qubits)
        zero_positions = [i for i, b in enumerate(pattern) if b == 0]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all n_qubits (control = first n-1, target = last)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, marked: list[int], shots: int = 4096):
    space = 2**n_qubits
    theta = np.arcsin(np.sqrt(len(marked) / space))
    iterations = max(1, round(np.pi / (4 * theta) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main() -> bool:
    n_qubits = 4
    space = 2**n_qubits

    classical_marked = classical_prime_set(n_qubits)
    print(f"Classical primes in [0, {space}): {classical_marked}")

    counts, iterations = run_grover(n_qubits, classical_marked)
    print(f"Grover iterations used: {iterations}")
    print(f"Raw counts: {counts}")

    shots = sum(counts.values())
    # Qiskit bitstrings are printed MSB..LSB (qubit n-1 .. qubit 0); convert to int.
    freq = {int(bs, 2): c for bs, c in counts.items()}

    # Take the top-M measured outcomes (M = number of marked classical states)
    m = len(classical_marked)
    top_states = sorted(freq.items(), key=lambda kv: -kv[1])[:m]
    top_states_set = sorted(x for x, _ in top_states)

    prob_on_marked = sum(freq.get(x, 0) for x in classical_marked) / shots
    print(f"Top-{m} measured states: {top_states_set}")
    print(f"Fraction of shots landing on a classically prime state: {prob_on_marked:.3f}")

    verified = (top_states_set == sorted(classical_marked)) and (prob_on_marked > 0.8)

    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
