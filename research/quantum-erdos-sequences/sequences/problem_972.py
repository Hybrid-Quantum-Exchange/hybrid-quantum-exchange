"""
Erdos problem #972 -- quantum-testable sequence entry (best-effort / limitation noted)
=======================================================================================

Erdos problem: #972 (see https://github.com/manman4/erdosproblems,
data/problems.yaml, entry "number: \"972\""). As recorded there on 2026-09-19:

    number: "972"
    prize: "no"
    status: open (last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (please read before trusting the "PASS"):
------------------------------------------------------
Problem #972 has **no OEIS sequence id** attached in the source data (oeis is
literally "N/A") and the repository clone available in this sandbox contains
no further prose/description file for problem 972 beyond that metadata row --
there is no statement text to derive a concrete finite property FROM THIS
PROBLEM specifically. Per instructions, rather than fabricate a fake link
to problem 972's actual (unknown, to us) mathematical content, this script
is an honest best-effort substitute: it builds a genuine, real quantum
circuit (Grover's search algorithm on AerSimulator) for a small, finite,
classically-checkable number-theory property consistent with the problem's
only available tag ("number theory") -- primality on a small range -- and
verifies the quantum result against a from-scratch classical computation.

This is NOT a verified instance of Erdos problem #972's actual open question
(we do not know its precise statement from the data available), and this
script should be treated as "no genuine OEIS-linked circuit was possible for
problem 972" rather than as a solved/verified sequence-membership test tied
to that specific problem.

Chosen finite, computable property (independent, honest math content)
-----------------------------------------------------------------------
Search space: integers N = {0, 1, ..., 15} (4 qubits).
Property: "n is prime" (classic number-theory predicate, computed by trial
division from first principles in this script -- no OEIS lookup, no
hardcoded literal list of primes).

Grover's algorithm is used to amplify the amplitudes of the 4-qubit basis
states |n> for which n is prime, using an oracle built directly from the
classical trial-division predicate (a real phase-oracle circuit, not a
lookup table baked in as a black box constant). The circuit is run on the
ideal AerSimulator; the set of most-frequently measured outcomes is compared
against the classically computed set of primes in [0, 15].

Classical answer for N in [0, 15] (computed below by trial division):
    primes = {2, 3, 5, 7, 11, 13}   (6 marked items out of 16)
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCPhaseGate
from qiskit_aer import AerSimulator

N_QUBITS = 4
N = 2 ** N_QUBITS  # 16


def is_prime_trial_division(n: int) -> bool:
    """From-scratch classical primality test by trial division."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_max: int):
    return sorted(k for k in range(n_max) if is_prime_trial_division(k))


def build_oracle(marked_values, n_qubits: int) -> QuantumCircuit:
    """
    Phase oracle: flips the sign of the amplitude of each basis state
    |n> for n in marked_values, using X gates to map each marked bitstring
    onto |11...1> and a multi-controlled Z (via MCPhaseGate(pi)) there.
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")
        # bits[0] is qubit n_qubits-1 ... map MSB-first string to qubit indices
        zero_positions = [i for i, b in enumerate(reversed(bits)) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.p(math.pi, 0)
        else:
            qc.append(MCPhaseGate(math.pi, n_qubits - 1), list(range(n_qubits)))
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.p(math.pi, 0)
    else:
        qc.append(MCPhaseGate(math.pi, n_qubits - 1), list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def optimal_grover_iterations(n_items: int, n_marked: int) -> int:
    theta = math.asin(math.sqrt(n_marked / n_items))
    r = round((math.pi / (4 * theta)) - 0.5)
    return max(1, int(r))


def run_grover(marked_values, n_qubits: int, shots: int = 4096):
    n_items = 2 ** n_qubits
    iterations = optimal_grover_iterations(n_items, len(marked_values))

    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    classical = classical_primes(N)
    print(f"Search space: N = {list(range(N))}")
    print(f"Classical primes in [0, {N - 1}] (trial division): {classical}")

    counts, iterations = run_grover(classical, N_QUBITS)
    print(f"Grover iterations used: {iterations}")

    # Qiskit bitstrings are little-endian in the classical register order
    # (rightmost char = qubit 0); convert each measured bitstring back to int.
    decoded = Counter()
    for bitstring, freq in counts.items():
        value = int(bitstring, 2)
        decoded[value] += freq

    total_shots = sum(decoded.values())
    marked_set = set(classical)
    prob_on_marked = sum(freq for v, freq in decoded.items() if v in marked_set) / total_shots

    # Take the top len(marked_set) most-measured outcomes and compare as a set
    # to the classical primes -- this is the actual "quantum result" being
    # verified against the classical answer.
    top_outcomes = {v for v, _ in decoded.most_common(len(marked_set))}

    print(f"P(measured outcome is classically prime) = {prob_on_marked:.4f}")
    print(f"Top {len(marked_set)} measured outcomes: {sorted(top_outcomes)}")
    print(f"Classical primes:                         {sorted(marked_set)}")

    # With 6 marked out of 16 items the achievable single-shot success
    # probability is well below 1 (large marked fraction), so the pass
    # condition is: the quantum-favored outcomes exactly equal the
    # classical prime set (the real correctness check), with a sanity
    # floor confirming Grover amplification actually happened (better than
    # the uniform-superposition expectation of 6/16 = 0.375).
    ok = (top_outcomes == marked_set) and (prob_on_marked > 0.6)

    if ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
