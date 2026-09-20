"""
Erdos problem #344 -- quantum-testable instance.

Source: erdosproblems.com data (problems.yaml), problem number "344".
Recorded metadata for #344:
    prize: no
    status: proved (as of 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "complete sequences"]

LIMITATION (read this first): the erdosproblems.com dataset lists no OEIS id
for problem #344 (oeis: ["N/A"]). There is therefore no concrete integer
sequence attached to this specific problem that can be looked up and tested
against a quantum circuit. Rather than fabricate an OEIS id or copy a value
with no real connection to #344, this script instead builds a genuine,
non-trivial quantum circuit around the one substantive piece of metadata that
*is* attached to the problem: the tag "complete sequences".

A sequence of positive integers S is called "complete" if every sufficiently
large integer can be written as a sum of a finite subset of distinct terms
of S. Completeness for a *finite* instance reduces to an exact, checkable,
finite property: "does some subset of S sum to target T?" -- which is
precisely the SUBSET-SUM decision problem, a canonical Grover search
instance with genuine mathematical content (NP search over 2^n candidate
subsets).

Concrete finite instance tested here:
    S = [1, 2, 3, 5]   (4 elements -> 4 index qubits, one bit per element)
    T = 6

Classical ground truth (computed from first principles below, by brute-force
enumeration of all 2^4 = 16 subsets -- no OEIS lookup, no hardcoding):
    the subsets of S summing exactly to T=6 are found by enumeration.

Quantum method: Grover's algorithm.
    - 4 "index" qubits represent the 2^4 = 16 possible subsets of S (bit i
      set <=> element i included).
    - An oracle, built directly from the classically-enumerated solution
      set, marks (phase-flips) exactly the computational basis states
      corresponding to subsets whose sum equals T.
    - The standard Grover diffusion operator amplifies those marked states.
    - The number of Grover iterations is computed from the true count of
      marked states (found classically) via the standard formula
      floor(pi/4 * sqrt(N/M)).
    - We run on the ideal AerSimulator, take the most frequently measured
      bitstring, decode it back to a subset, and check classically that its
      sum really equals T.

PASS/FAIL: the script prints PASS iff the quantum circuit's most-likely
measured subset is a genuine (classically verified) T-sum subset of S, i.e.
the quantum search recovers a real solution to the finite completeness
witness problem for this instance.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. The finite instance and its classical ground truth (first principles).
# ---------------------------------------------------------------------------

S = [1, 2, 3, 5]   # small set of positive integers, tag: "complete sequences"
TARGET = 6
N_ELEMENTS = len(S)          # number of index qubits
N_STATES = 2 ** N_ELEMENTS   # 16 candidate subsets


def subset_sum(bits: tuple[int, ...]) -> int:
    """Sum of S[i] for each bit set in `bits` (bits[0] = least-significant)."""
    return sum(s for s, b in zip(S, bits) if b)


def all_bitstrings(n: int):
    for combo in itertools.product([0, 1], repeat=n):
        yield combo  # combo[0] is qubit 0 (LSB), ..., combo[n-1] is qubit n-1


# Brute-force classical enumeration: which subsets sum exactly to TARGET?
classical_solutions = []
for bits in all_bitstrings(N_ELEMENTS):
    if subset_sum(bits) == TARGET:
        classical_solutions.append(bits)

assert len(classical_solutions) > 0, "instance has no solution; pick a different TARGET"

M = len(classical_solutions)  # number of marked states
print(f"S = {S}, TARGET = {TARGET}")
print(f"Classical brute force over all {N_STATES} subsets found {M} solution(s):")
for bits in classical_solutions:
    chosen = [s for s, b in zip(S, bits) if b]
    print(f"  bits={bits} -> subset {chosen}, sum={subset_sum(bits)}")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classically-found solution set.
# ---------------------------------------------------------------------------

def build_oracle(n: int, marked_states: list[tuple[int, ...]]) -> QuantumCircuit:
    """Phase-flip exactly the basis states in `marked_states` (multi-controlled Z)."""
    qc = QuantumCircuit(n, name="oracle")
    for bits in marked_states:
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(N_ELEMENTS, classical_solutions)
diffuser = build_diffuser(N_ELEMENTS)

n_iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M)))
print(f"Grover iterations used: {n_iterations} (N={N_STATES}, M={M})")

qc = QuantumCircuit(N_ELEMENTS, N_ELEMENTS)
qc.h(range(N_ELEMENTS))
for _ in range(n_iterations):
    qc.append(oracle.to_instruction(), range(N_ELEMENTS))
    qc.append(diffuser.to_instruction(), range(N_ELEMENTS))
qc.measure(range(N_ELEMENTS), range(N_ELEMENTS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
result = backend.run(compiled, shots=2048).result()
counts = result.get_counts()

# Qiskit's classical-register bit order is c[n-1]...c[0], i.e. reversed
# relative to our (qubit 0 = LSB) convention used above.
def bitstring_to_tuple(bs: str) -> tuple[int, ...]:
    return tuple(int(c) for c in bs[::-1])

most_likely_bs = max(counts, key=counts.get)
most_likely_bits = bitstring_to_tuple(most_likely_bs)
most_likely_subset = [s for s, b in zip(S, most_likely_bits) if b]
most_likely_sum = subset_sum(most_likely_bits)

print(f"Measurement counts (top 5): "
      f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
print(f"Most likely measured bitstring: {most_likely_bs} "
      f"-> subset {most_likely_subset}, sum={most_likely_sum}")

# Probability mass landing on ANY of the true marked states.
marked_prob = sum(
    v for k, v in counts.items() if bitstring_to_tuple(k) in classical_solutions
) / sum(counts.values())
print(f"Fraction of shots landing on a true solution state: {marked_prob:.3f}")


# ---------------------------------------------------------------------------
# 4. Verify quantum result against the classical answer.
# ---------------------------------------------------------------------------

is_correct_solution = most_likely_bits in classical_solutions
is_amplified = marked_prob > 0.5  # Grover should concentrate most mass on solutions

if is_correct_solution and is_amplified:
    print("PASS")
else:
    print("FAIL")
