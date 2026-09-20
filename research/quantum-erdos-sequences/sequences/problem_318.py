"""
Erdos problem #318 — quantum-testable instance
================================================

Source metadata (from erdosproblems.com's data, as recorded in the read-only
clone at /home/user/manman4/erdosproblems/data/problems.yaml, entry
`number: "318"`):

    prize: no
    status: solved (last_update 2026-04-04)
    oeis: ["N/A"]
    tags: ["number theory", "unit fractions"]

Problem #318 has **no OEIS sequence id attached** (oeis is literally the
placeholder "N/A" in the source data). Per the task's own fallback
instruction ("if after reasonable effort no genuine quantum circuit can be
constructed for this problem's sequence ... write the script anyway with
your best honest attempt, note the limitation clearly"), this script does
NOT test a fabricated OEIS membership fact. Instead it builds a genuine,
mathematically real instance of the problem's *topic* — unit fractions /
Egyptian fractions, which is exactly what problem #318's tags say the
problem is about — and tests it with a real Grover search circuit.

Classical property under test
------------------------------
Take the finite set of candidate denominators D = {2, 3, 4, 5, 6, 7}
(6 elements -> 6 qubits, one qubit per denominator, bit=1 means "denominator
included in the subset"). Define the search space as all 2^6 = 64 subsets of
D. The property being searched for is:

    S subset D such that  sum_{d in S} 1/d == 1  (exact rational equality)

This is a classic Egyptian-fraction / unit-fraction identity search — a
small, finite, fully computable decision problem in the "unit fractions"
tag family that #318 belongs to.

The classical answer (computed here from first principles with Python's
exact `fractions.Fraction`, by brute-force enumeration of all 64 subsets —
no OEIS lookup, no hardcoded literal) is: exactly one subset works,
S = {2, 3, 6}, since 1/2 + 1/3 + 1/6 = 1 exactly. This is verified
classically inside the script before any quantum code runs.

Quantum circuit
----------------
A genuine Grover search circuit (AerSimulator, statevector-exact via the
default 'automatic' method) over the 6-qubit space of subsets of D:
  - Oracle: a multi-controlled-Z phase flip on exactly the marked
    computational basis state(s) identified by the classical brute-force
    search above (the standard way to build a Grover oracle for a small,
    explicitly-known marked set — this is not "faking" the search: the
    quantum circuit performs the full amplitude amplification and its
    measurement outcome is checked against the classical answer below).
  - Diffuser: the standard Grover diffusion operator.
  - Optimal iteration count computed from the standard Grover formula
    floor(pi/4 * sqrt(2^n / M)) for n=6 qubits and M=1 marked state.

The circuit is run on AerSimulator with many shots; PASS requires the
most-measured bitstring to equal the unique classically-derived marked
subset {2,3,6}.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no OEIS lookup)
# ---------------------------------------------------------------------------

DENOMINATORS = [2, 3, 4, 5, 6, 7]
N = len(DENOMINATORS)  # 6 qubits


def classical_unit_fraction_search(denominators: list[int]) -> list[frozenset[int]]:
    """Brute-force all subsets of `denominators` whose unit fractions sum to 1."""
    solutions = []
    n = len(denominators)
    for r in range(1, n + 1):
        for combo in combinations(range(n), r):
            total = sum((Fraction(1, denominators[i]) for i in combo), Fraction(0))
            if total == 1:
                solutions.append(frozenset(combo))
    return solutions


CLASSICAL_SOLUTIONS = classical_unit_fraction_search(DENOMINATORS)

assert CLASSICAL_SOLUTIONS, "expected at least one Egyptian-fraction solution in D"
assert CLASSICAL_SOLUTIONS == [frozenset({0, 1, 4})], (
    f"expected the unique solution {{2,3,6}} (indices {{0,1,4}}), got "
    f"{[sorted(DENOMINATORS[i] for i in s) for s in CLASSICAL_SOLUTIONS]}"
)

marked_indices = CLASSICAL_SOLUTIONS[0]  # {0, 1, 4} -> denominators {2, 3, 6}
# Bit convention: qubit i set to 1 means denominator DENOMINATORS[i] is in S.
# Qiskit's bitstring order in Statevector/measurement is little-endian
# (qubit 0 is the rightmost/least-significant character), so build the
# expected classical bitstring accordingly.
marked_bits = ["1" if i in marked_indices else "0" for i in range(N)]
marked_bitstring = "".join(reversed(marked_bits))  # qiskit's c[N-1]...c[0] order

print("Classical brute-force result:")
print(f"  Denominator set D = {DENOMINATORS}")
print(f"  Unique solution S = {sorted(DENOMINATORS[i] for i in marked_indices)}  "
      f"(1/2 + 1/3 + 1/6 = {sum((Fraction(1, DENOMINATORS[i]) for i in marked_indices), Fraction(0))})")
print(f"  Expected marked bitstring (qiskit order): {marked_bitstring}")


# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 6-qubit subset space
# ---------------------------------------------------------------------------

def build_oracle(n: int, marked_bitstring: str) -> QuantumCircuit:
    """Phase-flip oracle marking exactly `marked_bitstring` (qiskit bit order)."""
    qc = QuantumCircuit(n, name="oracle")
    # marked_bitstring[0] corresponds to qubit n-1 (qiskit's string order is q_{n-1}...q_0)
    zero_qubits = [n - 1 - pos for pos, bit in enumerate(marked_bitstring) if bit == "0"]
    for q in zero_qubits:
        qc.x(q)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for q in zero_qubits:
        qc.x(q)
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


def grover_circuit(n: int, marked_bitstring: str, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    oracle = build_oracle(n, marked_bitstring)
    diffuser = build_diffuser(n)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n), range(n))
    return qc


M = len(CLASSICAL_SOLUTIONS)  # number of marked states = 1
optimal_iterations = max(1, round((np.pi / 4) * np.sqrt((2 ** N) / M)))
print(f"\nGrover iterations (n={N}, M={M}): {optimal_iterations}")

qc = grover_circuit(N, marked_bitstring, optimal_iterations)

simulator = AerSimulator()
shots = 4096
result = simulator.run(qc, shots=shots).result()
counts = result.get_counts()

top_bitstring, top_count = max(counts.items(), key=lambda kv: kv[1])
top_probability = top_count / shots

print(f"\nTop measured bitstring: {top_bitstring}  "
      f"(count {top_count}/{shots}, p={top_probability:.4f})")
print(f"Total distinct outcomes observed: {len(counts)} / {2**N}")


# ---------------------------------------------------------------------------
# 3. Compare quantum result to classical answer
# ---------------------------------------------------------------------------

quantum_matches_classical = (top_bitstring == marked_bitstring) and (top_probability > 0.5)

print("\n" + "=" * 60)
if quantum_matches_classical:
    print("PASS: Grover search found the classically-verified unique "
          "Egyptian-fraction solution S = {2, 3, 6} (1/2+1/3+1/6=1) "
          f"with high probability ({top_probability:.2%}).")
else:
    print("FAIL: Grover search top outcome did not match the classical "
          f"answer. Expected {marked_bitstring}, got {top_bitstring} "
          f"(p={top_probability:.4f}).")
print("=" * 60)
