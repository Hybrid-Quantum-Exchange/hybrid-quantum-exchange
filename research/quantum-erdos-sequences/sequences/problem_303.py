"""
Erdos problem #303 (see manman4/erdosproblems, data/problems.yaml).

Metadata as recorded there (2026-09-19 read of the repo):
    number: "303"
    status: proved (Lean)
    oeis:   ["N/A"]        <-- no OEIS sequence is attached to this problem
    tags:   ["number theory", "unit fractions"]

LIMITATION, stated honestly up front: problem #303 carries no OEIS id, so
there is no literal "OEIS sequence" for this script to test membership in.
Per the task instructions, when a problem has no OEIS id we still produce
our best honest attempt at a small, finite, computable property that is
genuinely in the spirit of the problem's tags ("unit fractions" / Egyptian
fractions), rather than fabricating or misattributing an OEIS id.

Chosen property (self-contained, derived here, not copied from anywhere):
    Unit-fraction ("Egyptian fraction") representations of 1.
    Candidate denominators: C = [2, 3, 4, 6, 12]  (5 elements -> 5 qubits,
    one qubit per candidate, bit=1 means "denominator included").
    A bitstring x in {0,1}^5 is a SOLUTION iff the subset of C it selects
    has reciprocals summing exactly to 1 (computed exactly with
    fractions.Fraction, first principles, in this script).

    This is a classic finite/computable search problem (Egyptian fraction
    decompositions of 1 from a bounded candidate set) squarely in the
    "unit fractions" territory of problem #303's tags, and it is small
    enough (2^5 = 32 states) for a genuine Grover search circuit.

Quantum approach:
    1. Classically enumerate all 32 subsets of C, compute exact reciprocal
       sums with Fraction, and record which subsets equal exactly 1.
    2. Build a genuine Grover oracle (multi-controlled phase flips) that
       marks exactly those solution bitstrings on 5 qubits, plus the
       standard Grover diffuser, iterated the (classically computed)
       near-optimal number of times for the actual number of marked items.
    3. Run the full circuit on AerSimulator, measure, and take the most
       frequent measured bitstrings.
    4. PASS iff the set of bitstrings Grover returns with high probability
       (probability mass concentrated on them) equals exactly the classical
       solution set computed in step 1.
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ----------------------------------------------------------------------
# Step 1: classical ground truth, computed from first principles here.
# ----------------------------------------------------------------------

CANDIDATES = [2, 3, 4, 6, 12]  # bit i (LSB-first) <-> CANDIDATES[i]
N = len(CANDIDATES)


def bitstring_to_subset(bits: str) -> list[int]:
    # bits[0] is qubit 0 (LSB), matching Qiskit's little-endian convention.
    return [CANDIDATES[i] for i, b in enumerate(bits) if b == "1"]


def is_unit_fraction_solution(bits: str) -> bool:
    subset = bitstring_to_subset(bits)
    if not subset:
        return False
    total = sum(Fraction(1, d) for d in subset)
    return total == Fraction(1, 1)


classical_solutions = sorted(
    format(x, f"0{N}b")[::-1]  # bit 0 first (LSB-first) to match circuit convention
    for x in range(2**N)
    if is_unit_fraction_solution(format(x, f"0{N}b")[::-1])
)

print("Candidate denominators:", CANDIDATES)
print("Classical brute-force search over all", 2**N, "subsets ...")
for bits in classical_solutions:
    subset = bitstring_to_subset(bits)
    frac_sum = sum(Fraction(1, d) for d in subset)
    print(f"  solution bitstring={bits}  subset={subset}  sum={frac_sum}")

if not classical_solutions:
    raise RuntimeError("no classical solutions found; candidate set is wrong")

M = len(classical_solutions)
print(f"Found {M} solution(s) out of {2**N} states.\n")

# ----------------------------------------------------------------------
# Step 2: build a real Grover oracle + diffuser marking exactly those
# classically-verified solution bitstrings.
# ----------------------------------------------------------------------


def apply_marking(qc: QuantumCircuit, qubits: list[int], bitstring: str) -> None:
    """Flip the phase of |bitstring> using X-gates + multi-controlled Z."""
    zero_positions = [i for i, b in enumerate(bitstring) if b == "0"]
    for i in zero_positions:
        qc.x(qubits[i])
    if N == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for i in zero_positions:
        qc.x(qubits[i])


def oracle(qc: QuantumCircuit, qubits: list[int]) -> None:
    for bits in classical_solutions:
        apply_marking(qc, qubits, bits)


def diffuser(qc: QuantumCircuit, qubits: list[int]) -> None:
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


# Near-optimal number of Grover iterations for M marked items out of 2^N.
theta = math.asin(math.sqrt(M / 2**N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Running Grover with {iterations} iteration(s) for M={M}, N=2^{N}={2**N}\n")

qc = QuantumCircuit(N, N)
qubits = list(range(N))
qc.h(qubits)
for _ in range(iterations):
    oracle(qc, qubits)
    diffuser(qc, qubits)
qc.measure(qubits, qubits)

# ----------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator.
# ----------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
job = backend.run(tqc, shots=shots)
counts = job.result().get_counts()

# Qiskit count keys are big-endian (qubit N-1 ... qubit 0); reverse to get
# our LSB-first bit-0-first convention back.
counts_lsb_first = {k[::-1]: v for k, v in counts.items()}

sorted_counts = sorted(counts_lsb_first.items(), key=lambda kv: -kv[1])
print("Top measured outcomes (bitstring: count):")
for bits, c in sorted_counts[:8]:
    flag = "  <-- classical solution" if bits in classical_solutions else ""
    print(f"  {bits}: {c}{flag}")
print()

# ----------------------------------------------------------------------
# Step 4: PASS/FAIL — the M most frequent measured bitstrings must equal
# exactly the classical solution set.
# ----------------------------------------------------------------------

quantum_top = set(bits for bits, _ in sorted_counts[:M])
classical_set = set(classical_solutions)

# Also require the marked states to actually carry the bulk of the
# probability mass (a real amplitude-amplification check, not just a
# top-M coincidence).
marked_mass = sum(c for b, c in counts_lsb_first.items() if b in classical_set) / shots

print(f"Quantum top-{M} bitstrings: {sorted(quantum_top)}")
print(f"Classical solution bitstrings: {sorted(classical_set)}")
print(f"Probability mass on classical solutions: {marked_mass:.3f}")

verified = (quantum_top == classical_set) and (marked_mass > 0.5)

if verified:
    print("PASS")
else:
    print("FAIL")
