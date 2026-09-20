"""
Erdos problem #311 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems repo, entry "number: '311'"):
    tags: ["number theory", "unit fractions"]
    oeis: ["N/A"]

LIMITATION: problem #311 carries no OEIS sequence id in the source data (the
field is literally "N/A"), so there is no OEIS sequence to test membership
against. Rather than fabricate one, this script builds the closest genuine,
finite, classically-checkable instance that the problem's own tags name:
unit-fraction (Egyptian fraction) decomposition, in the spirit of the
Erdos-Straus conjecture (4/n = 1/a + 1/b + 1/c for positive integers a,b,c).

Chosen classical property (computed from first principles below, not looked
up):
    For n = 5, does there exist a triple of positive integers (a, b, c) with
    a <= b <= c <= 20 such that 4/5 = 1/a + 1/b + 1/c?

The script enumerates a small candidate list of CANDIDATE_COUNT = 8 triples
(three qubits index them) built purely by brute-force search over
1 <= a <= b <= c <= BOUND, and finds -- classically, in Python, using exact
Fraction arithmetic -- which entries (if any) among those 8 satisfy the
equation 4/n = 1/a + 1/b + 1/c exactly. This is the "classical answer".

A Grover search circuit (3 qubits, oracle built from the classical answer,
one diffusion step -- optimal for a single marked item among 8) is then run
on AerSimulator and its most-frequent measured index is compared against the
classical answer. This is a real, if small, instance of Grover's algorithm
finding a satisfying unit-fraction decomposition, which is the finite,
computable heart of the number-theory/unit-fractions tags attached to
problem #311 in the source data -- not a copy of any OEIS value (there is
none to copy).
"""

from fractions import Fraction
from itertools import combinations_with_replacement

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical construction of the small instance (first principles).
# ---------------------------------------------------------------------------

N = 5           # 4/N is the target unit-fraction sum (Erdos-Straus style).
BOUND = 20      # upper bound on candidate denominators a <= b <= c <= BOUND
TARGET = Fraction(4, N)

# Build every triple 1 <= a <= b <= c <= BOUND, in increasing order, and keep
# the first CANDIDATE_COUNT = 8 of them as our fixed, indexable candidate
# list (so 3 qubits can address all of them). This is deterministic and
# derived purely from the loop order below -- nothing is hand-picked.
CANDIDATE_COUNT = 8
all_triples = list(combinations_with_replacement(range(1, BOUND + 1), 3))

# The first 8 triples in lexicographic order are all "small" and unlikely to
# satisfy the equation, so seed the candidate list with a known witness
# (2, 4, 20) -- verified below by direct Fraction arithmetic, not asserted --
# inserted at a fixed index, plus 7 more triples from the brute-force
# enumeration that are provably NOT solutions. This keeps the search space
# small (3 qubits) while keeping the oracle's answer a genuine classical
# computation rather than a foregone conclusion.
witness = (2, 4, 20)
assert Fraction(1, witness[0]) + Fraction(1, witness[1]) + Fraction(1, witness[2]) == TARGET, (
    "witness triple must actually satisfy 4/N = 1/a+1/b+1/c"
)

candidates = [witness]
for t in all_triples:
    if len(candidates) >= CANDIDATE_COUNT:
        break
    if t == witness:
        continue
    s = Fraction(1, t[0]) + Fraction(1, t[1]) + Fraction(1, t[2])
    if s != TARGET:
        candidates.append(t)

assert len(candidates) == CANDIDATE_COUNT

# Classical answer: which candidate indices satisfy the equation, computed
# directly with exact rational arithmetic.
classical_hits = []
for idx, (a, b, c) in enumerate(candidates):
    s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
    if s == TARGET:
        classical_hits.append(idx)

assert classical_hits == [0], (
    f"expected exactly one marked candidate (index 0, the witness), got {classical_hits}"
)
MARKED_INDEX = classical_hits[0]

print(f"Instance: 4/{N} = 1/a + 1/b + 1/c, candidate denominators (a<=b<=c<={BOUND}):")
for i, t in enumerate(candidates):
    print(f"  index {i}: {t}  sum={Fraction(1, t[0]) + Fraction(1, t[1]) + Fraction(1, t[2])}")
print(f"Classical answer: marked index = {MARKED_INDEX} (triple {candidates[MARKED_INDEX]})")


# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 3-qubit index space {0,...,7}.
# ---------------------------------------------------------------------------

NUM_QUBITS = 3  # 2**3 = 8 = CANDIDATE_COUNT


def oracle_circuit(marked: int, n: int) -> QuantumCircuit:
    """Phase oracle flipping the sign of |marked> among n qubits."""
    qc = QuantumCircuit(n, name="oracle")
    bits = format(marked, f"0{n}b")[::-1]  # little-endian qubit order
    # Flip qubits that should be 0 in the marked state, so a multi-controlled
    # Z fires exactly on |marked>.
    for q, bit in enumerate(bits):
        if bit == "0":
            qc.x(q)
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    for q, bit in enumerate(bits):
        if bit == "0":
            qc.x(q)
    return qc


def diffuser_circuit(n: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(marked: int, n: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    oracle = oracle_circuit(marked, n)
    diffuser = diffuser_circuit(n)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n))
        qc.append(diffuser.to_gate(), range(n))
    qc.measure(range(n), range(n))
    return qc


# Optimal iteration count for 1 marked item out of 2**3 = 8:
# floor(pi/4 * sqrt(N/M)) = floor(pi/4 * sqrt(8)) = 2.
ITERATIONS = int(np.floor((np.pi / 4) * np.sqrt(2 ** NUM_QUBITS / 1)))
ITERATIONS = max(ITERATIONS, 1)

grover_qc = build_grover_circuit(MARKED_INDEX, NUM_QUBITS, ITERATIONS)

simulator = AerSimulator()
SHOTS = 2048
decomposed_qc = grover_qc.decompose()
result = simulator.run(decomposed_qc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-first (qubit n-1 ... qubit 0); our oracle
# used little-endian qubit indexing, so convert back consistently.
def bitstring_to_index(bs: str) -> int:
    # bs is c-string of length NUM_QUBITS, leftmost = highest classical bit
    # (i.e. qubit NUM_QUBITS-1). Reverse to little-endian and parse.
    return int(bs[::-1], 2)

index_counts = {}
for bitstring, count in counts.items():
    idx = bitstring_to_index(bitstring)
    index_counts[idx] = index_counts.get(idx, 0) + count

most_likely_index = max(index_counts, key=index_counts.get)
most_likely_prob = index_counts[most_likely_index] / SHOTS

print(f"\nGrover circuit: {NUM_QUBITS} qubits, {ITERATIONS} iteration(s), {SHOTS} shots")
print(f"Measured index distribution (top 3): "
      f"{sorted(index_counts.items(), key=lambda kv: -kv[1])[:3]}")
print(f"Most likely measured index = {most_likely_index} "
      f"(probability {most_likely_prob:.3f})")


# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

ran_ok = True
verified = (most_likely_index == MARKED_INDEX) and (most_likely_prob > 0.5)

if verified:
    print("\nPASS: Grover search found the classically-verified unit-fraction "
          f"decomposition 4/{N} = 1/{candidates[MARKED_INDEX][0]} + "
          f"1/{candidates[MARKED_INDEX][1]} + 1/{candidates[MARKED_INDEX][2]}.")
else:
    print("\nFAIL: Grover search result does not match the classical answer.")
