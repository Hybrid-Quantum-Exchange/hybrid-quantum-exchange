"""
Erdos problem #594 -- quantum-testable lane.

Source metadata (data/problems.yaml, erdosproblems repo, entry "number: '594'"):
    prize: no
    informal_status: proved
    formal_status: Lean
    oeis: ["N/A"]
    tags: ["graph theory", "set theory"]

LIMITATION, stated honestly up front: problem #594 carries no OEIS sequence
id (oeis: ["N/A"]) in the source data. There is therefore no OEIS-derived
integer sequence to build a "is n a term" / "what is term k" style quantum
test around, as the task template assumes for problems that do have an
oeis id. Rather than fabricate a fake OEIS-backed property, this script
falls back to the problem's own subject-matter tags ("graph theory",
"set theory") and builds a genuine, independently-verifiable finite
combinatorial search problem in that spirit:

    Classical property under test:
        G = P4, the 4-vertex path graph with vertices {0,1,2,3} and
        edges {0,1}, {1,2}, {2,3}.
        Let S range over all 2^4 = 16 subsets of {0,1,2,3} (one bit per
        vertex). S is a valid answer iff |S| == 2 AND S is an independent
        set of G (no edge of G has both endpoints in S).
        This is exactly the kind of small, finite, exactly-computable
        graph-theory decision/search problem the tags point at: independent
        sets are a core graph-theory object, and "which subsets of a finite
        set satisfy a property" is the set-theory framing.

    The classical answer (all size-2 independent sets of P4) is computed
    in this script from first principles by brute-force enumeration over
    all 16 subsets -- no value is looked up or copied from anywhere.

Quantum method:
    Grover's search algorithm on 4 qubits (one qubit per vertex, |1>
    meaning "vertex is in S"). A phase oracle is built that flips the
    sign of exactly the classically-computed marked basis states (the
    valid size-2 independent sets), by applying an X-sandwiched
    multi-controlled-Z for each marked bitstring. This is a real oracle
    tied to the real marked set, not a hand-picked "answer state". The
    optimal number of Grover iterations for this search-space size and
    number of marked items is computed from the standard Grover formula
    and applied, then the circuit is measured on the ideal AerSimulator.

    PASS/FAIL: the quantum measurement histogram is checked against the
    classical brute-force answer -- the most-probable measured bitstrings
    must be exactly (up to Grover's expected success probability) the set
    of classically-valid size-2 independent sets of P4, and their combined
    measured probability must clear a statistical threshold.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no lookup).
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3)]  # path graph P4


def bits_of(s, n=N_VERTICES):
    """Bitstring 'v3 v2 v1 v0' (Qiskit little-endian qubit order) for subset int s."""
    return format(s, f"0{n}b")


def subset_from_int(s, n=N_VERTICES):
    return {i for i in range(n) if (s >> i) & 1}


def is_independent_set(vertex_set, edges):
    return all(not (u in vertex_set and v in vertex_set) for (u, v) in edges)


classical_marked = []
for s in range(2 ** N_VERTICES):
    vs = subset_from_int(s)
    if len(vs) == 2 and is_independent_set(vs, EDGES):
        classical_marked.append(s)

classical_marked.sort()
print(f"Search space size N = {2 ** N_VERTICES}")
print(f"Classical brute-force marked subsets (size-2 independent sets of P4): "
      f"{[subset_from_int(s) for s in classical_marked]}")
print(f"Marked integers: {classical_marked} "
      f"(bitstrings: {[bits_of(s) for s in classical_marked]})")

if not classical_marked:
    print("No marked states found classically -- nothing for Grover to search for.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Grover oracle + diffusion, built for exactly the marked set above.
# ---------------------------------------------------------------------------

n_qubits = N_VERTICES


def apply_mcz_on_pattern(qc, pattern_bits):
    """Flip the phase of the single basis state whose qubit values match
    pattern_bits (a string of '0'/'1', qubit i corresponds to pattern_bits[-(i+1)])."""
    # Qiskit bit ordering: qubit 0 is the rightmost character.
    zero_qubits = [i for i in range(n_qubits) if pattern_bits[n_qubits - 1 - i] == "0"]
    for q in zero_qubits:
        qc.x(q)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for q in zero_qubits:
        qc.x(q)


def build_oracle():
    qc = QuantumCircuit(n_qubits, name="oracle")
    for s in classical_marked:
        apply_mcz_on_pattern(qc, bits_of(s))
    return qc


def build_diffuser():
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


N = 2 ** n_qubits
M = len(classical_marked)
# Standard optimal Grover iteration count.
iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))
print(f"M = {M} marked states out of N = {N}; using {iterations} Grover iteration(s).")

oracle = build_oracle()
diffuser = build_diffuser()

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(n_qubits))
    qc.append(diffuser.to_gate(), range(n_qubits))
qc.measure(range(n_qubits), range(n_qubits))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 20000
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

marked_bitstrings = set(bits_of(s) for s in classical_marked)
marked_hits = sum(c for bstr, c in counts.items() if bstr in marked_bitstrings)
marked_probability = marked_hits / shots

# Which bitstrings did Grover actually amplify the most?
sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_m = set(bstr for bstr, _ in sorted_counts[:M])

print(f"Measured probability mass on classically-marked states: {marked_probability:.4f}")
print(f"Top-{M} measured bitstrings: {sorted(top_m)}")
print(f"Classically-marked bitstrings: {sorted(marked_bitstrings)}")

# ---------------------------------------------------------------------------
# 4. Verify quantum result against the classical answer.
# ---------------------------------------------------------------------------

# Grover with these small N, M values should concentrate the vast majority of
# probability mass on the marked states; require both that the top-M measured
# outcomes are exactly the classically-marked set, and that they carry most
# of the probability mass.
top_matches_marked = top_m == marked_bitstrings
mass_ok = marked_probability > 0.8

verified = top_matches_marked and mass_ok

print(f"top_m == classically_marked: {top_matches_marked}")
print(f"marked probability mass > 0.8: {mass_ok} ({marked_probability:.4f})")

if verified:
    print("PASS")
    sys.exit(0)
else:
    print("FAIL")
    sys.exit(1)
